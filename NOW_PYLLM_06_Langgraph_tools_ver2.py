"""
PYLLM_06_Langgraph_tools2.py

학습 목표
1. LangGraph 에서 "정보 수집 -> 부족 정보 재질문 -> 조건 충족 시 다음 단계 실행" 구조를 익힌다.
2. 하나의 그래프 안에서 과일 작업 / 영화 작업 두 가지 흐름을 분기 처리하는 방법을 본다.
3. LLM 이 구조화된 데이터까지 직접 만들게 하는 with_structured_output() 사용법을 익힌다.

전체 LangGraph 흐름 시퀀스
START
-> init_system
-> add_user_message
-> analyze
-> 조건 분기
   1. ask_user
      : 정보가 부족하면 무엇이 부족한지 LLM 이 정리해서 다시 질문
   2. generate_fruit_tool_call
      : 과일 정보가 모두 모이면 합계 계산용 TOOL 호출
      -> run_tools
      -> finalize_fruit_answer
   3. generate_movie_answer
      : 영화 정보가 모두 모이면 새로운 영화와 시놉시스 생성
-> END

이 파일에서 처리하는 두 가지 대상
1. 과일
   - 필요한 정보:
     과일 종류 3가지
     각 과일의 개수
     각 과일의 색깔
   - 최종 작업:
     개수 총합 계산

2. 영화
   - 필요한 정보:
     영화 제목 4가지
     각 영화의 장르
     각 장르별 대표 감독 이름
   - 최종 작업:
     제목, 장르, 감독 요소를 섞어서 새로운 영화 1편 창조
     해당 영화의 시놉시스 생성
"""


"""
과일 완성 시 실제 순서

START
NODE_init_system
NODE_add_user_message
NODE_analyze
> ROUTE_after_analyze           : state["route"]를 읽음. 값이 "RUN_FRUIT_TOOL"이면 NODE_generate_fruit_tool_call로 보냄
NODE_generate_fruit_tool_call   : 실제 TOOL을 실행하는 건 아님. LLM에게: “정리된 과일 정보는 이거다” / “sum_fruit_counts TOOL을 호출해라” 라고 시킴
> ROUTE_run_tools               : “모델이 진짜 TOOL 호출 요청을 만들었는지” 확인 -  마지막 메시지를 보고 AIMessage이고 tool_calls가 있으면 "GOTO run_tools" 반환
NODE_run_tools                  : 여기서 실제 파이썬 TOOL 함수 실행. 결과는 ToolMessage(...)
NODE_finalize_fruit_answer      : 최종 자연어 답변 생성 : ToolMessage를 읽음. content 안 JSON 문자열을 json.loads(...) 해서 dict 복원. 그리고 summary_text + total을 LLM에 보내서 최종 자연어 답변. 
END
"""

"""
영화 완성 시 실제 순서
START
-> init_system
-> add_user_message
-> analyze
-> ROUTE_after_analyze
-> generate_movie_answer
-> END
"""

# ---------------------------------------------------------------------------
# [1] 기본 import
# ---------------------------------------------------------------------------
import json
import os

from typing import Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# [1-1] 디버깅용 invoke 메시지 출력 스위치
# ---------------------------------------------------------------------------
DEBUG_SHOW_INVOKE_MESSAGES = False


def DEBUG_print_invoke_messages(node_name, message_list):
    """
    노드 내부에서 model.invoke([...]) 로 보내는
    임시 SystemMessage / HumanMessage 목록을 출력한다.

    중요:
    이것들은 State["messages"] 에 누적 저장되는 대화가 아니라,
    해당 노드 안에서 그 순간만 LLM 에 보내는 일시적 입력이다.
    """

    if not DEBUG_SHOW_INVOKE_MESSAGES:
        return

    print(f"     ■ [INVOKE INPUT @ {node_name}]")
    for idx, msg in enumerate(message_list):
        print(f"        □ MSG #{idx} [{type(msg).__name__}]")
        print(f"            {msg.content}")


# ---------------------------------------------------------------------------
# [2] State 정의
# ---------------------------------------------------------------------------
# LangGraph 는 노드 사이를 오갈 때 State 딕셔너리를 공유한다.
#
# user_input:
#   이번 턴 사용자가 새로 입력한 원문
#
# messages:
#   대화 메시지 누적 리스트
#   add_messages 를 붙였기 때문에 새 메시지를 덮어쓰지 않고 뒤에 계속 이어붙인다.
#
# target_type:
#   현재 작업 대상
#   "fruit" 또는 "movie" 또는 아직 미정이면 ""
#
# collected_items:
#   지금까지 LLM 이 정리해 둔 구조화 데이터
#   fruit 인 경우 예시:
#   [
#       {"name": "사과", "count": 2, "color": "빨강"},
#       {"name": "바나나", "count": 3, "color": "노랑"},
#   ]
#
#   movie 인 경우 예시:
#   [
#       {"title": "인터스텔라", "genre": "SF", "director": "크리스토퍼 놀란"},
#       {"title": "기생충", "genre": "드라마", "director": "봉준호"},
#   ]
#
# route:
#   analyze 노드가 다음에 어디로 갈지 정한 결과값
#   "ASK_USER" / "RUN_FRUIT_TOOL" / "MAKE_MOVIE"
#
# missing_message:
#   정보가 부족할 때 사용자에게 다시 물어볼 질문 문장
#
# summary_text:
#   현재까지 정리한 정보를 사람이 읽기 좋게 정리한 문장
#
# fruit_counts:
#   과일 모드에서 TOOL 로 넘길 숫자 목록
class State(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    target_type: str
    collected_items: list[dict]
    route: str
    missing_message: str
    summary_text: str
    fruit_counts: list[int]


# ---------------------------------------------------------------------------
# [3] Structured Output 스키마
# ---------------------------------------------------------------------------
# BaseModel:
#   Pydantic 이 제공하는 "데이터 형식 설계도"
#   LLM 이 어떤 구조로 답을 만들어야 하는지 명확하게 알려줄 수 있다.
#
# Field(...):
#   각 필드에 대한 설명을 붙이는 용도
#   이 설명도 LLM 이 structured output 을 만들 때 참고한다.
#
# 이 파일에서 중요한 포인트:
#   with_structured_output(AnalyzeResult) 를 쓰면
#   LLM 이 자유문장 대신 AnalyzeResult 형태에 맞춰 결과를 내도록 강하게 유도할 수 있다.
#
#   즉,
#   "분석 결과를 그냥 텍스트로 쓰지 말고
#    target_type, route, collected_items, missing_message ... 칸에 맞춰 넣어라"
#   라고 시키는 방식이다.


class AnalyzeResult(BaseModel):
    """
    analyze 노드에서 LLM 이 반환해야 하는 결과 구조.

    처음 보면 낯설 수 있지만, 그냥
    "LLM 응답을 담을 설계도"라고 생각하면 된다.

    이 구조를 미리 정해두면 좋은 점:
    1. LLM 응답을 사람이 다시 문자열 파싱할 필요가 줄어든다.
    2. 다음 노드가 어떤 값을 믿고 써야 하는지 분명해진다.
    3. route 같은 분기값도 LLM 이 명확하게 내놓게 만들 수 있다.
    """

    target_type: str = Field(
        default="",
        description="현재 작업 대상. fruit 또는 movie 중 하나.",
    )

    collected_items: list[dict] = Field(
        default_factory=list,
        description=(
            "현재까지 정리된 전체 항목 목록. "
            "fruit 이면 name/count/color 구조, "
            "movie 이면 title/genre/director 구조."
        ),
    )

    route: str = Field(
        default="ASK_USER",
        description=(
            "다음 분기. "
            "정보가 부족하면 ASK_USER, "
            "과일 정보가 완성되면 RUN_FRUIT_TOOL, "
            "영화 정보가 완성되면 MAKE_MOVIE."
        ),
    )

    missing_message: str = Field(
        default="",
        description="정보가 부족할 때 사용자에게 보여줄 한국어 질문 문장.",
    )

    summary_text: str = Field(
        default="",
        description="현재까지 정리된 상태를 짧게 요약한 한국어 문장.",
    )

    fruit_counts: list[int] = Field(
        default_factory=list,
        description="fruit 모드일 때만 사용. TOOL 로 넘길 과일 개수 목록.",
    )


# ---------------------------------------------------------------------------
# [4] TOOL 정의
# ---------------------------------------------------------------------------
# 실제 합계 계산은 TOOL 이 담당한다.
# LLM 은 계산 자체보다 "언제 TOOL 을 호출할지"를 판단하는 역할을 맡는다.
@tool
def sum_fruit_counts(counts):
    """과일 개수 목록을 받아 총합을 계산한다."""

    total = sum(counts)

    # TOOL 함수는 가능한 한 파이썬 객체 자체를 반환하는 쪽이 깔끔하다.
    # 여기서는 dict 를 그대로 반환하고,
    # 나중에 ToolMessage.content 에 넣을 때만 문자열(JSON)로 변환한다.
    return {
        "counts": counts,
        "total": total,
    }


# 모델에 바인딩할 TOOL 목록
MY_TOOLS = [sum_fruit_counts]


# ---------------------------------------------------------------------------
# [5] 모델 생성
# ---------------------------------------------------------------------------
def create_model():
    """OpenAI 채팅 모델 객체 생성."""

    return ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# [6] 시스템 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
너는 학습용 정보 수집 챗봇이다.

반드시 두 작업 중 하나만 처리한다.
1. fruit 작업
2. movie 작업

fruit 작업 규칙
- 과일 종류는 정확히 3가지 필요하다.
- 각 과일마다 이름, 개수, 색깔이 필요하다.
- 정보가 부족하면 부족한 정보만 짚어서 다시 물어본다.
- 정보가 모두 모이면 과일 개수 총합을 계산한다.

movie 작업 규칙
- 영화 제목은 정확히 4개 필요하다.
- 각 영화마다 장르가 필요하다.
- 각 장르별 대표 감독 이름이 필요하다.
- 정보가 모두 모이면 제목, 장르, 감독 요소를 섞어 새로운 영화 1편을 창조한다.
- 마지막에는 새 영화 제목과 시놉시스를 말한다.

중요 규칙
- 사용자가 fruit 또는 movie 중 무엇을 할지 아직 안 정했다면 먼저 그 대상부터 확인한다.
- 정보가 부족할 때는 최종 결과를 만들지 말고 부족한 정보만 질문한다.
- 추측으로 빈칸을 채우지 마라.
- 사용자가 중간에 fruit 와 movie 를 바꾸면 이전 대상 정보는 버리고 새 대상으로 다시 시작한다.
""".strip()


ANALYZE_PROMPT = """
너는 현재 대화 상태를 보고 다음 행동을 결정하는 분석기다.

입력으로 주어지는 값
1. current_target_type
2. current_collected_items
3. user_input

너의 할 일
1. 현재 대상이 fruit 인지 movie 인지 결정한다.
2. 이전에 모아둔 정보와 이번 입력을 합쳐 최신 collected_items 를 만든다.
3. 사용자가 대상을 바꾸면 이전 대상 정보는 버리고 새 대상 기준으로 다시 정리한다.
4. 정보가 부족하면 route 를 ASK_USER 로 두고 missing_message 를 자연스럽게 작성한다.
5. fruit 정보가 완성되면 route 를 RUN_FRUIT_TOOL 로 두고 fruit_counts 를 채운다.
6. movie 정보가 완성되면 route 를 MAKE_MOVIE 로 둔다.

fruit 완성 조건
- 과일 3개
- 각 과일마다 name, count, color 존재

movie 완성 조건
- 영화 4개
- 각 영화마다 title, genre, director 존재

중요
- 답은 반드시 AnalyzeResult 구조로 채운다.
- 부족한 정보 질문은 한국어로 짧고 분명하게 작성한다.
- summary_text 는 현재까지 정리된 내용을 사람이 읽기 좋게 써라.
""".strip()


FRUIT_TOOL_PROMPT = """
정리된 fruit 정보가 모두 완성되었다.
반드시 sum_fruit_counts TOOL 을 정확히 한 번 호출하라.
TOOL 호출 전에는 최종 자연어 답변을 쓰지 마라.
""".strip()


FRUIT_FINAL_PROMPT = """
너는 TOOL 계산 결과를 받아 최종 답변을 만드는 챗봇이다.
과일 3가지의 이름, 색깔, 개수와 총합을 한국어로 짧게 정리하라.
""".strip()


MOVIE_FINAL_PROMPT = """
너는 창작 영화 기획 챗봇이다.

입력으로 받은 4개의 영화 제목, 각 영화의 장르, 각 장르별 대표 감독 이름을 참고해서
그 요소들을 섞은 새로운 영화 1편을 창조하라.

출력 규칙
1. 새 영화 제목
2. 한 줄 콘셉트
3. 4~6문장 정도의 시놉시스

한국어로 작성하라.
""".strip()


# ---------------------------------------------------------------------------
# [7] TOOL 실행 NODE
# ---------------------------------------------------------------------------
# 클래스 형태로 만든 이유:
#   LangGraph 에서는 "호출 가능한 객체"도 노드로 넣을 수 있다.
#   그래서 __call__ 을 구현하면 클래스를 함수처럼 그래프 노드에 등록할 수 있다.
#
# 이 방식은 PYLLM_05_Langgraph_tools.ipynb 와 같은 구조이다.
class NODE_run_tools:
    """마지막 AIMessage 의 tool_calls 를 읽어서 실제 TOOL 을 실행하는 노드"""

    def __init__(self, tools):
        # tool.name 을 key 로 두고 실제 tool 객체를 value 로 둔다.
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state):
        """AI 가 요청한 TOOL 들을 순회 실행하고 ToolMessage 로 반환한다."""

        messages = state.get("messages", [])
        last_message = messages[-1]

        ToolResults = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool_result = self.tools_by_name[tool_name].invoke(tool_args)

            ToolResults.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_name,
                    tool_call_id=tool_id,
                )
            )

        return {"messages": ToolResults}


# ---------------------------------------------------------------------------
# [8] NODE 정의
# ---------------------------------------------------------------------------
def NODE_init_system(state):
    """첫 턴에만 SystemMessage 를 messages 맨 앞에 추가한다."""

    messages = state.get("messages", [])

    if messages and isinstance(messages[0], SystemMessage):
        return {}

    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
    }


def NODE_add_user_message(state):
    """이번 턴의 사용자 입력을 HumanMessage 로 변환해 messages 에 추가한다."""

    return {
        "messages": [
            HumanMessage(content=state["user_input"]),
        ]
    }


def NODE_analyze(state):
    """
    현재까지의 누적 정보와 이번 user_input 을 함께 보고
    LLM 이 다음 행동을 구조화된 값으로 결정하게 하는 노드.

    with_structured_output(AnalyzeResult) 설명
    - 보통 LLM 은 자유 문장으로 답한다.
    - 그런데 이 메서드를 쓰면 "자유문장 말고 AnalyzeResult 형태로 답해"라고 강하게 유도한다.
    - 그래서 아래 코드의 result 는 그냥 문자열이 아니라
      AnalyzeResult 구조를 따르는 객체처럼 다룰 수 있다.
    - 즉, result.target_type / result.route / result.collected_items 같은 식으로 값을 꺼낼 수 있다.
    """

    model = create_model().with_structured_output(AnalyzeResult)
    #model = ChatOpenAI(model="gpt-4o").with_structured_output(AnalyzeResult)

    current_target_type = state.get("target_type", "")
    current_collected_items = state.get("collected_items", [])
    user_input = state["user_input"]

    invoke_messages = [
        SystemMessage(content=ANALYZE_PROMPT),
        HumanMessage(
            content=(
                f"current_target_type:\n{current_target_type}\n\n"
                f"current_collected_items:\n{json.dumps(current_collected_items, ensure_ascii=False)}\n\n"
                f"user_input:\n{user_input}\n"
            )
        ),
    ]

    DEBUG_print_invoke_messages("NODE_analyze", invoke_messages)
    result = model.invoke(invoke_messages)

    return {
        "target_type": result.target_type,
        "collected_items": result.collected_items,
        "route": result.route,
        "missing_message": result.missing_message,
        "summary_text": result.summary_text,
        "fruit_counts": result.fruit_counts,
    }


def NODE_ask_user(state):
    """정보가 부족할 때 LLM 이 만든 missing_message 를 그대로 사용자에게 보여주는 노드."""

    text = state.get("missing_message", "")

    if not text:
        text = "fruit 또는 movie 중 무엇을 할지와 필요한 정보를 알려주세요."

    return {
        "messages": [
            AIMessage(content=text),
        ]
    }


def NODE_generate_fruit_tool_call(state):
    """과일 정보가 완성되었을 때 sum_fruit_counts TOOL 호출용 AIMessage 를 만드는 노드."""

    model = create_model().bind_tools(MY_TOOLS)

    invoke_messages = [
        SystemMessage(content=FRUIT_TOOL_PROMPT),
        HumanMessage(
            content=(
                f"정리된 fruit 정보:\n{state.get('summary_text', '')}\n"
                f"개수 목록:\n{state.get('fruit_counts', [])}"
            )
        ),
    ]

    DEBUG_print_invoke_messages("NODE_generate_fruit_tool_call", invoke_messages)
    response = model.invoke(invoke_messages)

    return {"messages": [response]}


# 
def NODE_finalize_fruit_answer(state):
    """TOOL 계산 결과를 읽고 과일 최종 답변을 자연어로 만드는 노드."""

    tool_message = state["messages"][-1]
    tool_payload = json.loads(tool_message.content)
    total = tool_payload["total"]

    model = create_model()

    # 지금까지 누적된 전체 messages를 그대로 다시 보내는 방식은 아니고,
    # state에서 뽑은 일부 정보만 골라서 일시적으로 LLM에 보내는 것
    invoke_messages = [
        SystemMessage(content=FRUIT_FINAL_PROMPT),
        HumanMessage(
            content=(
                f"정리된 fruit 정보:\n{state.get('summary_text', '')}\n"
                f"총합:\n{total}"
            )
        ),
    ]

    DEBUG_print_invoke_messages("NODE_finalize_fruit_answer", invoke_messages)
    response = model.invoke(invoke_messages)

    return {"messages": [response]}


def NODE_generate_movie_answer(state):
    """영화 정보가 완성되었을 때 새로운 영화를 창조하고 시놉시스를 만드는 노드."""

    model = create_model()

    invoke_messages = [
        SystemMessage(content=MOVIE_FINAL_PROMPT),
        HumanMessage(
            content=(
                f"정리된 movie 정보:\n{state.get('summary_text', '')}\n"
                f"구조화 데이터:\n{json.dumps(state.get('collected_items', []), ensure_ascii=False)}"
            )
        ),
    ]

    DEBUG_print_invoke_messages("NODE_generate_movie_answer", invoke_messages)
    response = model.invoke(invoke_messages)

    return {"messages": [response]}


# ---------------------------------------------------------------------------
# [9] 라우팅 함수
# ---------------------------------------------------------------------------
def ROUTE_after_analyze(state):
    """
    analyze 노드의 결과(route 값)를 보고 다음 노드를 결정한다.

    가능한 반환값
    1. "ASK_USER"
    2. "RUN_FRUIT_TOOL"
    3. "MAKE_MOVIE"
    """

    route = state.get("route", "ASK_USER")

    if route == "RUN_FRUIT_TOOL":
        return "RUN_FRUIT_TOOL"
    elif route == "MAKE_MOVIE":
        return "MAKE_MOVIE"
    else:
        return "ASK_USER"


def ROUTE_run_tools(state):
    """
    generate_fruit_tool_call 의 출력(AIMessage)을 보고
    실제 TOOL 실행이 필요한지 판단한다.
    """

    messages = state.get("messages", [])
    if not messages:
        return "DONE"

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "GOTO run_tools"
    else:
        return "DONE"


# ---------------------------------------------------------------------------
# [10] 그래프 조립
# ---------------------------------------------------------------------------
def MY_build_graph(checkpointer=None):
    """
    그래프 구조

    START
    -> init_system
    -> add_user_message
    -> analyze
       -> ask_user -> END
       -> generate_fruit_tool_call -> run_tools -> finalize_fruit_answer -> END
       -> generate_movie_answer -> END
    """

    # 1. 빌더 생성
    graph_builder = StateGraph(State)

    # 2. 노드 등록
    graph_builder.add_node("init_system", NODE_init_system)
    graph_builder.add_node("add_user_message", NODE_add_user_message)
    graph_builder.add_node("analyze", NODE_analyze)
    graph_builder.add_node("ask_user", NODE_ask_user)
    graph_builder.add_node("generate_fruit_tool_call", NODE_generate_fruit_tool_call)
    graph_builder.add_node("run_tools", NODE_run_tools(MY_TOOLS))
    graph_builder.add_node("finalize_fruit_answer", NODE_finalize_fruit_answer)
    graph_builder.add_node("generate_movie_answer", NODE_generate_movie_answer)

    # 3. 노드 연결
    graph_builder.add_edge(START, "init_system")
    graph_builder.add_edge("init_system", "add_user_message")
    graph_builder.add_edge("add_user_message", "analyze")

    # analyze -> (ask_user / generate_fruit_tool_call / generate_movie_answer)
    graph_builder.add_conditional_edges(
        "analyze",
        ROUTE_after_analyze,
        {
            "ASK_USER": "ask_user",
            "RUN_FRUIT_TOOL": "generate_fruit_tool_call",
            "MAKE_MOVIE": "generate_movie_answer",
        },
    )

    # generate_fruit_tool_call -> (run_tools / END)
    graph_builder.add_conditional_edges(
        "generate_fruit_tool_call",
        ROUTE_run_tools,
        {
            "GOTO run_tools": "run_tools",
            "DONE": END,
        },
    )

    graph_builder.add_edge("run_tools", "finalize_fruit_answer")
    graph_builder.add_edge("ask_user", END)
    graph_builder.add_edge("finalize_fruit_answer", END)
    graph_builder.add_edge("generate_movie_answer", END)

    # 4. 빌더 컴파일
    MY_graph = graph_builder.compile(checkpointer=checkpointer)

    return MY_graph


# ---------------------------------------------------------------------------
# [11] 실행 함수
# ---------------------------------------------------------------------------
def run_chatbot(memory, thread_id):
    """MemorySaver 와 thread_id 를 사용해 상태가 이어지는 대화형 챗봇 실행."""

    graph = MY_build_graph(checkpointer=memory)

    # thread_id 가 같으면 같은 대화 스레드로 간주되어 상태가 이어진다.
    config_now = {"configurable": {"thread_id": thread_id}}

    print("대화형 챗봇을 시작합니다. 종료하려면 q 를 입력하세요.")
    print("예시1: fruit 로 할게. 사과 2개 빨강, 바나나 3개 노랑")
    print("예시2: movie 로 할게. 인터스텔라 SF 놀란, 기생충 드라마 봉준호")

    while True:
        user_input = input("[User] : ").strip()

        if user_input.lower() == "q":
            print("챗봇을 종료합니다.")
            break

        if not user_input:
            continue

        for event in graph.stream(
            {
                "user_input": user_input,
                "messages": [],
                "target_type": "",
                "collected_items": [],
                "route": "",
                "missing_message": "",
                "summary_text": "",
                "fruit_counts": [],
            },
            config=config_now,
            stream_mode="values",
        ):
            messages = event.get("messages", [])
            if not messages:
                continue

            last_message = messages[-1]

            if isinstance(last_message, AIMessage) and last_message.content:
                last_message.pretty_print()


# ---------------------------------------------------------------------------
# [12] 디버깅 함수
# ---------------------------------------------------------------------------
# 디버깅 함수 01.
# 한 번의 질문을 넣고 graph.invoke() 결과의 최종 답변만 확인
def ask_graph(question: str):
    """질문 1개를 넣고 최종 AIMessage.content 만 확인하는 디버깅 함수."""

    graph = MY_build_graph()

    result = graph.invoke(
        {
            "user_input": question,
            "messages": [],
            "target_type": "",
            "collected_items": [],
            "route": "",
            "missing_message": "",
            "summary_text": "",
            "fruit_counts": [],
        }
    )

    final_message = result["messages"][-1]
    print(f"[FINAL_ANSWER] : {final_message.content}")

    return final_message.content


# 디버깅 함수 02.
# Stream 방식 실행 - 각 step 에서의 State 와 추가 정보 출력
def stream_graph(question: str):
    """
    질문 1개를 넣고 graph.stream(..., stream_mode='values') 로
    각 step 의 전체 state 변화를 순서대로 확인하는 디버깅 함수.

    이 함수에서 보는 핵심 포인트
    1. messages 가 어떻게 누적되는지
    2. analyze 뒤에 route 가 어떻게 바뀌는지
    3. AIMessage(tool_calls=...) 가 언제 생기는지
    4. ToolMessage 가 언제 생기고 무엇을 담는지
    5. target_type / collected_items / summary_text 가 step 마다 어떻게 바뀌는지
    """

    global DEBUG_SHOW_INVOKE_MESSAGES

    graph = MY_build_graph()

    step_no = 0
    DEBUG_SHOW_INVOKE_MESSAGES = True

    try:
        for event in graph.stream(
            {
                "user_input": question,
                "messages": [],
                "target_type": "",
                "collected_items": [],
                "route": "",
                "missing_message": "",
                "summary_text": "",
                "fruit_counts": [],
            },
            stream_mode="values",
        ):
            step_no += 1

            messages = event.get("messages", [])
            number_message = len(messages)

            if number_message == 0:
                last_message = None
                type_message = "EMPTY"
                last_message_content = ""
            else:
                last_message = messages[-1]
                type_message = type(last_message).__name__
                last_message_content = last_message.content

            print(f"\n[STEP #{step_no}]")
            print(f"[#{number_message}][{type_message}] : {last_message_content}")

            # -------------------------------------------------------------------
            # State 요약 출력
            # -------------------------------------------------------------------
            print("     ■ [STATE]")
            print(f"        □ target_type    : {event.get('target_type', '')}")
            print(f"        □ route          : {event.get('route', '')}")
            print(f"        □ fruit_counts   : {event.get('fruit_counts', [])}")
            print(f"        □ missing_message: {event.get('missing_message', '')}")
            print(f"        □ summary_text   : {event.get('summary_text', '')}")
            print(f"        □ collected_items: {event.get('collected_items', [])}")

            # AI Tool Call 정보 for NODE_run_tools
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                print("     ■ [AI's TOOL CALLS]")
                for idx, tool_call in enumerate(last_message.tool_calls):
                    print(f"        □ TOOL #{idx}")
                    print(f"            □ NAME(input) : {tool_call['name']}({tool_call['args']})")
                    print(f"            □ ID          : {tool_call['id']}")

            # ToolMessage 정보 for NODE_finalize_fruit_answer
            if last_message is not None and isinstance(last_message, ToolMessage):
                print("     ■ [TOOL Result]")
                print(f"        □ NAME   : {last_message.name}")
                print(f"        □ output : {last_message.content}")
                print(f"        □ ID     : {last_message.tool_call_id}")
    finally:
        DEBUG_SHOW_INVOKE_MESSAGES = False


# 디버깅 함수 03.
# MemorySaver 를 붙여 여러 턴 누적 상태를 보면서 디버깅
def stream_graph_with_memory(question_list: list[str], thread_id: str = "DEBUG-THREAD-001"):
    """
    여러 질문을 같은 thread_id 로 순서대로 넣어서
    상태 누적까지 확인하는 디버깅 함수.

    예:
    stream_graph_with_memory([
        "fruit 로 할게. 사과 2개 빨강",
        "바나나 3개 노랑, 포도 5개 보라",
    ])
    """

    global DEBUG_SHOW_INVOKE_MESSAGES

    memory = MemorySaver()
    graph = MY_build_graph(checkpointer=memory)
    config_now = {"configurable": {"thread_id": thread_id}}
    DEBUG_SHOW_INVOKE_MESSAGES = True

    try:
        for turn_no, question in enumerate(question_list, start=1):
            print("\n" + "=" * 70)
            print(f"[TURN #{turn_no}] USER INPUT : {question}")
            print("=" * 70)

            step_no = 0

            for event in graph.stream(
                {
                    "user_input": question,
                    "messages": [],
                    "target_type": "",
                    "collected_items": [],
                    "route": "",
                    "missing_message": "",
                    "summary_text": "",
                    "fruit_counts": [],
                },
                config=config_now,
                stream_mode="values",
            ):
                step_no += 1

                messages = event.get("messages", [])
                number_message = len(messages)

                if number_message == 0:
                    last_message = None
                    type_message = "EMPTY"
                    last_message_content = ""
                else:
                    last_message = messages[-1]
                    type_message = type(last_message).__name__
                    last_message_content = last_message.content

                print(f"\n[TURN #{turn_no} / STEP #{step_no}]")
                print(f"[#{number_message}][{type_message}] : {last_message_content}")
                print(f"     ■ target_type    : {event.get('target_type', '')}")
                print(f"     ■ route          : {event.get('route', '')}")
                print(f"     ■ fruit_counts   : {event.get('fruit_counts', [])}")
                print(f"     ■ summary_text   : {event.get('summary_text', '')}")
                print(f"     ■ collected_items: {event.get('collected_items', [])}")

                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    print("     ■ [AI's TOOL CALLS]")
                    for idx, tool_call in enumerate(last_message.tool_calls):
                        print(f"        □ TOOL #{idx}")
                        print(f"            □ NAME(input) : {tool_call['name']}({tool_call['args']})")
                        print(f"            □ ID          : {tool_call['id']}")

                if last_message is not None and isinstance(last_message, ToolMessage):
                    print("     ■ [TOOL Result]")
                    print(f"        □ NAME   : {last_message.name}")
                    print(f"        □ output : {last_message.content}")
                    print(f"        □ ID     : {last_message.tool_call_id}")
    finally:
        DEBUG_SHOW_INVOKE_MESSAGES = False


# ---------------------------------------------------------------------------
# [13] main
# ---------------------------------------------------------------------------
def main():
    """실행 진입점."""

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY 환경 변수가 필요합니다.")

    MY_memory = MemorySaver()
    MY_thread_id = "TARGET-STUDY-001"

    print("실행 모드를 숫자로 선택하세요.")
    print("0 : 일반 챗봇 실행")
    print("1 : Debug #1 - 최종 답변만 확인")
    print("2 : Debug #2 - Stream 방식으로 각 step 의 State 확인")
    print("3 : Debug #3 - MemorySaver 를 붙여 여러 턴 누적 상태 확인")

    mode = input("MODE : ").strip()

    if mode == "1":
        i_said = input("YOU : ")
        ask_graph(i_said)

    elif mode == "2":
        i_said = input("YOU : ")
        stream_graph(i_said)

    elif mode == "3":
        print("여러 턴 디버깅을 시작합니다. 종료하려면 q 를 입력하세요.")

        question_list = []

        while True:
            i_said = input("YOU : ").strip()

            if i_said.lower() == "q":
                break

            if not i_said:
                continue

            question_list.append(i_said)

        if question_list:
            stream_graph_with_memory(question_list, thread_id=MY_thread_id)
        else:
            print("입력된 질문이 없어 종료합니다.")

    else:
        run_chatbot(MY_memory, MY_thread_id)


if __name__ == "__main__":
    main()
