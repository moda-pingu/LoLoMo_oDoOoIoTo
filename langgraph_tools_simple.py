"""
langgraph_tools_simple.py

학습 목표
1. LangGraph에서 State가 무엇인지 이해한다.
2. LLM이 tool을 "직접 실행"하는 것이 아니라 "tool 호출 지시(tool call)"를 만든다는 점을 이해한다.
3. generate 노드와 tools 노드가 왜 분리되어 있는지 이해한다.
4. conditional edge를 이용해 그래프의 흐름을 제어하는 방법을 익힌다.

이 파일은 웹검색 같은 외부 의존성이 큰 예제 대신,
아주 단순한 계산/문자열 처리 도구를 사용해서 LangGraph의 핵심 구조만 학습하도록 설계했다.
"""

# ---------------------------------------------------------------------------
# [1] 기본 import
# ---------------------------------------------------------------------------
# Annotated:
#   타입에 "추가 정보"를 붙일 때 사용한다.
#   여기서는 messages 필드가 어떻게 누적되어야 하는지(add_messages)를 함께 지정한다.
from typing import Annotated

# TypedDict:
#   "딕셔너리처럼 생긴 상태 객체"의 구조를 명시할 때 사용한다.
#   LangGraph에서 상태(State)를 정의할 때 자주 사용한다.
from typing_extensions import TypedDict

# json:
#   tool 실행 결과를 ToolMessage에 문자열 형태로 담기 위해 사용한다.
#   ToolMessage.content는 보통 문자열이므로, 파이썬 객체를 안전하게 문자열로 바꾸는 용도다.
import json

# os:
#   OPENAI_API_KEY 환경 변수가 있는지 확인하기 위해 사용한다.
import os

# HumanMessage:
#   사용자의 입력 메시지를 나타낸다.
# SystemMessage:
#   모델의 역할/규칙을 지정하는 메시지다.
# ToolMessage:
#   tool이 실행된 뒤 그 결과를 다시 모델에게 넘길 때 사용하는 메시지다.
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# @tool:
#   일반 파이썬 함수를 LLM이 호출 가능한 도구로 등록해준다.
from langchain_core.tools import tool

# ChatOpenAI:
#   OpenAI 채팅 모델을 LangChain 인터페이스로 사용하기 위한 클래스다.
from langchain_openai import ChatOpenAI

# StateGraph:
#   LangGraph에서 상태 기반 그래프를 구성하는 핵심 클래스다.
# START, END:
#   그래프의 시작 지점과 종료 지점을 나타내는 특수 상수다.
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# add_messages:
#   상태의 messages 필드가 "덮어쓰기"가 아니라 "누적"되도록 만드는 함수다.
#   대화형 그래프에서는 거의 필수라고 생각하면 된다.
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# [2] State 정의
# ---------------------------------------------------------------------------
# LangGraph는 "노드 간에 상태(State)를 주고받으며" 동작한다.
# 이 예제에서는 상태를 최소화해서 messages 하나만 사용한다.
# 실제 프로젝트에서는 여기에 user_id, session_id, documents, summary 같은 필드를 추가할 수 있다.
class State(TypedDict):
    """LangGraph 전체에서 공유하는 상태 구조."""

    # messages:
    #   지금까지의 대화와 tool 실행 결과를 저장하는 핵심 필드다.
    #
    # Annotated[list, add_messages]의 의미:
    #   1) 이 필드는 list 형태다.
    #   2) 노드가 {"messages": [...]}를 반환할 때 기존 값을 덮어쓰지 말고 누적(add)하라.
    #
    # 즉, 다음과 같은 흐름이 가능해진다.
    #   HumanMessage -> AIMessage(tool call) -> ToolMessage -> AIMessage(final answer)
    messages: Annotated[list, add_messages]

    # user_input:
    #   그래프 시작 시점에 사용자가 입력한 "원본 질문"을 담는다.
    #
    # 왜 따로 두는가?
    #   이번 버전에서는 START 직후에 초기 설정용 노드(init_system_context)를 둔다.
    #   이 노드가 SystemMessage와 HumanMessage를 함께 넣어서
    #   generate가 바로 model.invoke(state["messages"])를 할 수 있게 만든다.
    #
    # 중요한 점:
    #   graph.invoke(...) 또는 graph.stream(...)를 호출하는 쪽에서는
    #   사용자 입력을 곧바로 messages에 넣지 않는다.
    #   대신 user_input 필드에만 담아두고,
    #   init_system_context 노드가 그것을 HumanMessage로 변환한다.
    user_input: str


# ---------------------------------------------------------------------------
# [3] Tool 정의
# ---------------------------------------------------------------------------
# 아래 함수들은 실제로 일을 수행하는 "도구"다.
# 중요한 점:
#   - LLM은 이 함수를 직접 실행하지 않는다.
#   - LLM은 "이 도구를 써야겠다"라고 판단해서 tool call만 만든다.
#   - 실제 실행은 뒤에서 만들 BasicToolNode가 담당한다.


@tool
def add_numbers(a: int, b: int) -> str:
    """
    두 정수를 더한 결과를 반환한다.

    학습 포인트:
    - 가장 단순한 tool 형태를 보여준다.
    - 입력 인자가 명확하다: a, b
    - 출력도 단순한 문자열이다.
    - LLM은 이 함수명과 설명을 보고 "덧셈 요청일 때 써야 하는 도구"라고 이해한다.
    """

    return f"{a} + {b} = {a + b}"


@tool
def multiply_numbers(a: int, b: int) -> str:
    """
    두 정수를 곱한 결과를 반환한다.

    학습 포인트:
    - add_numbers와 거의 같은 구조라서, tool 추가가 얼마나 단순한지 보여준다.
    - LangGraph 구조는 바꾸지 않고 tool만 늘릴 수 있다는 점이 중요하다.
    """

    return f"{a} x {b} = {a * b}"


@tool
def reverse_text(text: str) -> str:
    """
    입력 문자열을 뒤집어서 반환한다.

    학습 포인트:
    - 숫자 계산이 아닌 문자열 처리 tool 예시다.
    - LLM이 "요청 종류"에 따라 적절한 tool을 골라야 함을 보여준다.
    """

    return text[::-1]


# TOOLS:
#   모델에 바인딩(bind)할 전체 도구 목록이다.
#   여기 리스트에 들어 있는 함수만 모델이 tool call 대상으로 인식할 수 있다.
TOOLS = [add_numbers, multiply_numbers, reverse_text]


# ---------------------------------------------------------------------------
# [4] 시스템 프롬프트
# ---------------------------------------------------------------------------
# 이 프롬프트는 모델에게 "언제 tool을 사용해야 하는가"를 분명하게 알려준다.
# 학습용 예제에서는 프롬프트를 짧지만 명확하게 쓰는 것이 좋다.
SYSTEM_PROMPT = """
너는 학습용 도우미다.
사용자 요청을 보고 계산이나 문자열 뒤집기가 필요하면 반드시 적절한 도구를 호출하라.
도구 결과를 받은 뒤에는 최종 답변을 한국어 한두 문장으로 간단히 정리하라.
""".strip()


# ---------------------------------------------------------------------------
# [5] 모델 생성 함수
# ---------------------------------------------------------------------------
# 모델 생성을 함수로 분리한 이유:
# 1) generate 함수가 너무 길어지지 않게 하기 위해
# 2) 나중에 모델명을 바꾸거나 옵션을 조정할 때 수정 지점을 한 곳으로 모으기 위해
# 3) 학습 시 "그래프 로직"과 "모델 설정"을 구분해서 보기 쉽게 하기 위해
def create_model() -> ChatOpenAI:
    """OpenAI 채팅 모델 객체를 생성해서 반환한다."""

    # temperature=0:
    #   가능한 한 일관된 출력을 얻기 위해 0으로 두었다.
    #   학습용 예제는 답이 들쭉날쭉하지 않는 편이 좋다.
    return ChatOpenAI(model="gpt-4o", temperature=0)


# ---------------------------------------------------------------------------
# [6] 시스템 초기화 노드
# ---------------------------------------------------------------------------
# 이 노드는 그래프가 시작되자마자 가장 먼저 실행된다.
#
# 역할:
# 1) 전반적인 규칙을 담은 SystemMessage를 만든다.
# 2) 첫 턴에만 SystemMessage를 넣고, 이후 턴에는 중복 삽입을 막는다.
#
# 왜 이런 노드를 두는가?
# - 중요한 시스템 프롬프트를 그래프 시작 시점에 명시적으로 주입할 수 있다.
# - generate 노드에서 매번 [SystemMessage(...)] + state["messages"] 를 만들 필요가 없다.
# - 시스템 규칙 주입과 사용자 입력 추가를 서로 다른 노드로 분리할 수 있다.
# - MemorySaver를 붙여 여러 턴 대화를 이어갈 때도, 첫 턴에만 SystemMessage를 넣을 수 있다.
def init_system(state: State):
    """첫 턴에만 SystemMessage를 messages 맨 앞에 추가한다."""

    messages = state.get("messages", [])

    # 이전 대화가 복원된 상태에서 이미 첫 메시지가 SystemMessage라면
    # 시스템 프롬프트를 다시 넣지 않는다.
    if messages and isinstance(messages[0], SystemMessage):
        return {}

    return {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
    }


# ---------------------------------------------------------------------------
# [7] 사용자 입력 추가 노드
# ---------------------------------------------------------------------------
# 이 노드는 이번 턴의 원본 입력(user_input)을 HumanMessage로 바꿔
# messages 리스트 뒤에 추가한다.
#
# 역할을 분리하는 이유:
# - init_system: 시스템 규칙 주입
# - add_user_message: 사용자 메시지 추가
# - generate: LLM 호출
# 이렇게 나누면 각 노드의 책임이 훨씬 선명해진다.
def add_user_message(state: State):
    """이번 턴의 사용자 입력을 HumanMessage로 변환해 messages에 추가한다."""

    return {
        "messages": [
            HumanMessage(content=state["user_input"]),
        ]
    }


# ---------------------------------------------------------------------------
# [8] generate 노드
# ---------------------------------------------------------------------------
# 이 함수는 LangGraph의 "생성 노드" 역할을 한다.
#
# 핵심 역할:
# 1) 현재까지의 messages를 읽는다.
# 2) 모델에게 다음 행동을 맡긴다.
# 3) 모델은 두 가지 중 하나를 한다.
#    - 바로 최종 답변을 생성한다.
#    - tool call을 포함한 AIMessage를 생성한다.
#
# 중요한 개념:
# - generate는 "판단"을 담당한다.
# - 실제 tool 실행은 담당하지 않는다.
def generate(state: State):
    """현재 상태(messages)를 바탕으로 다음 AIMessage를 생성한다."""

    # 모델 생성
    model = create_model()

    # bind_tools:
    #   모델에게 "이런 도구들이 있다"라고 알려주는 과정이다.
    #   이 설정을 해야 모델이 tool_calls를 포함한 응답을 생성할 수 있다.
    model_with_tools = model.bind_tools(TOOLS)

    # SystemMessage는 init_system 노드가, HumanMessage는 add_user_message 노드가
    # 이미 messages에 추가해두었다.
    # 따라서 generate는 현재까지 누적된 messages를 그대로 모델에 전달하면 된다.
    #
    # messages 안에는 보통 다음과 같은 흐름이 들어 있다.
    #   [SystemMessage, HumanMessage, AIMessage(tool call), ToolMessage, AIMessage...]
    response = model_with_tools.invoke(state["messages"])

    # 반환 형식:
    #   {"messages": [response]}
    #
    # 왜 리스트로 감싸는가?
    #   messages 필드는 add_messages로 누적되므로,
    #   "이번 노드에서 새로 추가할 메시지 목록" 형태로 반환하는 것이 자연스럽다.
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# [9] BasicToolNode 정의
# ---------------------------------------------------------------------------
# 이 클래스는 "LLM이 요청한 tool을 실제로 실행하는 노드"다.
#
# 왜 별도 노드가 필요한가?
# - 모델은 tool을 직접 실행하지 않는다.
# - 모델은 단지 "이 도구를 이런 인자로 호출하라"는 구조화된 지시(tool_calls)를 만든다.
# - 그 지시를 읽어서 실제 파이썬 함수를 호출하는 컴포넌트가 필요하다.
class BasicToolNode:
    """
    마지막 AIMessage의 tool_calls를 읽어서 실제 tool을 실행하는 노드.

    학습 포인트:
    - generate와 tools를 분리해야 LangGraph의 구조가 선명하게 보인다.
    - 나중에 이 자리에 더 복잡한 실행 정책(로깅, 예외처리, 재시도 등)을 넣을 수 있다.
    """

    def __init__(self, tools: list) -> None:
        # tools_by_name:
        #   tool 이름을 키로, 실제 tool 객체를 값으로 갖는 딕셔너리다.
        #
        # 예:
        #   {
        #       "add_numbers": <tool object>,
        #       "multiply_numbers": <tool object>,
        #       "reverse_text": <tool object>,
        #   }
        #
        # 이렇게 저장해두면, 모델이 "add_numbers"를 호출하라고 했을 때
        # 해당 함수를 빠르게 찾아 실행할 수 있다.
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        """
        LangGraph 노드처럼 동작하는 호출 메서드.

        inputs 예시:
            {
                "messages": [
                    HumanMessage(...),
                    AIMessage(... tool_calls=[...])
                ]
            }
        """

        # 현재 상태에서 messages를 읽는다.
        messages = inputs.get("messages", [])

        # 방어 코드:
        #   messages가 비어 있으면 tool 실행 자체가 불가능하다.
        if not messages:
            raise ValueError("messages가 없습니다.")

        # 마지막 메시지는 보통 generate 노드가 만든 AIMessage다.
        # 여기 안에 tool_calls가 들어 있다고 가정한다.
        last_message = messages[-1]

        # 여러 tool call을 담을 수 있으므로 결과도 리스트로 모은다.
        outputs = []

        # 모델이 요청한 모든 tool call을 순회한다.
        for tool_call in last_message.tool_calls:
            # 예:
            #   tool_call["name"] -> "add_numbers"
            #   tool_call["args"] -> {"a": 17, "b": 25}
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 실제 tool 실행
            # invoke를 사용하면 tool이 기대하는 형식대로 인자를 전달할 수 있다.
            tool_result = self.tools_by_name[tool_name].invoke(tool_args)

            # ToolMessage 생성 이유:
            #   tool의 실행 결과를 다시 모델이 읽을 수 있는 메시지 형태로 만들어야 하기 때문이다.
            #
            # tool_call_id를 함께 넣는 이유:
            #   어떤 tool 호출에 대한 응답인지 모델이 매칭할 수 있게 하기 위해서다.
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

        # 매우 중요:
        #   여기서 messages 전체를 다시 반환하지 않고 "새로 생성된 ToolMessage들만" 반환한다.
        #   add_messages가 알아서 기존 messages 뒤에 이어붙여준다.
        return {"messages": outputs}


# ---------------------------------------------------------------------------
# [10] 라우팅 함수
# ---------------------------------------------------------------------------
# LangGraph에서 실력을 가르는 핵심은 "분기 설계"다.
# 이 함수는 generate 노드의 출력(AIMessage)을 보고,
# 다음에 어디로 이동할지 결정한다.
def route_tools(state: State):
    """
    마지막 메시지에 tool_calls가 있으면 tools 노드로 이동한다.
    tool_calls가 없으면 더 할 일이 없다고 보고 END로 종료한다.
    """

    # 상태에서 messages를 읽는다.
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("route_tools에 messages가 없습니다.")

    # 분기 판단 대상은 항상 "마지막 메시지"다.
    # 이 메시지는 generate 노드가 직전에 만든 AIMessage라고 보면 된다.
    last_message = messages[-1]

    # hasattr(...)를 사용하는 이유:
    #   마지막 메시지가 항상 AIMessage라는 보장이 없기 때문이다.
    #   안전하게 tool_calls 속성이 있는지 먼저 확인한다.
    #
    # last_message.tool_calls가 비어 있지 않다면:
    #   모델이 tool 사용을 결정했다는 뜻이므로 tools 노드로 이동한다.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # tool_calls가 없다면:
    #   모델이 이미 최종 답변을 만들었다는 뜻이므로 그래프를 종료한다.
    return END


# ---------------------------------------------------------------------------
# [11] 그래프 조립 함수
# ---------------------------------------------------------------------------
# 여기서 실제 LangGraph의 구조를 만든다.
# 이 부분이 전체 예제의 "뼈대"다.
def build_graph(checkpointer=None):
    """학습용 LangGraph를 생성하고 compile 해서 반환한다."""

    # StateGraph(State):
    #   "이 그래프는 State 구조를 공유 상태로 사용합니다"라는 뜻이다.
    graph_builder = StateGraph(State)

    # 노드 등록
    # "init_system" 노드:
    #   첫 턴에만 시스템 프롬프트를 주입하는 노드
    graph_builder.add_node("init_system", init_system)

    # "add_user_message" 노드:
    #   이번 턴의 사용자 입력을 HumanMessage로 추가하는 노드
    graph_builder.add_node("add_user_message", add_user_message)

    # "generate" 노드:
    #   모델이 다음 행동을 결정하는 노드
    graph_builder.add_node("generate", generate)

    # "tools" 노드:
    #   실제 tool을 실행하는 노드
    graph_builder.add_node("tools", BasicToolNode(TOOLS))

    # 시작점은 시스템 초기화 노드로 연결한다.
    # 그래프는 먼저 시스템 프롬프트를 세팅하고,
    # 그 다음 사용자 메시지를 추가한 뒤,
    # generate 노드에서 모델 판단을 시작한다.
    graph_builder.add_edge(START, "init_system")
    graph_builder.add_edge("init_system", "add_user_message")
    graph_builder.add_edge("add_user_message", "generate")

    # generate 다음에는 조건 분기:
    # - tool call이 있으면 tools로 이동
    # - 없으면 종료
    graph_builder.add_conditional_edges(
        "generate",
        route_tools,
        {"tools": "tools", END: END},
    )

    # tools가 실행된 뒤에는 다시 generate로 돌아간다.
    #
    # 이유:
    #   tool 결과를 모델에게 다시 보여주고,
    #   그 결과를 바탕으로 자연어 최종 답변을 생성하게 해야 하기 때문이다.
    graph_builder.add_edge("tools", "generate")

    # compile():
    #   지금까지 정의한 노드와 엣지를 실제 실행 가능한 그래프로 만든다.
    #   checkpointer를 넘기면 같은 thread_id 안에서 상태가 이어진다.
    return graph_builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# [12] 그래프 호출용 편의 함수
# ---------------------------------------------------------------------------
# 학습용 코드에서는 "그래프 구성"과 "그래프 사용"을 분리해두는 편이 좋다.
# ask_graph는 질문 하나를 받아서 최종 답변 문자열만 꺼내준다.
def ask_graph(question: str) -> str:
    """질문 하나를 그래프에 넣고 최종 답변 문자열만 반환한다."""

    # 실행 가능한 그래프 생성
    graph = build_graph()

    # graph.invoke(...) 입력 형식:
    #   State 구조에 맞는 딕셔너리를 넣어야 한다.
    #
    # 이번 버전에서는 messages를 직접 넣지 않고,
    # user_input만 넣어서
    # init_system -> add_user_message 순서로
    # [SystemMessage, HumanMessage]가 만들어지도록 한다.
    #
    # 즉, 호출하는 쪽에서 이렇게 하지 않는다:
    #   {"messages": [HumanMessage(content=question)]}
    #
    # 대신 이렇게 시작한다:
    #   {"user_input": question, "messages": []}
    result = graph.invoke({"user_input": question, "messages": []})

    # result["messages"]에는 전체 메시지 히스토리가 들어 있다.
    # 맨 마지막 메시지가 최종 AI 답변이므로 그것의 content만 반환한다.
    return result["messages"][-1].content


def stream_graph(question: str):
    """질문 하나를 stream 방식으로 실행하며 중간 상태를 순서대로 출력한다."""

    graph = build_graph()

    # stream에서도 동일한 원칙을 유지한다.
    # 사용자 질문을 처음부터 messages에 넣지 않고,
    # user_input 필드로 넘긴 뒤 그래프 내부 노드가 HumanMessage로 변환한다.
    for event in graph.stream(
        {"user_input": question, "messages": []},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
        print(f"현재 메시지 개수: {len(event['messages'])}")


def run_memory_chatbot():
    """MemorySaver를 이용해 대화가 누적되는 터미널 챗봇을 실행한다."""

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)

    # thread_id가 같으면 같은 대화 스레드로 간주되어 상태가 이어진다.
    config_now = {"configurable": {"thread_id": "langgraph-tools-demo"}}

    print("대화형 챗봇을 시작합니다. 종료하려면 exit 또는 quit 를 입력하세요.")

    while True:
        user_input = input("\nUser: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("챗봇을 종료합니다.")
            break

        if not user_input:
            continue

        # 중요한 점:
        #   여기서 HumanMessage를 state["messages"]에 직접 넣지 않는다.
        #   원본 문자열만 user_input에 담아 보내고,
        #   init_system_context 노드가 SystemMessage/HumanMessage 구성을 담당한다.
        for event in graph.stream(
            {"user_input": user_input, "messages": []},
            config=config_now,
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()
            print(f"현재 메시지 개수: {len(event['messages'])}")


# ---------------------------------------------------------------------------
# [13] 데모 실행 함수
# ---------------------------------------------------------------------------
# 여러 질문을 순차적으로 넣어보면서
# 모델이 적절한 tool을 고르는지 확인하기 위한 함수다.
def run_demo():
    """샘플 질문으로 LangGraph의 tool 호출 흐름을 확인한다."""

    questions = [
        "17과 25를 더해줘.",
        "6과 8을 곱해줘.",
        "strawberry를 뒤집어줘.",
        "7과 9를 더한 뒤 결과를 한 문장으로 설명해줘.",
    ]

    for question in questions:
        print(f"\n[질문] {question}")
        answer = ask_graph(question)
        print(f"[답변] {answer}")


# ---------------------------------------------------------------------------
# [13] 스크립트 직접 실행 시 진입점
# ---------------------------------------------------------------------------
# 이 블록은 파일을 직접 실행했을 때만 동작한다.
# import해서 사용할 때는 실행되지 않는다.
if __name__ == "__main__":
    # OpenAI API를 사용하려면 API 키가 필요하다.
    # 없으면 모델 호출 단계에서 실패하므로, 먼저 명확한 오류를 발생시킨다.
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY 환경 변수가 필요합니다.")

    # 대화 누적형 챗봇 실행
    run_memory_chatbot()
