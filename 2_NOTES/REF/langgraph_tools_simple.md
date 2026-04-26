# LangGraph Tools Simple

이 문서는 [langgraph_tools_simple.py](/c:/Users/sycsn/OneDrive/STUDY/Coding/PYTHON/PY_LLM/PY_DoitLLM_Solution/chap12/sec03/langgraph_tools_simple.py:1)를 기준으로 LangGraph의 기본 구조를 공부하기 위한 설명서입니다.

## 1. 이 예제를 왜 이렇게 구성했는가

원래 웹검색 도구 예제는 다음 요소가 한꺼번에 섞입니다.

- 검색 API 또는 외부 서비스 의존성
- 네트워크 상태
- 검색 결과 품질
- 도구 호출 구조
- LangGraph 라우팅 구조

처음 LangGraph를 공부할 때는 핵심이 흐려집니다.  
그래서 이 예제는 외부 API 없이도 이해할 수 있도록 다음 세 가지 도구만 사용합니다.

- `add_numbers(a, b)`
- `multiply_numbers(a, b)`
- `reverse_text(text)`

즉, 지금 예제의 핵심은 "도구가 무엇을 하느냐"가 아니라 아래 흐름입니다.

1. LLM이 사용자 요청을 읽는다.
2. 도구가 필요한지 판단한다.
3. 필요하면 tool call을 만든다.
4. LangGraph가 tools 노드로 이동한다.
5. tools 노드가 실제 파이썬 함수를 실행한다.
6. 실행 결과를 다시 LLM에 넘긴다.
7. LLM이 최종 자연어 답변을 만든다.

## 2. 전체 구조 한눈에 보기

이 파일의 핵심 구성요소는 아래 5개입니다.

- `State`: 그래프 전체가 공유하는 상태
- `@tool` 함수들: 실제 작업을 수행하는 도구
- `generate()`: LLM이 다음 행동을 결정하는 노드
- `BasicToolNode`: LLM이 요청한 tool을 실제로 실행하는 노드
- `route_tools()`: 다음 이동 경로를 정하는 분기 함수

그래프 흐름은 아래와 같습니다.

```text
START
  |
  v
generate
  | \
  |  \ tool_calls 없음
  |   \
  |    v
  |   END
  |
  | tool_calls 있음
  v
tools
  |
  v
generate
```

즉, `generate -> tools -> generate`가 한 바퀴 돌 수 있습니다.

## 3. State: LangGraph가 공유하는 상태

코드:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

의미:

- 이 그래프는 `messages` 하나를 중심으로 동작합니다.
- `messages`에는 `HumanMessage`, `AIMessage`, `ToolMessage` 등이 차곡차곡 들어갑니다.
- `add_messages`는 기존 메시지를 덮어쓰지 않고 이어붙이도록 도와줍니다.

왜 중요할까:

- LangGraph는 노드들이 상태를 주고받으면서 동작합니다.
- 이 예제에서는 상태를 최소화해서 `messages`만 남겼습니다.
- 나중에는 여기에 `user_id`, `task`, `documents`, `plan` 같은 필드를 추가할 수 있습니다.

## 4. Tool: LLM이 호출할 수 있는 함수

코드:

```python
@tool
def add_numbers(a: int, b: int) -> str:
    return f"{a} + {b} = {a + b}"
```

`@tool`의 역할:

- 일반 파이썬 함수를 "LLM이 호출 가능한 도구"로 바꿉니다.
- 함수 이름, 인자, docstring이 모델에게 전달됩니다.
- 모델은 사용자 요청을 보고 이 도구를 쓸지 결정합니다.

이 예제에서 중요한 포인트:

- 함수 이름이 명확합니다.
- 인자 타입이 단순합니다.
- 설명(docstring)이 짧고 분명합니다.

처음 공부할 때는 tool을 복잡하게 만들지 않는 것이 좋습니다.  
tool 내부가 어려우면 LangGraph 구조가 아니라 함수 구현에 시선이 빼앗깁니다.

## 5. generate 노드: LLM이 다음 행동을 고르는 곳

코드 위치: [generate](/c:/Users/sycsn/OneDrive/STUDY/Coding/PYTHON/PY_LLM/PY_DoitLLM_Solution/chap12/sec03/langgraph_tools_simple.py:47)

핵심 코드:

```python
model = create_model()
model_with_tools = model.bind_tools(TOOLS)

response = model_with_tools.invoke(
    [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
)
```

이 노드의 역할은 두 가지입니다.

1. 그냥 바로 답할지
2. tool을 호출할지

모델은 `bind_tools(TOOLS)`가 되어 있기 때문에, 필요하면 아래 같은 결정을 할 수 있습니다.

- "이건 덧셈이니까 `add_numbers`를 호출하자"
- "이건 문자열 뒤집기니까 `reverse_text`를 호출하자"
- "이미 tool 결과가 있으니 그걸 바탕으로 최종 답변을 하자"

즉, `generate`는 실제 작업을 직접 하지 않고 "판단"을 담당합니다.

## 6. BasicToolNode: 실제 도구 실행 담당

코드 위치: [BasicToolNode](/c:/Users/sycsn/OneDrive/STUDY/Coding/PYTHON/PY_LLM/PY_DoitLLM_Solution/chap12/sec03/langgraph_tools_simple.py:58)

핵심 흐름:

1. 마지막 메시지를 가져온다.
2. 그 메시지 안의 `tool_calls`를 확인한다.
3. 각 tool call에 대해 실제 파이썬 함수를 실행한다.
4. 결과를 `ToolMessage`로 감싸서 반환한다.

중요 코드:

```python
for tool_call in last_message.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_result = self.tools_by_name[tool_name].invoke(tool_args)
```

여기서 꼭 이해해야 할 점:

- LLM이 만든 tool call은 "실행 지시"일 뿐입니다.
- 실제 함수 실행은 LangGraph 노드가 해야 합니다.
- 그래서 `generate`와 `tools`가 분리되어 있습니다.

이 분리를 이해하면 나중에 아래 구조도 쉽게 이해됩니다.

- 검색용 tool 노드
- DB 조회 tool 노드
- 계산 tool 노드
- 외부 API 호출 노드

## 7. route_tools: 어디로 갈지 결정하는 분기 함수

코드 위치: [route_tools](/c:/Users/sycsn/OneDrive/STUDY/Coding/PYTHON/PY_LLM/PY_DoitLLM_Solution/chap12/sec03/langgraph_tools_simple.py:87)

핵심 코드:

```python
last_message = messages[-1]
if hasattr(last_message, "tool_calls") and last_message.tool_calls:
    return "tools"
return END
```

이 함수는 아주 중요합니다.  
LangGraph에서는 노드 자체보다 "다음에 어디로 이동할지"를 설계하는 능력이 실력 차이를 만듭니다.

이 예제에서는 분기가 단순합니다.

- tool call이 있으면 `tools`
- 없으면 `END`

하지만 실전에서는 이런 식으로 확장됩니다.

- 검색이 필요하면 `search_node`
- 요약이 필요하면 `summarize_node`
- 사람이 승인해야 하면 `human_review_node`
- 오류가 나면 `retry_node`

즉, LangGraph 실력은 결국 "상태 설계 + 분기 설계"입니다.

## 8. build_graph: 그래프 연결하기

코드 위치: [build_graph](/c:/Users/sycsn/OneDrive/STUDY/Coding/PYTHON/PY_LLM/PY_DoitLLM_Solution/chap12/sec03/langgraph_tools_simple.py:102)

핵심 코드:

```python
graph_builder.add_node("generate", generate)
graph_builder.add_node("tools", BasicToolNode(TOOLS))

graph_builder.add_edge(START, "generate")
graph_builder.add_conditional_edges(
    "generate",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "generate")
```

여기서 핵심은 세 줄입니다.

- `START -> generate`
- `generate -> tools 또는 END`
- `tools -> generate`

이 세 줄만 정확히 이해해도 LangGraph의 기본 도구 호출 패턴은 거의 이해한 것입니다.

## 9. 실제 실행 시 메시지 흐름

예를 들어 질문이 `"17과 25를 더해줘."`라면 내부 흐름은 대략 아래와 같습니다.

1. `HumanMessage("17과 25를 더해줘.")`
2. `generate` 실행
3. 모델이 `add_numbers(a=17, b=25)` tool call 생성
4. `route_tools`가 `"tools"` 반환
5. `BasicToolNode`가 `add_numbers` 실행
6. `ToolMessage("17 + 25 = 42")` 생성
7. 다시 `generate` 실행
8. 모델이 tool 결과를 읽고 `"17과 25를 더한 결과는 42입니다."` 같은 최종 답변 생성
9. `route_tools`가 `END` 반환

이 순서를 머릿속에 그릴 수 있어야 합니다.

## 10. 왜 ask_graph 함수가 따로 있는가

코드 위치: [ask_graph](/c:/Users/sycsn/OneDrive/STUDY/Coding/PYTHON/PY_LLM/PY_DoitLLM_Solution/chap12/sec03/langgraph_tools_simple.py:118)

```python
def ask_graph(question: str) -> str:
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
```

이 함수는 학습할 때 유용합니다.

- 입력 지점을 한 군데로 모읍니다.
- 테스트하기 쉬워집니다.
- 나중에 Streamlit, FastAPI, CLI로 붙일 때 재사용하기 좋습니다.

즉, 그래프를 만드는 코드와 그래프를 사용하는 코드를 분리한 것입니다.

## 11. 실력 향상을 위한 공부 순서

이 파일 하나로 아래 순서대로 연습하면 좋습니다.

### 1단계: 구조 눈에 익히기

먼저 아래 함수 4개만 반복해서 읽으세요.

- `generate`
- `BasicToolNode.__call__`
- `route_tools`
- `build_graph`

이 단계 목표:

- "LLM이 판단"
- "ToolNode가 실행"
- "조건부 edge가 분기"

이 3개를 분명히 구분하는 것

### 2단계: tool 추가해보기

예를 들어 아래 tool을 직접 추가해보세요.

```python
@tool
def count_characters(text: str) -> str:
    """문자열 길이를 반환합니다."""
    return f"문자 수: {len(text)}"
```

그리고 `TOOLS`에 넣은 뒤 이런 질문을 던져보세요.

- `"banana의 글자 수를 세어줘."`
- `"hello world는 몇 글자야?"`

이 단계 목표:

- tool을 추가해도 그래프 구조는 안 바뀐다는 점 이해

### 3단계: 상태 확장하기

`State`에 새 필드를 추가해보세요.

예:

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_level: str
```

이 단계 목표:

- LangGraph는 단순 채팅 로그만 다루는 것이 아니라,
  구조화된 상태를 관리하는 프레임워크라는 점 이해

### 4단계: 분기 늘리기

지금은 `tools` 또는 `END`만 있습니다.  
다음에는 아래처럼 노드를 하나 더 만들어보세요.

- `explain_result`
- `validate_input`
- `retry`

이 단계 목표:

- "LangGraph는 tool 호출용 라이브러리"가 아니라
  "상태 기반 워크플로우 엔진"이라는 점 체감

## 12. 추천 실습 과제

아래 순서로 해보면 좋습니다.

1. `count_characters` tool 추가
2. `to_uppercase` tool 추가
3. `route_tools`가 아니라 다른 조건 분기 함수 하나 더 만들기
4. 최종 답변 전에 설명 전용 노드 하나 추가하기
5. `graph.invoke` 결과 전체를 출력해서 message 목록 직접 보기

특히 5번이 중요합니다.  
최종 답변 문자열만 보지 말고, `messages` 전체를 확인해야 LangGraph 내부 흐름이 보입니다.

## 13. 실행 방법

필수 조건:

- `OPENAI_API_KEY` 환경 변수 설정
- `langgraph`, `langchain-openai`, `langchain-core`, `typing_extensions` 설치

실행:

```bash
python PY_DoitLLM_Solution/chap12/sec03/langgraph_tools_simple.py
```

루트에 있는 같은 이름의 파일로 실행해도 됩니다.

```bash
python langgraph_tools_simple.py
```

## 14. 이 예제를 보고 반드시 이해해야 하는 한 문장

LangGraph에서 중요한 것은 "도구 자체"보다 "상태를 기준으로 어떤 노드로 이동할지 설계하는 것"입니다.

그 감각이 생기면 이후의 웹검색, RAG, 메모리, 멀티에이전트 구조도 훨씬 빠르게 이해할 수 있습니다.
