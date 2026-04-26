# 용어
    - invoke : 호출하다. ; 실제로 llm 한테 메시지 보내는 함수 - llm 대답이 반환된다.
    - annotated : 주석 ; 파이썬에서 타입에 대한 메타데이터를 추가해주는 기능이다.

# 트렌드
LLMChain보다 prompt | llm 같은 Runnable 스타일 권장
langchain에서 바로 loader, embeddings 가져오는 방식도 deprecated 방향이었음
__call__보다 invoke() 권장

# langchain lib
langchain_core = 공통 규격, 기본 부품
langchain = 그 위에서 쓰는 상위 프레임워크
langchain-openai, langchain-community 같은 것들 = 각 provider/integration 연결부

# langgraph

# START / END
Python LangGraph에서 START, END는 실제로 문자열
START = sys.intern("__start__")
END = sys.intern("__end__")
interned string일 뿐, 여전히 그냥 문자열입니다.


# 쓰레드
“이 대화 기록은 누구 거고, 어디까지 이어진 거냐를 구분하는 이름표”
“한 대화방”, “한 세션”, “한 사람과 이어지는 대화 줄기”
thread_id는 “이 입력이 어느 대화방의 연속인지 구분하는 번호”

# 체크포인터란?
체크포인터는 LangGraph에서 꽤 핵심
그냥 “메모리 저장기”라고만 보면 반은 맞고, 실제 역할은 더 넓어요.
한 줄로 먼저 말하면, 체크포인터는 그래프의 state를 실행 중간중간 저장해서, 나중에 같은 대화를 이어가거나, 멈췄다 재개하거나, 과거 시점으로 돌아가게 해주는 persistence 장치예요

LangGraph 공식 문서도 체크포인터를 붙이면 그래프 state의 snapshot이 각 실행 단계마다 checkpoint로 저장되고, 이것이 threads 단위로 관리된다고 설명합니다.

체크포인터가 저장하는 건 “채팅 메시지 몇 개”만이 아니고, 그 시점의 그래프 상태 전체 snapshot 을 말한다.

# 체크포인터와 메모리의 차이

state = 현재 작업 데이터
short-term memory = thread 안에서 누적되는 state 성격의 기억
checkpointer = 그 state를 저장하고 다시 꺼내오게 해주는 persistence 계층

# MemorySaver()
MemorySaver() 호출 1번마다 새로운 독립 메모리 저장소 인스턴스 생성
따라서 “다른 메모리 영역이 생성되냐?” → 맞습니다

memory_a = MemorySaver()
memory_b = MemorySaver()
이건 내부적으로 각자 별도의 in-memory 저장소를 갖습니다. 

그래서:
graph_a = builder_a.compile(checkpointer=memory_a)
graph_b = builder_b.compile(checkpointer=memory_b)
그러면 graph_a와 graph_b는 체크포인트 저장 공간 자체가 분리됩니다.
다른 MemorySaver를 쓰면 thread_id가 같아도 물리적으로 분리됨

# Annotated
Annotated[list, add_messages]를 보면
“아, 이건 list 타입이고, 나중에 엔진이 이 필드를 합칠 때 add_messages를 쓰라는 뜻이구나”
이렇게 읽으면 돼.

# Tool
LLM 이 Tool 쓸지 판단하는 근거
1. 함수 이름
2. docstring 설명
   - 이 도구가 무슨 일을 하는지
   - 언제 써야 하는지
   - 어떤 입력이 필요한지
   - 어떤 결과가 나오는지
3. 파라미터 이름과 타입
**. return 값은 “사후 활용”에는 중요하지만, 보통 “사전 선택 근거”는 아니다.
**. 그래서 툴을 더 잘 고르게 만들고 싶으면 return보다 description을 더 보강하는 게 효과가 크다고 보는 게 맞아.

```python
from langchain_core.tools import tool
@tool
def add_numbers(first_number: int, second_number: int) -> str:
    """두 정수를 입력받아 합을 계산해 반환합니다.
    사용자가 두 수의 합이나 덧셈 결과를 요청할 때 사용합니다.
    """
    return f"{first_number} + {second_number} = {first_number + second_number}"
```
- 실제 참고하는 함수내의 객체
    1. 함수 이름은 add_numbers.__name__
    2. docstring은 add_numbers.__doc__
    3. 타입 힌트는 add_numbers.__annotations__
    4. 파라미터 이름과 순서는 inspect.signature()

# ToolMassage
SystemMessage: 모델의 규칙/역할 설정
HumanMessage: 사용자 입력
AIMessage: 모델의 응답
ToolMessage: 도구 실행 결과

# json.dumps(dict) / json.loads() 사용 이유
json.dumps(...)는 파이썬 dict객체를 JSON 형식의 문자열
llm 한테 던질 때는 문자열로 던져야되니까.

json.dumps() : dict -> 문자열
json.loads() : 문자열 -> dict
