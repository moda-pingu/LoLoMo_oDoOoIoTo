# pip install streamlit openai
# To run: streamlit run PYLLM_01_StreamLit.py
import streamlit as st
import random
import time

# OpenAI API 설정
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 대화 기록 비휘발 기록
import json

# -------------------------
# 기록 JSON
# -------------------------
# [0] 대화 기록 저장 폴더 및 파일 경로 설정
here = os.getcwd()
d_history = "chat_history"
path_d_history = os.path.join(here, d_history)
f_history = "history.json"
path_f_history = os.path.join(path_d_history, f_history)
os.makedirs(path_d_history, exist_ok=True)

# [1] JSON 파일 로드 함수
def load_json(file_path: str) -> list[dict]:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# [2] JSON 파일 저장 함수
def save_json(file_path: str, data: list[dict]) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# # [3] 이전 대화 기록 불러오기 : 매번 수행되지 않도록 초기화 시점에만 실행되도록 위치 변경
# st.session_state 사용법 참고:
# 단순 dict 형이며, key로 상태를 추가 및 저장/읽기 가능. (key는 문자열, 자유롭게 사용 가능)
# 예: st.session_state["key"] = value
# 일반 변수와 달리, rerun 시에도 값이 유지됨.
# if "messages" not in st.session_state:
#     loaded_history = load_json(path_f_history)

# -------------------------
# Sidebar: API Key 입력
# -------------------------
st.sidebar.header("🔑 Settings")
input_api_key = st.sidebar.text_input(
    "OpenAI API Key",                      # Label
    type            ="password",           # Input type; password로 설정 - 입력값이 가려짐
    placeholder     ="sk-xxxx",            # 흐린 글씨로 표시되는 힌트
    help            ="API Key를 입력해주세요.",   # 마우스 오버 시 도움말 표시
)

openai_api_key = None
# [0] 빈 문자열 입력 시 세션 상태에서 키 삭제
if input_api_key == "":
    st.session_state.pop("OPENAI_API_KEY", None)
    openai_api_key = None

# [1] 사용자 입력 최우선
if input_api_key:
    openai_api_key = input_api_key
    st.session_state['OPENAI_API_KEY'] = input_api_key
    st.sidebar.success("OPENAI API Key가 설정되었습니다.")

# [2] 세션 상태에 저장된 키가 있으면 그걸 사용
elif "OPENAI_API_KEY" in st.session_state:
    openai_api_key = st.session_state['OPENAI_API_KEY']

# [3] 그 외에는 .env 파일의 환경변수 사용
elif os.getenv("OPENAI_API_KEY"):
    openai_api_key = os.getenv("OPENAI_API_KEY")

else:
    st.sidebar.info("OpenAI API Key를 입력해주세요.")

if not openai_api_key:
    st.warning("OpenAI API Key가 설정되지 않았습니다.")
    st.stop()  # 앱 실행 중지

# OpenAI 클라이언트 생성 via openai_api_key
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# -----------------------------
# 1) 기본 UI 텍스트
# -----------------------------
st.title("🤓 드와이트봇")
st.write("나는 드와이트 슈루트. 이 지점의 비공식 관리자. 공식적으로는 아니지만, 사실상 그렇다.")
st.caption("Using Streamlit's chat interface to simulate a Dwight_Schrute bot.")

# 3) 세션 상태(Session State) 초기화'
# -----------------------------
# st.session_state에 "messages"가 없으면 초기값을 설정
# Streamlit은 사용자 입력/클릭이 있을 때마다 스크립트를 "처음부터" 다시 실행(rerun)함.
# 그래서 "대화 히스토리" 같은 상태는 st.session_state에 저장해야 유지됨.
loaded_history = []
if "messages" not in st.session_state:
    loaded_history = load_json(path_f_history)
    if loaded_history:
        st.session_state.messages = loaded_history

    else:
        # role은 보통 "system", "user", "assistant"를 사용
        st.session_state.messages = [] # 대화기록 생성 및 초기화 
        # System 프롬프트 추가
        st.session_state.messages.append({
            "role": "system",
            "content": ( #하나의 문자열로 처리됨 - 파이썬 : 문자열 리터럴이 붙어 있으면 자동 결합(implicit concatenation)
                "너는 드와이트 슈루트다. "
                "드와이트 슈루트는 미국 NBC 시리즈 'The Office'의 등장인물로, "
                "그는 농장주이자 자칭 '비공식 관리자'로서, 회사 규칙과 질서를 엄격히 준수한다. "
                "대화에서 드와이트의 독특한 성격과 어투를 반영하여 답변하라."
                "말투 규칙:\n"
                "- 짧고 단호한 문장\n"
                "- '사실:', '규칙:', '경고:' 같은 레이블 자주 사용\n"
                "- 훈계 또는 체크리스트 형식 선호\n"
                "- 스스로는 진지하지만 결과적으로 웃긴 톤\n"

                "행동 규칙:\n"
                "- 질문에는 반드시 실질적인 해결책을 제공\n"
                "- 항상 캐릭터를 유지\n"
            ),
        })
        # 맨처음 어시스턴트 인사말 추가
        st.session_state.messages.append({
            "role": "assistant",
            "content": "안녕은 생략한다. 시간 낭비다.",
        })

# -----------------------------
# 4) 이전 대화 기록 출력 (렌더링)
# -----------------------------
# Rerun될 때마다, session_state에 저장된 메시지를 전부 다시 그려서 UI를 복원함.
for message in st.session_state.messages:
    if message["role"] == "system": continue  # system 메시지는 화면에 표시하지 않음
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# 5) OpenAI 호출 함수 (Responses API)
# -----------------------------
def call_openai(client, messages: list[dict], model: str = "gpt-5.2") -> str:
    """
    messages: [{"role": "...", "content": "..."}] 형태의 리스트
    model: 사용할 모델명

    Returns:
        assistant 답변 텍스트
    """
    if client is None:
        return "API Key가 설정되지 않아 OpenAI 호출을 할 수 없습니다."

    response = client.chat.completions.create(
        model=model,
        messages=messages,  # 멀티턴 대화를 그대로 전달
        # stream=True,       # 스트리밍 여부 (True로 하면 chunk 단위로 응답이 옴)
    )
    # SDK에서 편하게 답변 텍스트를 뽑을 수 있음
    # SDK(Software Development Kit) : OpenAI에서 제공하는 Python용 라이브러리로, API 호출을 더 쉽게 만들어줌
    return response.choices[0].message.content


# -----------------------------
# 6) 사용자 입력 처리
# -----------------------------
# 입력창 생성 :
# st.chat_input은 화면 하단에 입력창을 만들고,
# 사용자가 Enter를 치면 prompt 문자열을 반환함. 입력이 없으면 None.
prompt = st.chat_input("필요한 것 있나?")

if prompt:
    # (1) 기록-입력; 유저 메시지를 세션 히스토리에 저장
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # (2) 출력-입력; 유저 메시지를 즉시 화면에 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # (3) 어시스턴트 응답 영역 만들기
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # 사용자가 입력하면 OpenAI API 호출(네트워크/모델 계산으로 시간이 걸릴 수 있음)
        # UI가 멈춘 것처럼 보이지 않게 spinner 사용
        with st.spinner("생각중이다...조용할 것."):
            try:
                assistant_text = call_openai(
                    client = client,
                    model = "gpt-4o",
                    messages = st.session_state.messages,
                )
            except Exception as e:
                assistant_text = f"OpenAI 호출 중 에러가 발생했습니다: `{e}`"

        # (4) 기존 데모처럼 '타이핑' 효과를 주고 싶으면 아래처럼 흉내낼 수 있음
        full_response = ""
        for chunk in assistant_text:
            full_response += chunk
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # (5) 기록-응답; 어시스턴트 응답을 세션 히스토리에 저장 (다음 턴의 컨텍스트가 됨)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

    # (6) 대화 기록을 JSON 파일로 저장
    save_json(path_f_history, st.session_state.messages)

