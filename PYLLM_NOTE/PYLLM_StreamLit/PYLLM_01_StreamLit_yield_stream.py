# pip install streamlit openai python-dotenv
# To run: streamlit run PYLLM_01_StreamLit_yield_stream.py
from __future__ import annotations

from collections.abc import Iterator
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st


load_dotenv()


# -------------------------
# Chat history JSON
# -------------------------
here = os.getcwd()
d_history = "chat_history"
path_d_history = os.path.join(here, d_history)
f_history = "history.json"
path_f_history = os.path.join(path_d_history, f_history)
os.makedirs(path_d_history, exist_ok=True)


def load_json(file_path: str) -> list[dict]:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(file_path: str, data: list[dict]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# -------------------------
# Sidebar: API Key input
# -------------------------
st.sidebar.header("Settings")
input_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-xxxx",
    help="API Key를 입력하세요.",
)
use_streaming = st.sidebar.checkbox("Streaming (stream=True)", value=True)


openai_api_key = None
if input_api_key == "":
    st.session_state.pop("OPENAI_API_KEY", None)
    openai_api_key = None
elif input_api_key:
    openai_api_key = input_api_key
    st.session_state["OPENAI_API_KEY"] = input_api_key
    st.sidebar.success("OpenAI API Key가 설정되었습니다.")
elif "OPENAI_API_KEY" in st.session_state:
    openai_api_key = st.session_state["OPENAI_API_KEY"]
elif os.getenv("OPENAI_API_KEY"):
    openai_api_key = os.getenv("OPENAI_API_KEY")
else:
    st.sidebar.info("OpenAI API Key를 입력하세요.")


if not openai_api_key:
    st.warning("OpenAI API Key가 설정되지 않았습니다.")
    st.stop()


client = OpenAI(api_key=openai_api_key)


# -----------------------------
# UI
# -----------------------------
st.title("Streamlit Chat (yield + stream)")
st.caption("stream=False는 완성 문자열, stream=True는 delta를 yield로 받습니다.")


def call_openai(
    *,
    client: OpenAI,
    messages: list[dict],
    model: str = "gpt-4o",
    stream: bool = True,
) -> str | Iterator[str]:
    """
    - stream=False: 최종 문자열(str) 반환
    - stream=True : 텍스트 조각(delta)을 yield 하는 Iterator[str] 반환
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
    )

    if stream:

        def gen() -> Iterator[str]:
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta

        return gen()

    return response.choices[0].message.content or ""


# Session state init (load history once)
if "messages" not in st.session_state:
    loaded_history = load_json(path_f_history)
    if loaded_history:
        st.session_state.messages = loaded_history
    else:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "너는 드와이트 슈루트(Dwight Schrute)처럼 말하는 챗봇이야. "
                    "짧고 단호하게, 가끔 경고/체크리스트 톤으로 답해."
                ),
            },
            {"role": "assistant", "content": "안녕. 보고해."},
        ]


# Render history
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("무엇을 도와줄까?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("생각 중..."):
            try:
                result = call_openai(
                    client=client,
                    model="gpt-4o",
                    messages=st.session_state.messages,
                    stream=use_streaming,
                )
            except Exception as e:
                result = f"OpenAI 호출 중 오류: `{e}`"

        full_response = ""
        if isinstance(result, str):
            full_response = result
            message_placeholder.markdown(full_response)
        else:
            for delta in result:
                full_response += delta
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_json(path_f_history, st.session_state.messages)

