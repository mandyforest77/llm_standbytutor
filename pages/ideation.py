import os
import streamlit as st
from langchain_community.utilities import DallEAPIWrapper # 기존 import문

# --- [추가] 다른 페이지에서도 API Key를 연동하기 위한 코드 ---
if "API_KEY" in st.secrets:
    api_key = st.secrets["API_KEY"]
else:
    api_key = st.session_state.get("user_api", "")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("🔒 OpenAI API Key가 없습니다. 메인 페이지에서 Key를 먼저 입력해주세요.")
    st.stop() # API Key가 없으면 이후 코드 실행을 중단합니다.
# --------------------------------------------------------

# 기존 26번째 줄 코드
dalle = DallEAPIWrapper(model="dall-e-3")
