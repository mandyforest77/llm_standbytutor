import os
import streamlit as st
# 올바른 랭체인 DALL-E 래퍼 임포트 경로
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

# --- [필수] 다른 서브 페이지에서도 API Key를 연동하기 위한 코드 ---
if "API_KEY" in st.secrets:
    api_key = st.secrets["API_KEY"]
else:
    api_key = st.session_state.get("user_api", "")

if api_key:
    # 이 부분이 실행되어야 내부 DallEAPIWrapper가 에러 없이 작동합니다.
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.error("🔒 OpenAI API Key가 없습니다. 메인 페이지에서 Key를 먼저 입력해주세요.")
    st.stop() # 키가 없으면 아래 코드를 실행하지 않고 멈춥니다.
# --------------------------------------------------------

# 기존 26번째 줄에 있던 객체 생성 코드
dalle = DallEAPIWrapper(model="dall-e-3")
