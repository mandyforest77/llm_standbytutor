import os
import streamlit as st

# --- 1. API Key 연동 처리 ---
if "API_KEY" in st.secrets:
    api_key = st.secrets["API_KEY"]
else:
    api_key = st.session_state.get("user_api", "")

# 전역 변수로 dalle 선언 (Key가 없을 때는 우선 None으로 설정)
dalle = None

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        # API Key가 확실히 있을 때만 패키지를 불러오고 객체를 생성합니다.
        from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
        dalle = DallEAPIWrapper(model="dall-e-3")
    except Exception as e:
        st.error(f"DALL-E 로드 중 오류가 발생했습니다: {e}")
else:
    # Key가 없어도 화면이 멈추지 않고, 사용자에게 부드럽게 안내합니다.
    st.warning("🔒 OpenAI API Key가 입력되지 않았습니다. 메인 페이지에서 Key를 입력하시면 아이디에이션(이미지 생성) 기능을 사용하실 수 있습니다.")

# --- 2. 기존 UI 코드 시작 ---
# (이 아래부터 기존에 있던 ideation.py의 UI 및 텍스트 레이아웃 코드가 그대로 이어지면 됩니다.)
