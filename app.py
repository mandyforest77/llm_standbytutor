import os
import streamlit as st
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# --- 0. Streamlit 설정 및 API 키 처리 ---
# [필수 고침] st.secrets에 API_KEY가 없을 경우를 대비한 안전 장치
if "API_KEY" in st.secrets:
    api_key = st.secrets["API_KEY"]
else:
    # secrets에 없을 경우, 하단 text_input에서 입력한 값을 사용하도록 연동
    api_key = st.session_state.get("user_api", "")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# --- 1. UI 스타일 및 타이틀 ---
st.markdown("""
    <style>
    .beige-title {
        color: #008080 !important;
    }
    [data-testid="stImage"] img {
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="beige-title">StandbyTutor 🤖</h1>', unsafe_allow_html=True)


# --- 2. LLM 모델 초기화 & 인사말 ---
# API 키가 등록된 상태에서만 모델이 작동하도록 감싸야 에러가 안 납니다.
response_content = "안녕하세요! StandbyTutor에 오신 것을 환영합니다. 왼쪽 사이드바나 하단에서 API Key를 입력하시면 튜터링 서비스가 시작됩니다."

if api_key:
    try:
        model = ChatOpenAI(model_name="gpt-4o")
        response = model.invoke([
            HumanMessage(content="우리의 손님이 오셨으니, 친절한 인사말을 해주세요. 저한테 대답하지 말고, 손님께 인사해주세요.")
        ])
        response_content = response.content
        st.info(response_content) # 주석 해제하여 인사말 출력
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
else:
    st.info(response_content)


# --- 3. 메인 이미지 처리 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'standbytutor_main.JPG')

# 이미지 파일 존재 여부 확인 후 base64 인코딩
if os.path.exists(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
        img = base64.b64encode(data).decode()
    
    col1, col2 = st.columns(2)
    with col1:
        # Streamlit 최신 버전 규격 반영 (use_column_width=True 대신 use_container_width=True)
        st.image(f"data:image/jpg;base64,{img}", use_container_width=True)
    
    with col2:
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; justify-content: flex-end; height: 100%; min-height: 200px;">
                <p style="margin: 0; font-size: 16px; font-weight: bold;">{response_content}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.warning("⚠️ 'standbytutor_main.JPG' 이미지를 찾을 수 없습니다. 파일명을 확인해 주세요.")


# --- 4. 사이드바 및 API Key 입력 세션 ---
st.sidebar.success("위의 필요한 기능을 선택하세요.")

# 사용자가 직접 입력할 수 있는 입력창
user_api = st.text_input("OpenAI API Key를 입력해주세요", type="password", value=st.session_state.get("user_api", ""))

if user_api:
    st.session_state["user_api"] = user_api
    st.sidebar.success("API Key가 저장되었습니다.")
    # 사용자가 값을 입력하면 즉시 페이지를 새로고침하여 상단 모델에 적용시킵니다.
    if api_key != user_api:
        st.rerun()
else:
    if not api_key:
        st.warning("API Key를 입력해주세요. 그래야 다른 페이지에서 기능을 사용할 수 있습니다.")
