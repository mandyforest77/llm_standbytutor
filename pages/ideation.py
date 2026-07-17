import os
import streamlit as st
import base64

# --- 0. Streamlit 설정 및 API 키 처리 ---
if "API_KEY" in st.secrets:
    api_key = st.secrets["API_KEY"]
else:
    api_key = st.session_state.get("user_api", "")

# DALL-E 객체 초기화용 변수
dalle = None

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        # API Key가 있을 때만 오류 없이 내부 패키지를 안전하게 로드합니다.
        from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
        dalle = DallEAPIWrapper(model="dall-e-3")
    except Exception as e:
        st.error(f"DALL-E 로드 중 오류가 발생했습니다: {e}")

# --- 1. UI 스타일 및 타이틀 (첫 번째 코드 완벽 복원) ---
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

st.markdown('<h1 class="beige-title">StandbyTutor 🤖 - 아이디에이션</h1>', unsafe_allow_html=True)


# --- 2. 인사말 및 안내 문구 세팅 ---
if api_key:
    response_content = "아이디에이션 페이지에 오신 것을 환영합니다! 아래에서 키워드를 입력하여 DALL-E 3 이미지 생성을 시작해 보세요."
else:
    response_content = "안녕하세요! 왼쪽 사이드바나 메인 페이지에서 API Key를 입력하시면 DALL-E 이미지 생성 서비스를 이용하실 수 있습니다."


# --- 3. 메인 이미지 및 인사말 레이아웃 처리 (첫 번째 코드 완벽 복원) ---
# 메인 파일과 같은 폴더 또는 상위 폴더에 있는 이미지를 찾기 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
# pages 폴더 안에 있으므로, 이미지 파일이 프로젝트 최상위에 있다면 '../standbytutor_main.JPG'로 찾습니다.
image_path = os.path.join(current_dir, '..', 'standbytutor_main.JPG')

# 만약 위 경로로 안 찾아지면 pages 폴더 안도 찾아보는 방어 코드
if not os.path.exists(image_path):
    image_path = os.path.join(current_dir, 'standbytutor_main.JPG')

if os.path.exists(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
        img = base64.b64encode(data).decode()
    
    col1, col2 = st.columns(2)
    with col1:
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
    # 이미지가 없을 때만 상단 알림창으로 대체
    st.info(response_content)
    st.warning("⚠️ 'standbytutor_main.JPG' 이미지를 찾을 수 없습니다. 파일 위치를 확인해 주세요.")


# --- 4. 사이드바 상태 표시 ---
if api_key:
    st.sidebar.success("🔑 API Key 연동 완료")
else:
    st.sidebar.warning("🔒 API Key 입력 필요")


# --- 5. 여기서부터 기존에 작성하셨던 이미지 생성 입력창 및 버튼 코드를 넣어주세요 ---
st.write("---") # 구분선

# 예시 구성 (질문자님의 기존 화면 기획에 맞게 수정하여 쓰시면 됩니다)
prompt = st.text_input("생성할 이미지의 설명(프롬프트)을 입력하세요:", disabled=not api_key)

if st.button("이미지 생성하기", disabled=not api_key):
    if dalle is not None:
        with st.spinner("DALL-E가 이미지를 그리는 중입니다..."):
            try:
                # 여기에 기존 생성 로직 반영
                # image_url = dalle.run(prompt)
                # st.image(image_url)
                pass
            except Exception as e:
                st.error(f"이미지 생성 중 오류 발생: {e}")
