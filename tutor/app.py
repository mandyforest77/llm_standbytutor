import os
from langchain_openai import ChatOpenAI
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from typing import List, Dict, Any, Literal
import requests
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import LLMToolEmulator, TodoListMiddleware, HumanInTheLoopMiddleware
from pydantic import BaseModel, Field
from dataclasses import dataclass
from langchain.agents.middleware import before_model, wrap_model_call
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from PIL import Image
import base64

api_key = st.secrets["API_KEY"]
os.environ["OPENAI_API_KEY"]=api_key
os.environ["LANGCHAIN_TRACING_V2"] = "false"


# 페이지 설정
st.markdown("""
    <style>
    .beige-title {
        color: #008080 !important; /* !important를 추가해 우선순위를 높임 */
    }
    </style>
    """, unsafe_allow_html=True)

# 2. HTML 적용
st.markdown('<h1 class="beige-title">StandbyTutor 🤖</h1>', unsafe_allow_html=True)

model = ChatOpenAI(model_name="gpt-4o")
# response = model.invoke([HumanMessage(content="우리의 손님이 오셨으니, 친절한 인사말을 해주세요. 저한테 대답하지 말고, 손님께 인사해주세요.")])
# st.info(response.content) # 인사말을 info 박스에 담아 강조

current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'standbytutor_main.JPG')

with open(image_path, "rb") as f:
    data = f.read()
    img= base64.b64encode(data).decode()


# --- 3. 컬럼 분할 및 이미지 표시 ---
# 2분할 컬럼 생성
st.markdown(
    """
    <style>
    [data-testid="stImage"] img {
        border-radius: 20px; /* 둥글기 정도 설정 (px 또는 %) */
        /* 필요시 테두리 추가: border: 2px solid #ddd; */
    }
    </style>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)

with col1:
    st.image(f"data:image/jpg;base64,{img}", use_column_width=True)

# with col2:
#     # CSS를 사용하여 col2의 내용을 아래로 정렬
#     st.markdown(
#         f"""
#         <div style="display: flex; flex-direction: column; justify-content: flex-end; height: 100%; min-height: 300px;">
#             <p style="margin: 0;">{response.content}</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )



# 사이드바 설정
st.sidebar.success("위의 필요한 기능을 선택하세요.")

# --- API Key 입력 세션 (가장 위로 올리는 것이 좋습니다) ---
user_api = st.text_input("OpenAI API Key를 입력해주세요", type="password")
if user_api:
    st.session_state["user_api"] = user_api
    st.sidebar.success("API Key가 저장되었습니다.")
else:
    st.warning("API Key를 입력해주세요. 그래야 다른 페이지에서 기능을 사용할 수 있습니다.")

# --- 메인 화면 ---
# 1. 타이틀 베이지 색상 적용


