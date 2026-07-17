import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Dict, TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import numpy as np
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from operator import add
from langchain.tools import tool
from dotenv import load_dotenv
import time
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_core.documents import Document
from langchain_classic.chains.summarize import load_summarize_chain

# --- 1. 페이지 레이아웃 설정 (기존 설정 복원) ---
st.set_page_config(page_title="Ideation", layout="wide")

# --- 2. API Key 및 DallEAPIWrapper 초기화 처리 ---
if "API_KEY" in st.secrets:
    api_key = st.secrets["API_KEY"]
else:
    api_key = st.session_state.get("user_api", "")

# 전역 변수로 dalle 선언
dalle = None

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        # API Key가 등록된 상태에서만 안전하게 패키지를 불러오고 객체를 생성합니다.
        from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
        dalle = DallEAPIWrapper(model="dall-e-3")
    except Exception as e:
        st.error(f"DALL-E 로드 중 오류가 발생했습니다: {e}")
else:
    # Key가 없을 때 화면 상단에 경고 메시지만 띄우고 아래 UI는 그대로 보여줍니다.
    st.warning("🔒 OpenAI API Key가 등록되지 않았습니다. 메인 페이지에서 Key를 먼저 등록하시면 이미지 생성 기능을 사용하실 수 있습니다.")


# --- 3. 이미지 생성 UI 및 실행 (기존 코드 완벽 복원) ---
# 기존에 작성하셨던 입력창과 조건문 구조를 그대로 유지합니다.
prompt = st.text_input("이미지 생성 실행을 위해 엔터를 눌러주세요.")

if "user_api" in st.session_state or "API_KEY" in st.secrets:  
    if prompt:
        if dalle is not None:
            with st.spinner("DALL-E가 이미지를 생성하는 중입니다..."):
                try:
                    image_url = dalle.run(prompt)
                    st.image(image_url)
                except Exception as e:
                    st.error(f"이미지 생성 중 오류가 발생했습니다: {e}")
        else:
            st.error("API Key가 유효하지 않거나 등록되지 않아 이미지를 생성할 수 없습니다.")
