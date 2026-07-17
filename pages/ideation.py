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

# --- 1. 페이지 레이아웃 설정 (원본 복원) ---
st.set_page_config(page_title="Ideation", layout="wide")

# --- 2. UI 타이틀 및 안내 메시지 ---
st.title("🎨 Ideation Image Generator")

# --- 3. 이미지 생성 UI 구성 ---
prompt = st.text_input("이미지 생성 실행을 위해 엔터를 눌러주세요.")

# --- 4. 예시 페이지와 동일한 조건문 구조 적용 ---
if "user_api" in st.session_state and st.session_state["user_api"]:
    
    # API 키 앞뒤에 생길 수 있는 눈에 안 보이는 공백을 완벽히 제거(strip)합니다.
    clean_api_key = st.session_state["user_api"].strip()
    
    if prompt:
        with st.spinner("DALL-E가 이미지를 생성하는 중입니다..."):
            try:
                # 랭체인 DALL-E 래퍼 패키지 로드
                from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
                
                # 모델 생성 시 인자에 깨끗한 API 키를 직접 명시하여 401 인증 에러를 원천 차단합니다.
                dalle = DallEAPIWrapper(
                    model="dall-e-3", 
                    openai_api_key=clean_api_key
                )
                
                # 이미지 생성 실행 및 결과 출력
                image_url = dalle.run(prompt)
                st.image(image_url)
                
            except Exception as e:
                # 상세 에러 메시지가 화면에 노출되도록 처리
                st.error(f"이미지 생성 중 오류가 발생했습니다: {e}")
else:
    # 예시 코드의 문구 스타일을 그대로 반영
    st.info("API Key가 등록되지 않았습니다. 첫 페이지에서 등록해주세요.")
