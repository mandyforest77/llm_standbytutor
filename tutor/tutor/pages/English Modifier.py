import streamlit as st
import requests
from bs4 import BeautifulSoup
from typing import Dict, TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
# from IPython.display import display, HTML, Image
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage, BaseMessage
import numpy as np
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from operator import add # Import add from operator
from langchain.tools import tool # Import the tool decorator
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_core.documents import Document # Added import
from langchain_classic.chains.summarize import load_summarize_chain # Added import

st.set_page_config(page_title="English Modifier", layout="wide")
st.title("📝 English Modifier")

# 3. 모델 및 프롬프트 설정
if "user_api" in st.session_state:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=st.session_state["user_api"])
    
    template = """
    다음 영어 문장의 문법을 교정하고, 어색한 부분을 더 자연스럽게 수정해주세요.
    수정된 문장만 출력해주세요.
    
    문장: {sentence}
    
    수정된 문장:
    """
    
    prompt = PromptTemplate(input_variables=["sentence"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # 4. 사용자 입력
    user_input = st.text_area("교정할 문장을 입력하세요:", height=150)

    if st.button("교정하기"):
        if user_input:
            with st.spinner("교정 중..."):
                # 5. 문법 교정 실행
                result = chain.run(user_input)
                st.success("교정 완료!")
                st.subheader("수정된 문장:")
                st.write(result)
        else:
            st.warning("문장을 입력해주세요.")
else:
    st.info("API Key가 등록되지 않았습니다. 첫 페이지에서 등록해주세요.")