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
from langchain_core.tools import Tool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

st.set_page_config(page_title="Ideation", layout="wide")

# 2. 클래스 초기화
# model을 "dall-e-3"로 지정하면 더 고품질의 이미지가 생성됩니다.
dalle = DallEAPIWrapper(model="dall-e-3")

# 3. 이미지 생성 실행
# 결과값으로 이미지 파일이 아닌 '이미지 URL' 문자열이 반환됩니다.
prompt = st.text_input("이미지 생성 실행을 위해 엔터를 눌러주세요.")
if prompt:
    image_url = dalle.run(prompt)

# 4. 결과 출력
    st.image(image_url)

