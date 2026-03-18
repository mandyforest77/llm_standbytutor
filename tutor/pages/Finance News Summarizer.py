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

load_dotenv()
st.set_page_config(page_title="네이버 경제뉴스 요약", layout="wide")

# --- 크롤링 함수 ---
def crawl_naver_economy_news(limit=15):
    url = "https://news.naver.com/section/101"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    news_list = []
    items = soup.select('.sa_text_title')
    for i, news_item in enumerate(items[:limit]):
           title = news_item.text.strip()
           link = news_item['href']
           news_list.append({'title': title, 'link': link})

    return news_list

# --- 요약 함수 ---
def summarize_news(news_item_dict): # Expects content as a dictionary
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=st.session_state["user_api"])
    prompt_template = """다음 뉴스 내용을 핵심만 4가지로 요약해줘 꼭 아래 구조로 작성해줘:
    {text}
    요약 결과:
* \n\n
* \n\n
* \n\n
* \n\n
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    url = news_item_dict["link"]
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.select_one("#newsct_article").text.strip()
    # print("a",content)
    doc = [Document(page_content=content)] # Use content_text here
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(doc)
    return news_item_dict["title"], news_item_dict["link"], summary

# news_items = crawl_naver_economy_news(15)

# for i, news_item in enumerate(news_items):
#   summary = summarize_news(news_item) # Pass the dictionary directly
#   print(f"제목: {summary[0]}\n 요약:{summary[2]} \n 링크:{summary[1]}" )

st.title("📰 네이버 경제뉴스 TOP 15 요약")
st.write("LangChain과 OpenAI를 활용하여 경제 뉴스 랭킹을 요약합니다.")

if "user_api" in st.session_state:
    if st.button("뉴스 가져오기 및 요약"):
        with st.spinner("뉴스를 크롤링하고 요약 중입니다..."):
            news_items = crawl_naver_economy_news(15)
            
            # 요약된 뉴스 데이터를 저장할 리스트
            summarized_list = []
            
            for news_item in news_items:
                summary = summarize_news(news_item)  # [제목, 링크, 요약내용] 형태라고 가정
                summarized_list.append(summary)
                
                # 화면에 즉시 표시
                st.info(f"💡 **제목:** {summary[0]}\n\n**요약:** {summary[2]}\n\n**링크:** {summary[1]}")

            # 다운로드용 텍스트 생성 (이미 요약된 데이터를 사용해 중복 계산 방지)
            download_text = "\n\n" + "\n\n".join([
                f"제목: {s[0]}\n요약: {s[2]}\n링크: {s[1]}" for s in summarized_list
            ])

            st.download_button(
                label="텍스트 파일 다운로드",
                data=download_text,
                file_name="Finance_News_Summary.txt",
                mime="text/plain"
            )
else:
    st.info("API Key가 등록되지 않았습니다. 첫 페이지에서 등록해주세요.")