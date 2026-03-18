import streamlit as st
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
import openai
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage

st.title("텍스트 리더")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt","pdf"])

if uploaded_file is not None:
    # 1. 파일 내용을 문자열로 읽기
    # uploaded_file은 BytesIO 객체이므로 decode('utf-8')이 필요합니다.
    df = uploaded_file.read().decode("utf-8")

    pages= [Document(page_content=df, metadata={"source": "uploaded_file"})]

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )


    texts = text_splitter.split_documents(pages)
    # print(texts[10])
    embedding_model= OpenAIEmbeddings()
    db=Chroma.from_documents(texts, embedding_model)

prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")


# 입력 위젯 예시
query = st.text_input("위의 파일에서 필요한 부분이 무엇인가요?")

if "user_api" in st.session_state:
    model = init_chat_model("gpt-4o", openai_api_key=st.session_state["user_api"])
    if query:
        if uploaded_file is not None:
            # 1. 세션 초기화 (새로운 질문일 경우 대화 기록 초기화)
            if "messages" not in st.session_state or st.session_state.get("last_query") != query:
                st.session_state.messages = []
                st.session_state.last_query = query
                
                # 초기 답변 생성
                document_chain = create_stuff_documents_chain(model, prompt)
                qa_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 10}), document_chain)
                response = qa_chain.invoke({"input": query})
                
                # 대화 기록에 추가
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

            # 2. 지금까지의 대화 기록 출력
            st.write(f"### '{query}'에 대한 대화")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # 3. 추가 피드백 받기 (st.chat_input은 루프를 다시 돌게 함)
            feedback = st.chat_input("수정하거나 추가하고 싶은 점을 알려주세요.")
            
            if feedback:
                # 피드백을 기록에 추가
                st.session_state.messages.append({"role": "user", "content": feedback})
                
                # 이전 답변들을 맥락으로 사용하여 재요청
                # (tip: messages 기록 전체를 context로 전달하면 더 정확합니다)
                refine_prompt = f"""
                이전 답변: {st.session_state.messages[-2]['content']}
                사용자 피드백: {feedback}
                Answer the following question based only on the provided context:
                """
                refined_response = model.invoke([HumanMessage(content=refine_prompt)])
                
                # 수정된 답변 기록 후 화면 갱신
                st.session_state.messages.append({"role": "assistant", "content": refined_response.content})
                st.rerun() # 화면을 즉시 새로고침하여 추가된 답변 표시

        else:
            st.warning("파일을 업로드해주세요")
else:
    st.info("API Key가 등록되지 않았습니다. 첫 페이지에서 등록해주세요.")
    