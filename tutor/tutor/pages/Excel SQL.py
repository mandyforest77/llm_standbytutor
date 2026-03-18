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
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd # Import pandas
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent # Import pandas agent


# file_path = "Practice GRE.pdf"


uploaded_file = st.file_uploader("Upload a .csv or Excel file", type=["xls","xlsx","csv"])
if "user_api" in st.session_state:
    if uploaded_file is not None:
        # Load the file into a Pandas DataFrame
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

    # print(docs[100])

    # Initialize ChatOpenAI with a valid model name
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=st.session_state["user_api"]) # Corrected model name to gpt-3.5-turbo

        # Create a pandas dataframe agent to query the CSV
        agent_executer = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True) # Added allow_dangerous_code=True

        query = st.text_input("위의 파일에서 필요한 부분이 무엇인가요?")
        if query:
            response = agent_executer.invoke({"input": query}) # Changed invoke format for pandas agent
            st.write(response)

    else:
        st.write("파일을 업로드 해주세요.")
else:
    st.info("API Key가 등록되지 않았습니다. 첫 페이지에서 등록해주세요.")
# response = agent_executer.invoke({"input": "회사내에 40대는 총 몇명이며, 평균 연봉이 어떻게 되나요?"}) # Changed invoke format for pandas agent
# response = agent_executer.invoke({"input": "연봉 대답에서 칸은 무슨 칸을 봤나요?"}) # Changed invoke format for pandas agent
# response = agent_executer.invoke({"input": "40대가 가장많은 부서는 어디에요?"})
# response