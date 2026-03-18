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

st.set_page_config(page_title="Ideation", layout="wide")
