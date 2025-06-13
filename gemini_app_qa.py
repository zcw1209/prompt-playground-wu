import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from google.api_core.exceptions import ResourceExhausted
import time

# 載入 .env 檔案中的 API 金鑰等環境變數
load_dotenv()

@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

llm = load_model()

# 建立提示模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot."),
    ("human", "Question: {question}")
])

# 輸出解析器
output_parser = StrOutputParser()

# 建立 chain
chain = prompt | llm | output_parser

# Streamlit UI
st.title('LangChain + Gemini Chatbot Demo')
input_text = st.text_input("請輸入你的問題：")

if input_text:
    with st.spinner("正在思考中，請稍候..."):
        try:
            response = chain.invoke({"question": input_text})
            st.success("回答：")
            st.write(response)
        except ResourceExhausted as e:
            st.error("❌ 已超出 API 使用限制，請稍後再試或升級帳號。")
            st.code(str(e), language="bash")
        except Exception as e:
            st.error("❌ 發生未知錯誤：")
            st.code(str(e), language="python")

# To run this code, write-  streamlit run gemini_app_qa.py