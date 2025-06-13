import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# 初始化本地模型（透過 Ollama）
# llm = Ollama(model="llama3")
llm = Ollama(model="gemma3")

# 建立 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot."),
    ("human", "Question: {question}")
])

# 建立 output parser
output_parser = StrOutputParser()

# 建立 chain
chain = prompt | llm | output_parser

# 建立 Streamlit UI
st.title("Ollama + LangChain Chatbot")
input_text = st.text_input("請輸入你的問題：")

if input_text:
    with st.spinner("模型正在思考中..."):
        try:
            response = chain.invoke({"question": input_text})
            st.success("回答如下：")
            st.write(response)
        except Exception as e:
            st.error("發生錯誤：")
            st.code(str(e))

# streamlit run test_ollama_chat_app.py
