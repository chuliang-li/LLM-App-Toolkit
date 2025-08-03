import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI  # 关键修改
from dotenv import load_dotenv
import os

load_dotenv() 
apk_key_ali = os.getenv('DASHSCOPE_API_KEY')

st.title("基于Langchain和Streamlit的大模型演示应用")

# 使用 ChatOpenAI 替代 OpenAI
llm = ChatOpenAI(
    model_name="qwen3-coder-30b-a3b-instruct",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=apk_key_ali
)

template = """
你是一个专业的助手，请根据以下信息提供详细回答：

用户问题: {question}
背景信息: {context}

请用清晰易懂的方式回答这个问题。
"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

question = st.text_input("请输入您的问题:")
context = st.text_area("请输入相关背景信息（可选）", "")

if st.button("获取回答"):
    if question:
        with st.spinner('正在生成回答...'):
            response = chain.run(question=question, context=context)
        st.subheader("回答:")
        st.write(response)
    else:
        st.warning("请输入一个问题")
