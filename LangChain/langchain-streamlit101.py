import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv() 
api_key_sf = os.getenv('SILICONFLOW_API_KEY')

# 设置页面标题
st.title("基于Langchain和Streamlit的大模型演示应用")

# 初始化LLM
llm = OpenAI(
    model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key_sf
)

# 定义Prompt模板
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

# 创建链
chain = LLMChain(llm=llm, prompt=prompt)

# 用户输入
question = st.text_input("请输入您的问题:")
context = st.text_area("请输入相关背景信息（可选）", "")

# 处理用户请求
if st.button("获取回答"):
    if question:
        with st.spinner('正在生成回答...'):
            # 执行链
            response = chain.run(question=question, context=context)
            
        # 显示结果
        st.subheader("回答:")
        st.write(response)
    else:
        st.warning("请输入一个问题")
