import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate # 虽然这个版本没用到，但保留不影响
import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_csv_agent 
from io import BytesIO # 导入 BytesIO

load_dotenv()
apk_key_ali = os.getenv('DASHSCOPE_API_KEY')

st.set_page_config(page_title="智能数据分析助手 📊")
st.header("智能数据分析助手 📊")


# 独立生成 LLM
llm = ChatOpenAI(
    model_name="qwen3-coder-30b-a3b-instruct",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=apk_key_ali,
    temperature=0
)

# 文件上传器
uploaded_file = st.file_uploader("上传你的 CSV 文件", type=["csv"])

if uploaded_file is not None:
    # 关键修改：将 uploaded_file 的内容读取到 BytesIO 中
    # 这样可以多次从 BytesIO 对象中读取，而不会影响原始文件指针
    file_content = uploaded_file.getvalue()
    csv_file_like_object = BytesIO(file_content)

    # 先使用 BytesIO 对象来预览数据
    try:
        df_preview = pd.read_csv(csv_file_like_object)
        st.write("已上传数据预览:")
        st.dataframe(df_preview.head())

        # 检查 DataFrame 是否为空（预览用）
        if df_preview.empty:
            st.error("上传的 CSV 文件为空，或者没有可解析的列。请确保文件包含数据。")
        else:
            # 重置 BytesIO 对象的文件指针到开头，以供 create_csv_agent 使用
            csv_file_like_object.seek(0) 
            
            # 传入 BytesIO 对象给 create_csv_agent
            agent = create_csv_agent(
                llm,
                csv_file_like_object,  # 传入 BytesIO 对象
                verbose=True,
                allow_dangerous_code=True,
            )

            # 用户输入框
            user_question = st.text_input("输入你的问题（例如：总销售额是多少？哪个产品的销售额最高？）")

            if user_question:
                with st.spinner("正在分析中..."):
                    try:
                        # 运行 Agent 并获取回答
                        response = agent.run(user_question)
                        st.success("分析完成！")
                        st.write(response)
                    except Exception as e:
                        st.error(f"分析过程中发生错误: {e}")
                        st.info("请尝试更具体的问题，或检查 CSV 文件格式。Agent 在此模型下可能会对某些问题无响应。")
    except pd.errors.EmptyDataError:
        st.error("上传的 CSV 文件没有可解析的列或为空。请检查文件内容。")
    except Exception as e:
        st.error(f"处理文件时发生错误: {e}") # 更改错误消息，更通用

st.markdown("---")
st.markdown("由 LangChain Agent & Streamlit 提供支持")