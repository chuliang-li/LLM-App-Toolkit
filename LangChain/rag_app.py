# app.py
import streamlit as st
import os
from data_prep import generate_rag_data, load_and_vectorize_data
from rag_core import get_rag_chain, get_no_rag_chain, get_dashscope_api_key
from langchain_core.messages import HumanMessage

# Check DashScope API key
try:
    get_dashscope_api_key()
except ValueError as e:
    st.error(f"请设置您的 DashScope API 密钥。错误: {e}\n您可以通过在终端运行 'export DASHSCOPE_API_KEY=\"YOUR_KEY_HERE\"' 来设置。")
    st.stop()

st.set_page_config(layout="wide", page_title="RAG vs. No RAG 对比工具")

st.title("💡 RAG vs. No RAG 对比工具")
st.write("输入一个问题，点击查询，查看有RAG（检索增强生成）和没有RAG的语言模型如何回答。")

# --- Generate RAG data and initialize vector store ---
st.sidebar.header("数据管理")
if st.sidebar.button("生成/更新 RAG 知识库数据"):
    generate_rag_data()
    st.sidebar.success("知识库数据已生成/更新。")
    # After generating data, we also want to rebuild the vector store
    # So we'll force a reload by clearing the cache
    if 'vectorstore' in st.session_state:
        del st.session_state['vectorstore']
    # Add a message to prompt user to refresh if needed
    st.sidebar.warning("知识库数据更新后，请刷新页面或等待自动重新加载以重建向量存储。")

# Cache resources to avoid re-loading on every interaction
@st.cache_resource
def load_resources():
    st.info("正在加载或创建向量存储...这可能需要一些时间。")
    vectorstore_instance = load_and_vectorize_data()
    rag_chain_instance = get_rag_chain(vectorstore_instance)
    no_rag_chain_instance = get_no_rag_chain()
    st.success("资源加载完成！")
    return vectorstore_instance, rag_chain_instance, no_rag_chain_instance

vectorstore, rag_chain, no_rag_chain = load_resources()


# --- User Input ---
# 使用 st.session_state 来保存 user_question 的值，以便按钮点击后仍能访问
if 'user_question' not in st.session_state:
    st.session_state.user_question = "量子计算的主要模型是什么？"

user_question_input = st.text_input("请输入您的问题：", st.session_state.user_question, key="question_input")
st.session_state.user_question = user_question_input # 更新 session_state

# 添加查询按钮
query_button = st.button("查询")

# 只有当点击了查询按钮或用户在输入框按了回车键（这里仅响应按钮）
# 或者当页面首次加载且user_question_input有值时，为了初始化显示
if query_button and st.session_state.user_question:
    st.markdown("---")
    st.subheader("回答对比")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🚀 **有 RAG 的回答**")
        with st.spinner("RAG 正在思考中..."):
            try:
                # 确保rag_chain的输入是字典，并且键与链期望的一致 (通常是 'query')
                rag_response = rag_chain.invoke({"query": st.session_state.user_question})
                st.write(rag_response["result"])
                st.markdown("---")
                st.markdown("**检索到的相关上下文:**")
                if "source_documents" in rag_response and rag_response["source_documents"]:
                    for i, doc in enumerate(rag_response["source_documents"]):
                        st.text(f"文档 {i+1}: {doc.page_content[:200]}...")
                else:
                    st.text("未检索到相关文档。")
            except Exception as e:
                st.error(f"RAG 回答出错: {e}")

    with col2:
        st.warning("🧠 **没有 RAG 的回答 (纯 LLM)**")
        with st.spinner("纯 LLM 正在思考中..."):
            try:
                no_rag_response = no_rag_chain.invoke([HumanMessage(content=st.session_state.user_question)])
                st.write(no_rag_response.content)
            except Exception as e:
                st.error(f"纯 LLM 回答出错: {e}")

    st.markdown("---")
    st.subheader("分析与结论")
    st.write(
        "通过对比，您可以观察到 RAG 如何利用外部知识库来提供更准确、更具体、减少幻觉的回答。 "
        "没有 RAG 的模型完全依赖其内部训练数据，可能无法回答特定领域的问题，或者产生不准确的信息。"
    )

st.markdown("---")
st.sidebar.info("请确保您的 `DASHSCOPE_API_KEY` 环境变量已设置。")
st.sidebar.markdown("© 2025 RAG 演示程序")