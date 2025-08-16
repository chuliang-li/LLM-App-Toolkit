import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

# --- Streamlit UI ---
st.title("LangChain Agent 调试器 (简化版)")
st.write("输入您的问题，Agent 的思考过程将在后台终端显示，最终答案将在这里显示。")

# 加载环境变量
load_dotenv()
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope_api_key:
    st.error("DASHSCOPE_API_KEY环境变量未设置。请在.env文件中或系统环境变量中设置。")
    st.stop()

# --- 初始化LLM和工具 ---
llm = ChatOpenAI(
    model_name="qwen3-coder-30b-a3b-instruct",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=dashscope_api_key,
    temperature=0
)
tools = load_tools(["llm-math", "wikipedia"], llm=llm)

# --- 创建 Agent ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能助手。"),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
agent = create_openai_tools_agent(llm, tools, prompt)

# 设置 verbose=True 以便在后台终端打印详细日志
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Session State 初始化和管理 ---
if "final_answer" not in st.session_state:
    st.session_state.final_answer = ""
if "running_agent" not in st.session_state:
    st.session_state.running_agent = False

# 用户输入框
user_query = st.text_input("输入你的问题:", "北京的人口总数乘以5是多少？", disabled=st.session_state.running_agent)

if st.button("运行 Agent", disabled=st.session_state.running_agent):
    if user_query:
        st.session_state.running_agent = True
        st.session_state.final_answer = "" # 清空之前的答案

        # 在Agent运行期间显示一个Spinner
        with st.spinner("Agent 正在思考中...请查看后台终端获取详细过程。"):
            try:
                # 使用阻塞式调用，等待结果
                result = agent_executor.invoke({"input": user_query})
                # 提取最终答案
                st.session_state.final_answer = result.get("output", "未找到最终答案。")
            except Exception as e:
                st.error(f"运行 Agent 时出错: {e}")
                st.session_state.final_answer = f"发生错误: {e}"
            finally:
                st.session_state.running_agent = False
                st.success("Agent 执行完毕！")
    else:
        st.warning("请输入一个问题。")

# 在页面上显示最终答案
if st.session_state.final_answer:
    st.subheader("最终答案:")
    st.info(st.session_state.final_answer)
