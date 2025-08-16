import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent # 引入 create_react_agent
from langchain.tools import Tool # ReAct Agent 通常需要手动封装 Tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain
from langchain_openai import ChatOpenAI # ChatOpenAI 也可以用于 ReAct
from langchain_core.prompts import PromptTemplate # ReAct 通常使用 PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage # 用于历史消息，但此处ReAct版本简化处理

# --- Streamlit UI ---
st.set_page_config(page_title="ReAct Agent 调试器", layout="wide")
st.title("ReAct Agent 调试器")
st.write("输入您的问题，ReAct Agent 的思考-行动-观察过程将在后台终端显示，最终答案将在这里显示。")
st.warning("注意：ReAct Agent 对模型的输出格式有严格要求，有时可能因格式不符而中断。")

# 加载环境变量
load_dotenv()
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope_api_key:
    st.error("DASHSCOPE_API_KEY 环境变量未设置。请在 .env 文件中或系统环境变量中设置。")
    st.stop()

# --- 初始化LLM和工具 ---
# ReAct Agent 通常也使用 ChatOpenAI，但需要确保模型能理解并遵循 ReAct 格式
llm = ChatOpenAI(
    model_name="qwen3-coder-30b-a3b-instruct",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=dashscope_api_key,
    temperature=0
)

# ReAct Agent 的工具需要通过 Tool 类进行封装
# load_tools(["llm-math", "wikipedia"], llm=llm) 返回的是 LangChain 的 BaseTool 列表
# 对于 create_react_agent，通常需要传入 Tool 类的实例
wikipedia_tool = WikipediaAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm)

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia_tool.run,
        description="""一个维基百科的封装工具。当你需要查询一般知识或事实时使用。
        输入应该是一个字符串，表示你要搜索的查询词。
        例如：'美国总统'"""
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="""一个用于执行数学计算的工具。当你需要回答有关数学的问题时非常有用。
        输入应该是一个字符串，表示一个数学表达式，例如 '10 * 5 + 2'"""
    )
]

# --- 创建 ReAct Agent ---
# ReAct Agent 的提示模板非常关键，它定义了 LLM 必须遵循的格式
# AgentExecutor 会根据这个格式来解析 LLM 的输出
react_prompt_template = PromptTemplate.from_template("""
回答以下问题，尽你所能。你可以使用以下工具：

{tools}

请使用以下格式：

问题: 你必须回答的输入问题
思考: 你应该总是思考接下来做什么
行动: 要执行的行动，必须是 [{tool_names}] 中的一个
行动输入: 行动的输入
观察: 行动的结果
... (这个思考/行动/行动输入/观察可以重复多次)
思考: 我现在知道了最终答案
最终答案: 原始输入问题的最终答案

开始!

问题: {input}
思考:{agent_scratchpad}
""")

# 创建 ReAct Agent
# create_react_agent 期望一个 PromptTemplate
agent = create_react_agent(llm, tools, react_prompt_template)

# 设置 verbose=True 以便在后台终端打印详细日志
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Session State 初始化和管理 ---
if "final_answer_react" not in st.session_state:
    st.session_state.final_answer_react = ""
if "running_agent_react" not in st.session_state:
    st.session_state.running_agent_react = False

# 用户输入框
user_query_react = st.text_input("输入你的问题 (ReAct Agent):", "北京的人口总数乘以5是多少？", disabled=st.session_state.running_agent_react, key="react_query")

if st.button("运行 ReAct Agent", disabled=st.session_state.running_agent_react, key="run_react_btn"):
    if user_query_react:
        st.session_state.running_agent_react = True
        st.session_state.final_answer_react = "" # 清空之前的答案

        # 在Agent运行期间显示一个Spinner
        with st.spinner("ReAct Agent 正在思考中...请查看后台终端获取详细过程。"):
            try:
                # 使用阻塞式调用，等待结果
                result = agent_executor.invoke({"input": user_query_react})
                # 提取最终答案
                st.session_state.final_answer_react = result.get("output", "未找到最终答案。")
            except Exception as e:
                st.error(f"运行 ReAct Agent 时出错: {e}")
                st.session_state.final_answer_react = f"发生错误: {e}\n这通常是由于LLM返回的格式不符合Agent预期导致的。"
            finally:
                st.session_state.running_agent_react = False
                st.success("ReAct Agent 执行完毕！")
    else:
        st.warning("请输入一个问题。")

# 在页面上显示最终答案
if st.session_state.final_answer_react:
    st.subheader("ReAct Agent 最终答案:")
    st.info(st.session_state.final_answer_react)