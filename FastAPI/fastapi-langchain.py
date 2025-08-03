from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv() 
apk_key_ali = os.getenv('DASHSCOPE_API_KEY')

app = FastAPI()
# 初始化LLM
llm = ChatOpenAI(
    model_name="qwen3-coder-30b-a3b-instruct",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=apk_key_ali
)

# 定义Prompt模板
template = """
你是一个专业的助手，请根据以下信息提供详细回答：

用户问题: {question}

请用清晰易懂的方式回答这个问题。
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# 创建链
chain = LLMChain(llm=llm, prompt=prompt)



def get_item_info_from_DB(item_id : int):
    # 模拟数据库查询操作
    return item_id+10


@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):  # 默认值使其成为可选参数
    # 这里通常会从数据库获取数据，使用 skip 和 limit
    return {"skip": skip, "limit": limit}

@app.get("/items/{item_id}")
async def read_item(item_id : int) -> dict:
    # 1. 从http请求得到item_id参数
    # 2. 从数据库里查询item_id对应的信息
    # 3. 返回查询结果
    item_info=get_item_info_from_DB(item_id)
    return {"item_info": item_info}

@app.get("/ai/")
def ask_ai(question: str):
    """调用大模型回答用户问题"""
    response =  chain.run(question=question)
    return {"question": question, "answer": response}
