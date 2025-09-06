from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
import os
import json
from typing import Optional, Literal
from pydantic import BaseModel, Field
import json
import sqlite3

EMAIL_DIR = "./customer_emails"
DATABASE_NAME = "customer_issues.db"

class CustomerIssue(BaseModel):
    """
    客户问题 Pydantic 模型，用于定义从邮件中提取的数据结构。
    """
    customer_name: str = Field(..., description="客户的姓名")
    product: str = Field(..., description="客户遇到问题的产品名称")
    issue_description: str = Field(..., description="客户问题的详细描述")
    priority: Literal["低", "中", "高", "紧急"] = Field(..., description="问题的优先级：低、中、高或紧急")
    assigned_department: str = Field(..., description="问题应分配到的部门")


def setup_llm():
    """
    加载环境变量并配置 DashScope LLM。
    """
    load_dotenv()
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
       
    llm = ChatOpenAI(
        model_name="qwen3-coder-30b-a3b-instruct",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=dashscope_api_key,
        temperature=0
    )
    return llm

def extract_issue_with_llm(llm, email_content: str) -> Optional[CustomerIssue]:
    """
    使用 DashScope LLM 结合 Pydantic 模型从邮件内容中提取客户问题。
    """
    schema = CustomerIssue.model_json_schema()
    schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

    # 步骤二：生成提示词模版，并往模版里填入具体内容后生成提示词
    # 完整且清晰的提示模板
    template = """
你是一个专业的客户服务数据分析师。你的任务是从客户的邮件内容中提取关键信息，
并严格按照提供的 JSON Schema 格式输出。

JSON Schema 定义：
```json
{schema}
客户邮件内容：

{email_content}
请根据上述 JSON Schema，从客户邮件中提取信息，并只输出一个符合该格式的 JSON 对象。
不要包含任何额外的文字、解释或代码块分隔符（例如```json）。

JSON 输出:
"""
    prompt = PromptTemplate.from_template(template)

    # 在提示词模版里填入实际内容，生成提示词，准备发送给 LLM
    formatted_prompt = prompt.format(schema=schema_str, email_content=email_content)

    print(f"\n--- 发送给 LLM 的提示 (部分展示) ---\n{formatted_prompt[:500]}...\n----------------------------------")

    # 步骤三：把指令和提示词发给LLM - 调用 LLM API
    try:
        # 使用 LangChain 的 invoke 方法发送请求
        # Chat models expect a list of messages, not a single string
        llm_raw_output = llm.invoke(
            [
                SystemMessage(content="你是一个JSON数据提取专家。"),
                HumanMessage(content=formatted_prompt)
            ]
        ).content

        llm_raw_output = llm_raw_output.strip()
        print(f"--- LLM 原始响应 ---\n{llm_raw_output}\n--------------------")

        # 步骤四：使用 Pydantic 解析和验证 LLM 输出
        data = json.loads(llm_raw_output)
        issue = CustomerIssue.model_validate(data)
        print("--- Pydantic 验证成功！---")
        return issue
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None


def insert_issue_into_db(issue: CustomerIssue):
    """
    将 Pydantic CustomerIssue 对象插入到数据库中。
    """
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO issues (customer_name, product, issue_description, priority, assigned_department)
        VALUES (?, ?, ?, ?, ?)
    """, (
        issue.customer_name,
        issue.product,
        issue.issue_description,
        issue.priority,
        issue.assigned_department
    ))
    conn.commit()
    conn.close()
    print(f"成功将客户问题 '{issue.customer_name}' 插入到数据库。")

#-------------上面是函数定义部分--------------
#  以下是主程序部分

# 步骤一：连接大模型，生成大模型实例
llm = setup_llm()

#  从customer_emails目录读取所有邮件文件
email_files = [os.path.join(EMAIL_DIR, f) for f in os.listdir(EMAIL_DIR) if f.endswith(".txt")]

#  这个循环对每个邮件文件读取后进行处理
for email_file_path in email_files:
    print(f"\n===== 处理文件: {os.path.basename(email_file_path)} =====")
    with open(email_file_path, "r", encoding="utf-8") as f:

        # 读取每个邮件的内容
        email_content = f.read()

        # 使用 LLM 和 Pydantic 提取信息
        customer_issue = extract_issue_with_llm(llm, email_content)

        #步骤五：把数据存入数据库
        if customer_issue:
            print(f"提取到的信息: {customer_issue.model_dump_json(indent=2)}")
            # 存储到数据库
            insert_issue_into_db(customer_issue)
        else:
            print(f"未能从 {os.path.basename(email_file_path)} 提取有效信息。")

