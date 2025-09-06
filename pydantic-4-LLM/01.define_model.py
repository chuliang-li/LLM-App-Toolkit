import json
from pydantic import BaseModel, Field
from typing import  Literal

# 程序用途：
# 1. 定义Pydantic邮件提取数据结构
# 2. 打印BaseModel.model_json_schema方法输出数据结构的描述（用于输入LLM）

class CustomerIssue(BaseModel):
    """
    客户问题 Pydantic 模型，用于定义从邮件中提取的数据结构。
    """
    customer_name: str = Field(..., description="客户的姓名")
    product: str = Field(..., description="客户遇到问题的产品名称")
    issue_description: str = Field(..., description="客户问题的详细描述")
    priority: Literal["低", "中", "高", "紧急"] = Field(..., description="问题的优先级：低、中、高或紧急")
    assigned_department: str = Field(..., description="问题应分配到的部门")


schema = CustomerIssue.model_json_schema()
schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

print( schema_str)