from typing import List, Optional

from pydantic import BaseModel, Field

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.client.llm_struct_client import LLMStructClient
from llmsdk.session.chat_session import ChatSession
from llmsdk.utils.exceptions import JsonParseError, PydanticValidateError

system_prompt = """
# 角色：结构化数据抽取工具
# 任务：从自我介绍中提取关键信息
# 强制规则
1. 只返回纯JSON，没有任何多余文字、注释、markdown；
2. JSON键固定英文：name（姓名）、age（年龄）、post（岗位）、ability（技能）；
3. age纯数字，不能带岁；
4. 缺数据填空字符串或者空列表，绝对不能编造；
5. 禁止中文key。"""

prompt = "面试官你好，我叫李孝永，今天30岁，今天应聘的职位是测试开发工程师，熟悉python和java"


# 商品抽取专用结构
class ProductInfo(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    post: str = Field(description="岗位")
    ability: List[str] = Field(description="技能")


if __name__ == "__main__":
    llm = LLMBaseClient()
    chat = ChatSession()
    llmstructclient = LLMStructClient()
    print(chat.chat("我叫lxy", llm=llm, max_completion_tokens=300))
    print(chat.chat("我叫什么", llm=llm, max_completion_tokens=300))
