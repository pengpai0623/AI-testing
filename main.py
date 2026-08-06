from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.client.llm_struct_client import LLMStructClient
from llmsdk.prompt_repo.code_analyze.v1 import code_analyze_v1
from llmsdk.prompt_repo.code_analyze.v2_CoT import code_analyze_cot_v2
from llmsdk.prompt_repo.prompt_factory import get_code_analyze_template
from llmsdk.prompt_repo.struct_extract.product_v1 import product_extract_prompt_v1
from llmsdk.prompt_repo.struct_extract.product_v2_fewshot import (
    product_extract_prompt_v2,
)
from llmsdk.session.chat_session import ChatSession
from llmsdk.utils.exceptions import JsonParseError, PydanticValidateError
from llmsdk.utils.logger import logger


# 商品抽取专用结构
class ProductInfo(BaseModel):

    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(alias="商品名称")
    price: int = Field(alias="售价")
    tags: list[str] = Field(alias="标签")


if __name__ == "__main__":
    llm = LLMBaseClient()
    chat = ChatSession()
    llmstructclient = LLMStructClient()
    bug_code = """
    def calc_total(price, num):
        return price + num
    print(calc_total("50", 2))
    """

    prompt2_object = get_code_analyze_template()
    prompt2 = prompt2_object.render(
        role="Python代码排错工程师",
        base_rules="排查代码错误",
        cot_rules="""
            1. 第一步：梳理代码整体功能、入参、执行流程；
            2. 第二步：逐行识别变量类型、运算逻辑；
            3. 第三步：定位异常发生的代码行，说明报错原因；
            4. 第四步：给出修改后的完整可运行代码；
            """,
        output_format="包含两部分，分步推理过程和修复后代码",
    )

    res = llm.chat_single(
        prompt=bug_code,
        system_prompt=prompt2,
    )
    print(res)
