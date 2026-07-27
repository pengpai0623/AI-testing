from pydantic import BaseModel, Field
from typing import List


# 商品抽取专用结构
class ProductInfo(BaseModel):
    name:str = Field(description="商品完整名称")
    price:float = Field(description="商品售价，纯数字，不要单位")
    tags: List[str] = Field(description="分类标签数组，多个标签放在列表")