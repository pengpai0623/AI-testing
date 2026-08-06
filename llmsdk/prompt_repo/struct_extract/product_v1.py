from llmsdk.prompt_repo.base_template import BasePromptTemplate


# 根据不同场景创建子类，避免重复赋值
class ProductV1(BasePromptTemplate):
    version = "v1"
    scene = "商品信息结构化抽取"
    template = """
# 角色：{role}
# 强制规则
{rules}
# 待解析商品文案
{input_text}
"""


# 全局实例，外部直接导入使用
product_extract_prompt_v1 = ProductV1()

"""
    示例
    prompt1 = product_extract_prompt_v1.render(
        role="商品信息抽取专家",
        rules="仅输出纯净JSON，无多余文字、解释、markdown代码块，key为name/price/tags，price纯数字无单位，无多余文字",
        input_text="",
    )
    res = llmstructclient.struct_chat(
        prompt="太子保温杯仅售99元，保温效果好",
        schema_cls=ProductInfo,
        system_prompt=prompt1,
    )
    print(res.model_dump_json())
"""
