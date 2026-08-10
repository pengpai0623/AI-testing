from llmsdk.prompt_repo.base_template import BasePromptTemplate


# 根据不同场景创建子类，避免重复赋值
class ProductExtractV2(BasePromptTemplate):
    version = "v2_fewshot"
    scene = "商品信息结构化抽取"
    template = """
# 角色：{role}
# 强制规则
{rules}
# 标准样例
{shot_examples}
"""


# 全局实例，外部直接导入使用
product_extract_prompt_v2 = ProductExtractV2()


"""
    # 示例
    prompt2 = product_extract_prompt_v2.render(
        role="商品信息抽取专家",
        rules="仅输出纯净JSON，无多余文字、解释、markdown代码块，key为name/price/tags，price纯数字无单位，无多余文字",
        shot_examples='''
            输入：无线蓝牙耳机269元，标签数码、耳机
            输出：{"name":"无线蓝牙耳机","price":269,"tags":["数码","耳机"]}''',

    )

    res = llmstructclient.struct_chat(
        prompt="太子保温杯仅售99元，保温效果好",
        schema_cls=ProductInfo,
        system_prompt=prompt2   ,
    )
    print(res.model_dump_json())
"""
