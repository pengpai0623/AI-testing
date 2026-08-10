from pydantic import BaseModel, ConfigDict, Field

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.client.llm_struct_client import LLMStructClient
from llmsdk.prompt_repo.prompt_factory import (
    get_chat_summary_template,
    get_code_analyze_template,
    get_product_extract_template,
)
from llmsdk.utils.prompt_safe import check_prompt_injection

llm_client = LLMBaseClient()
llmstructclient = LLMStructClient()


class ProductInfo(BaseModel):

    model_config = ConfigDict(populate_by_name=True)
    name: str = Field(alias="商品名称")
    price: int = Field(alias="售价")
    tags: list[str] = Field(alias="标签")


# 1.对话总结调用示例
def run_summary_demo():
    raw_text = "用户想要350ml售价59元办公保温杯，客服推荐店内现货太子保温杯"
    risk, clean_txt = check_prompt_injection(raw_text)
    if risk:
        print("总结请求触发注入拦截")
        return
    tpl = get_chat_summary_template()
    sys_prompt = tpl.render(source_text=clean_txt, limit_word="60字以内", output_format="纯段落")
    res = llm_client.chat_single(prompt="总结这段对话", system_prompt=sys_prompt)
    print("对话总结结果：\n", res)


# 2.代码排错调用示例
def run_code_demo():
    bug_code = """
def calc_total(price, num):
    return price + num
print(calc_total("50",2))
"""
    risk, clean_code = check_prompt_injection(bug_code)
    if risk:
        print("代码请求触发注入拦截")
        return
    tpl = get_code_analyze_template()
    sys_prompt = tpl.render(
        role="Python代码审计工程师",
        base_rules="定位语法、类型、逻辑错误，解释报错原理",
        cot_rules="""
            1. 第一步：梳理代码整体功能、入参定义、完整执行流程；
            2. 第二步：逐行解析变量数据类型、运算逻辑；
            3. 第三步：精准定位报错代码行，解释异常产生原因；
            4. 第四步：输出完整修复后的可运行代码；
            """,
        output_format="输出分为【推理过程】【修复代码】两块",
    )
    res = llm_client.chat_single(prompt=f"排查代码：\n{clean_code}", system_prompt=sys_prompt)
    print("代码排错结果：\n", res)


# 3.商品结构化抽取示例
def run_extract_demo():
    raw_text = "无线蓝牙耳机269元，适用日常通勤、运动场景"
    risk, clean_txt = check_prompt_injection(raw_text)
    if risk:
        print("抽取请求触发注入拦截")
        return
    tpl = get_product_extract_template()
    sys_prompt = tpl.render(
        role="商品信息抽取专家",
        rules="提取商品名称、价格、标签，输出标准JSON",
        shot_examples='输入：运动手环99，健身、睡眠监测；输出：{"name":"运动手环","price":99,"tags":["健身","睡眠监测"]}',
    )
    res = llmstructclient.struct_chat(
        prompt=f"解析文本：{clean_txt}",
        schema_cls=ProductInfo,
        system_prompt=sys_prompt,
    )
    print("商品抽取结果：\n", res)


if __name__ == "__main__":
    run_summary_demo()
    run_code_demo()
    run_extract_demo()
