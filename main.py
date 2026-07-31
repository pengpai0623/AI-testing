from typing import List

from pydantic import BaseModel, Field

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.client.llm_struct_client import LLMStructClient
from llmsdk.session.chat_session import ChatSession
from llmsdk.utils.exceptions import JsonParseError, PydanticValidateError


# 商品抽取专用结构
class ProductInfo(BaseModel):
    name: str = Field(description="商品完整名称")
    price: float = Field(description="商品售价，纯数字，不能带单位")
    tags: list[str] = Field(description="商品分类标签，字符串数组")


if __name__ == "__main__":
    llm = LLMBaseClient()
    chat = ChatSession()
    llmstructclient = LLMStructClient()

    # ==========原有4个基础测试==========
    # 测试1 单轮对话
    test1_res = llm.chat_single("你好")
    print("测试1单轮：", test1_res)

    # 测试2 多轮上下文
    test2_res1 = chat.chat("我叫xy", llm)
    print("测试2轮1：", test2_res1)
    test2_res2 = chat.chat("我叫什么", llm)
    print("测试2轮2：", test2_res2)

    # 测试3 结构化正常场景
    test_res3 = llmstructclient.struct_chat("新款无线蓝牙耳机，售价129，适合运动、通勤", schema_cls=ProductInfo)
    print("测试3结构化正常：", test_res3.name, test_res3.price, test_res3.tags)

    # 测试4 流式单轮
    test_res4 = chat.chat_stream("我叫xy", llm)
    full_text = ""
    for word in test_res4:
        full_text += word
        print("流式分片：", word)
    print("测试4完整流式：", full_text)

    # ==========新增补充测试（补齐覆盖率）==========
    # 补充1：单轮自定义system、temperature，验证参数生效
    print("\n====补充1：自定义参数单轮对话====")
    res_custom = llm.chat_single(prompt="介绍苹果", system_prompt="只用1句话回答", temperature=0.1)
    print(res_custom)

    # 补充2：ChatSession清空会话 + 超长文本压上下文截断
    print("\n====补充2：会话清空+超长上下文====")
    chat.clear_history()
    # 灌入大量文字，触发自动截断
    long_text = "重复文案" * 2000
    chat.chat(long_text, llm)
    print(chat.chat("现在复述我最开始的名字", llm))

    # 补充3：结构化强制英文key + 主动制造格式错误验证重试
    print("\n====补充3：结构化自定义规则+异常重试====")
    strict_rule = "只返回JSON，key必须为name/price/tags，禁止中文key"
    try:
        bad_input = "保温杯价格59，标签居家、水杯"
        res_struct_custom = llmstructclient.struct_chat(
            prompt=bad_input, schema_cls=ProductInfo, temperature=0.1, system_prompt=strict_rule
        )
        print(res_struct_custom)
    except (JsonParseError, PydanticValidateError) as e:
        print("捕获结构化格式异常：", e)

    # 补充4：多轮流式（上下文+流式组合，高频业务场景）
    print("\n====补充4：多轮上下文流式====")
    chat.clear_history()
    chat.chat("我喜欢蓝色", llm)
    stream_res = chat.chat_stream("推荐蓝色生活用品", llm)
    stream_full = ""
    for seg in stream_res:
        stream_full += seg
    print("多轮流式完整结果：", stream_full)
