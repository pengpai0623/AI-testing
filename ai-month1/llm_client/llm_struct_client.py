import json
import re

from base_llm import LLMBaseClient
from pydantic import ValidationError
from struct_schema_model import ProductInfo
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

"""
Day4 整体目标
掌握提示词写法，约束模型只返回 JSON，无额外自然语言；
使用 Pydantic 定义数据模板，规定必填字段、字段类型、限制规则；
AI 返回的字符串做 json 解析，用 Pydantic 校验合法性；
格式错乱、缺字段、类型错误时自动触发重试，重新请求 AI；
完全复用现有 base_llm 底层能力，上层新增结构化工具类。
"""

"""
why？
普通闲聊不需要结构化，程序自动处理 AI 返回内容必须结构化
你日常手动聊天、自己阅读文字，AI 随便写大白话完全没问题；
但如果是Python 代码要自动读取、提取、计算、入库、做判断，自由文本完全不可用。
"""

"""
公司用 AI 做数据清洗、内容提取、自动化工单、表单解析，全部依赖结构化输出。
面试官考察你：能不能管控 AI 输出，而不是只会随便聊天。
"""

# 全局LLM实例
llmbaseclient = LLMBaseClient()

_SYSTEM_PROMPT = (
    "你是信息抽取助手。严格遵守规则："
    "1. 只返回JSON字符串，绝对不要增加任何解释、前言、后语、换行说明；"
    "2. 不要用markdown代码块包裹；"
    "3. 必须包含name、price、tags三个字段；"
    "4. price必须是纯数字，不能带元、¥等符号；"
    "5. tags是字符串列表。用户输入商品文案，仅输出合法JSON。"
)

# 重试全局配置
# 注意与网络 / 超时等重试区分
MAX_RETRY_TIMES = 2
RETRY_WAIT_SEC = 1


def retry_log_callback(retry_state: RetryCallState):
    exc = retry_state.outcome.exception()
    print(f"【格式重试】第{retry_state.attempt_number}次重试，等待{RETRY_WAIT_SEC}s，异常：{exc}")


# 自定义异常
class JsonParseError(Exception):
    def __init__(self, msg: str, raw_err: Exception = None):
        self.code = 400
        self.msg = msg
        self.raw_error = raw_err
        super().__init__(self.msg)


class PydanticValidateError(Exception):
    def __init__(self, msg: str, validate_err: ValidationError):
        self.code = 422
        self.msg = msg
        self.validate_detail = validate_err.errors()
        super().__init__(self.msg)


class LLMStructClient:
    """
    prompt 强制要求返回json格式
    请求 LLM 拿到原始(json)字符串；
    清洗去除 markdown、换行；
    json.loads 解析，失败抛JsonParseError触发重试；
    解析成功用 Pydantic 校验全字段；字段错抛PydanticValidateError触发重试；
    重试耗尽直接报错；全部合法返回 Pydantic 对象，直接 .name/.price 取值。
    """

    def __init__(self):
        self.system_prompt = _SYSTEM_PROMPT

    def _format_json(self, raw_text: str) -> str:
        """清洗AI返回内容，得到纯净单行JSON字符串"""
        content = raw_text.strip()
        # 剔除markdown ```json ```
        pattern = r"```(?:json)?\n?([\s\S]*?)\n?```"
        match_result = re.search(pattern, content)
        if match_result:
            content = match_result.group(1).strip()
        # 清除换行制表符
        content = content.replace("\n", "").replace("\t", "")
        return content.strip()

    @retry(
        stop=stop_after_attempt(MAX_RETRY_TIMES),
        wait=wait_fixed(RETRY_WAIT_SEC),
        retry=retry_if_exception_type((JsonParseError, PydanticValidateError)),
        before_sleep=retry_log_callback,
    )
    def struct_chat(self, prompt: str, schema_cls):
        """
        通用结构化调用，支持任意Pydantic模型
        :param prompt: 用户输入文本
        :param schema_cls: 任意继承BaseModel的Pydantic类（如ProductInfo）
        :return: Pydantic实例
        """
        # 1 请求大模型
        llm_resp = llmbaseclient.chat_single(prompt, system_prompt=self.system_prompt)
        # json 在python中实际就是str
        if llm_resp["status"] != "success":
            raise Exception(f"大模型调用失败：{llm_resp['content']}")

        origin_text = llm_resp["content"]
        clean_json_str = self._format_json(origin_text)

        # 2 JSON解析，此处捕获报错抛自定义异常
        try:
            json_data = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            raise JsonParseError(f"JSON解析失败，清洗内容：{clean_json_str}", e) from e

        # 3 Pydantic校验
        try:
            # 通用化支持任意 Pydantic 模型, 不再写死 ProductInfo，调用时传入即可
            model_obj = schema_cls(**json_data)
        except ValidationError as e:
            raise PydanticValidateError("字段校验不通过", e) from e

        return model_obj


if __name__ == "__main__":
    llmstructClient = LLMStructClient()
    res = llmstructClient.struct_chat(
        prompt="新款无线蓝牙耳机，售价129，适合运动、通勤，音质清晰", schema_cls=ProductInfo  #
    )
    print(res)
    print(f"商品名：{res.name}，价格：{res.price}，标签：{res.tags}")
