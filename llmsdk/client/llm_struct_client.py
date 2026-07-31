import json
from typing import Optional, Type

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.utils.common import struct_retry_log
from llmsdk.utils.constants import (
    DEFAULT_TIMEOUT,
    MAX_RETRY_TIMES,
    RETRY_WAIT_SEC,
    STRUCT_DEFAULT_SYSTEM_PROMPT,
    STRUCT_DEFAULT_TEMP,
)
from llmsdk.utils.exceptions import (
    JsonParseError,
    LLMBaseError,
    LLMHttpError,
    LLMNetworkError,
    PydanticValidateError,
)
from llmsdk.utils.text_utils import clean_ai_json

"""
Day4 结构化能力封装
约束模型只返回纯JSON，无多余文字；
Pydantic做字段强校验，格式/类型/字段错误自动重试；
适配llmsdk统一配置、统一异常、统一重试参数；
完全复用LLMBaseClient底层网络/鉴权配置。

适用场景：需要程序自动读取AI返回数据（表单提取、商品信息、结构化入库、参数抽取）
"""


class LLMStructClient:
    def __init__(self, system_prompt: Optional[str] = None):
        self.llm_client = LLMBaseClient()
        # 实例自定义优先，否则用全局常量默认提示词
        self.default_system = system_prompt if system_prompt else STRUCT_DEFAULT_SYSTEM_PROMPT

    @retry(
        stop=stop_after_attempt(MAX_RETRY_TIMES),
        wait=wait_fixed(RETRY_WAIT_SEC),
        retry=retry_if_exception_type((JsonParseError, PydanticValidateError)),
        before_sleep=struct_retry_log,
    )
    def struct_chat(
        self,
        prompt: str,
        schema_cls: Type[BaseModel],
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> BaseModel:
        """
        通用结构化对话
        :param prompt: 用户输入自然语言
        :param schema_cls: Pydantic BaseModel子类，定义返回字段规则
        :param timeout: 单次请求超时，不传使用全局默认
        :param temperature: 结构化建议调低随机性，默认0.2
        :return: 实例化后的Pydantic对象
        """
        # 优先级：单次传参 > 实例初始化 > 全局常量
        final_system = system_prompt if system_prompt is not None else self.default_system
        final_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        final_temp = temperature if temperature is not None else STRUCT_DEFAULT_TEMP

        llm_resp = self.llm_client.chat_single(
            prompt=prompt,
            timeout=final_timeout,
            temperature=final_temp,
            system_prompt=final_system,
        )

        # 区分异常类型抛出
        if llm_resp["status"] != "success":
            err_msg = llm_resp["content"]
            match llm_resp["status"]:
                case "timeout" | "conn_error":
                    raise LLMNetworkError(f"结构化请求网络异常：{err_msg}")
                case "limit_429" | "http_err":
                    raise LLMHttpError(f"结构化接口异常：{err_msg}")
                case _:
                    raise LLMBaseError(f"模型调用失败：{err_msg}")

        raw_content = llm_resp["content"]
        clean_json_str = clean_ai_json(raw_content)

        # JSON解析
        try:
            json_data = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            raise JsonParseError(f"JSON解析失败，清洗后文本：{clean_json_str}") from e

        # Pydantic优化报错文案
        try:
            return schema_cls(**json_data)
        except ValidationError as e:
            err_detail = e.errors()[0]
            field = err_detail.get("loc", "未知字段")
            msg = err_detail.get("msg", "校验失败")
            raise PydanticValidateError(f"字段校验失败：字段{field}，原因：{msg}") from e
