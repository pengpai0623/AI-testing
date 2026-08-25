import json
from typing import Optional, Type

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.utils.common import struct_retry_log
from llmsdk.utils.constants import (
    DEFAULT_TIMEOUT,
    ERR_JSON_PARSE,
    ERR_PYDANTIC_VALIDATE,
    MAX_RETRY_TIMES,
    RETRY_WAIT_SEC,
    STRUCT_DEFAULT_SYSTEM_PROMPT,
    STRUCT_DEFAULT_TEMP,
)
from llmsdk.utils.exceptions import JsonParseError, LLMBaseError, PydanticValidateError
from llmsdk.utils.text_utils import clean_ai_json

"""
Day4 结构化能力封装
约束模型只返回纯JSON，无多余文字；
Pydantic做字段强校验，格式/类型/字段错误自动重试；
适配llmsdk统一配置、统一异常、统一重试参数；
完全复用LLMBaseClient底层网络/鉴权配置。

注意：
- 仅【JSON解析失败、Pydantic字段校验失败】会触发重试；
- 网络异常、鉴权错误、请求参数错误(LLMBaseError子类)直接向上抛出，**不会重试**，交给上层捕获处理。

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
        :raises JsonParseError: AI输出无法解析为JSON，触发重试
        :raises PydanticValidateError: 字段缺失/类型不匹配，触发重试
        :raises LLMBaseError: 网络、鉴权、参数错误，不重试，直接向外抛出
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

        raw_content = llm_resp.get("content", "")
        if not raw_content:
            raise JsonParseError(code=ERR_JSON_PARSE, msg="AI返回内容为空，无法解析JSON")

        clean_json_str = clean_ai_json(raw_content)

        # JSON解析
        try:
            json_data = json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            raise JsonParseError(code=ERR_JSON_PARSE, msg=f"JSON解析失败：{str(e)}") from e

        # Pydantic优化报错文案
        try:
            return schema_cls(**json_data)
        except ValidationError as e:
            err_detail = e.errors()[0]
            field = err_detail.get("loc", "未知字段")
            msg = err_detail.get("msg", "校验失败")
            raise PydanticValidateError(
                code=ERR_PYDANTIC_VALIDATE,
                msg=f"字段校验失败：字段{field}，原因：{msg}",
            ) from e
