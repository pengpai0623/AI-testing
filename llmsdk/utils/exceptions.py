class LLMBaseError(Exception):
    """SDK所有异常父类"""

    pass


class EnvConfigError(LLMBaseError):
    """环境变量缺失、配置错误"""

    pass


class LLMNetworkError(LLMBaseError):
    """网络超时、连接失败"""

    pass


class LLMSSEParseError(LLMBaseError):
    """SSE流式解析失败"""

    pass


class LLMHttpError(LLMBaseError):
    """接口4xx/5xx报错"""

    pass


# 结构化专用异常
class JsonParseError(LLMBaseError):
    """AI返回内容JSON解析失败"""

    pass


class PydanticValidateError(LLMBaseError):
    """Pydantic模型字段校验失败"""

    pass
