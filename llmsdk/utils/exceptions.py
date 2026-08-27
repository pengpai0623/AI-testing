class LLMBaseError(Exception):
    """
    大模型业务领域的可预期业务错误，SDK所有异常父类
    """

    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(msg)


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


class LLMValueError(LLMBaseError):
    """业务数值非法、参数数值错误"""

    pass


class LLMConnectionError(LLMBaseError):
    """业务层面连接异常"""

    pass


class MessageValidateError(LLMBaseError):
    """内存组装的消息列表校验失败"""

    pass


class ClientDisconnectError(LLMBaseError):
    """客户端主动断连（浏览器关闭页面)"""

    pass


class RedisTimeoutError(LLMBaseError):
    """Redis请求超时"""

    pass


class RedisConnectionError(LLMBaseError):
    """Redis连接异常"""

    pass


class RedisError(LLMBaseError):
    """Redis通用异常"""

    pass
