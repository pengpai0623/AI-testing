from tenacity import RetryCallState

from llmsdk.utils.constants import RETRY_WAIT_SEC


def format_retry_log(retry_state: RetryCallState, scene: str = "通用"):
    """全局统一重试日志函数，支持区分场景：流式/普通聊天/结构化"""
    exc = retry_state.outcome.exception()
    print(f"【{scene}重试】第{retry_state.attempt_number}次重试，等待{RETRY_WAIT_SEC}s，异常：{exc}")


# 结构化专用快捷封装
def struct_retry_log(retry_state: RetryCallState):
    format_retry_log(retry_state, scene="格式校验")
