"""
入参长度校验工具：prompt、system_prompt字符长度校验
业务错误码：ERR_PARAM_TOO_LONG = 4007
"""

from llmsdk.utils.constants import ERR_PARAM_TOO_LONG
from llmsdk.utils.exceptions import LLMBaseError, ParamTooLongError


def check_prompt_length(prompt: str, max_len: int) -> None:
    """
    校验prompt字符长度
    :param prompt: 用户提问
    :param max_len: 最大允许字符数
    :raises ParamTooLongError: 超过长度抛出业务异常
    """
    if not prompt:
        return
    if len(prompt) > max_len:
        raise ParamTooLongError(
            code=ERR_PARAM_TOO_LONG, msg=f"prompt字符长度超过限制，最大{max_len}，当前{len(prompt)}"
        )


def check_system_prompt_length(system_prompt: str | None, max_len: int) -> None:
    """
    校验system_prompt字符长度
    :param system_prompt: 系统提示词
    :param max_len: 最大允许字符数
    :raises ParamTooLongError
    """
    if system_prompt is None:
        return
    if len(system_prompt) > max_len:
        raise ParamTooLongError(
            code=ERR_PARAM_TOO_LONG, msg=f"system_prompt字符长度超过限制，最大{max_len}，当前{len(system_prompt)}"
        )


def check_chat_input(prompt: str, system_prompt: str | None, prompt_max_chars: int, system_max_chars: int) -> None:
    """
    一次性校验对话输入：prompt + system_prompt
    """
    check_prompt_length(prompt, prompt_max_chars)
    check_system_prompt_length(system_prompt, system_max_chars)
