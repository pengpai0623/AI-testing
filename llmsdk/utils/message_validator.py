from pydantic import ValidationError

from server.models.chat_models import MessageItem


def validate_messages(messages: list[dict]) -> list[dict]:
    """校验消息列表，过滤非法role，内部业务调用使用"""
    valid = []
    for msg in messages:
        item = MessageItem(**msg)
        valid.append(item.model_dump())
    return valid
