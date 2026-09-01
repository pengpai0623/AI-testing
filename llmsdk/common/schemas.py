from enum import Enum

from pydantic import BaseModel, Field


class MsgRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class MessageItem(BaseModel):

    role: MsgRole = Field(..., description="角色：user / assistant / system")
    content: str = Field(..., min_length=1, description="消息内容")
