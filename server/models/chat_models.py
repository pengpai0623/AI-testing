from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ========== 单轮问答 ==========
class SingleChatRequest(BaseModel):
    """单轮问答请求体"""

    prompt: str = Field(..., min_length=1, max_length=4000, description="用户提问内容")
    system_prompt: Optional[str] = Field(None, max_length=2000, description="系统提示词，可选")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="随机参数，0-2之间")
    stream: Optional[bool] = Field(False, description="是否流式返回")


class SingleChatResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ========== 多轮对话 ==========
class MsgRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class MessageItem(BaseModel):

    role: MsgRole = Field(..., description="角色：user / assistant / system")
    content: str = Field(..., min_length=1, description="消息内容")


class SessionChatRequest(BaseModel):
    """多轮对话请求体"""

    session_id: str = Field(..., min_length=1, max_length=64, description="会话ID，用于区分不同对话")
    prompt: str = Field(..., min_length=1, max_length=4000, description="本轮用户提问")
    system_prompt: Optional[str] = Field(None, max_length=2000, description="系统提示词，首次会话传入")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)

    class Config:
        extra = "forbid"  # V2版本，禁止多余字段


class SessionChatResponse(BaseModel):
    session_id: str
    answer: str
    history_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
