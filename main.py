from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class MsgRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class MessageItem(BaseModel):
    """单条消息"""

    role: MsgRole = Field(..., description="角色：user / assistant / system")
    content: str = Field(..., min_length=1, description="消息内容")


session_store = [
    {"role": "system", "content": "专业记名字大师"},
    {"role": "user", "content": "我叫lxy"},
    {
        "role": "assistant",
        "content": "哈喽lxy呀😉 这个拼音首字母缩写好有专属感，是你大名的缩写对不对？好奇背后藏着什么 好听的名字~\n\n不管你是今天想分享刚碰到的开心小事、吐槽遇到的糟心情况，还是需要帮忙整理资料、想创意点子、唠各种奇奇怪怪的脑洞话题，我都随时陪你聊哦~",
    },
    {"role": "user", "content": "我叫什么来着"},
]
valid_messages = [MessageItem(**item).model_dump() for item in session_store]

for item in session_store:
    print(MessageItem(**item).model_dump())
