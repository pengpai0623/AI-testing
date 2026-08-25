from typing import Any, Optional

from pydantic import BaseModel


class ApiResponse(BaseModel):
    """全局统一返回结构体"""

    code: int
    msg: str
    data: Optional[Any] = None
