import os
import uuid

import pytest
from httpx import AsyncClient


@pytest.fixture
def base_url():
    """接口基础地址，默认本地8000，可通过环境变量覆盖"""
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture
async def client(base_url: str):
    """异步HTTP客户端，测试结束自动关闭"""
    async with AsyncClient(base_url=base_url, timeout=120.0) as c:
        yield c


@pytest.fixture
def session_id():
    """每个用例独立session_id，避免测试间互相污染"""
    return f"e2e_{uuid.uuid4().hex[:12]}"
