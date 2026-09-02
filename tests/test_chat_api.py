"""
全接口E2E测试 - 主路径验证

前置条件：
  1. uvicorn main:app 已启动（默认 http://127.0.0.1:8000）
  2. Redis 已启动且可连接
  3. LLM API Key 有效（测试会真实调用大模型，产生费用）

运行方式：
  pytest tests/test_chat_api.py -v
  API_BASE_URL=http://your-host:8000 pytest tests/test_chat_api.py -v
"""

import json

import pytest
from httpx import AsyncClient


def _parse_sse_line(line: str, event: str, messages: list, done_data: dict, error_data: dict):
    """
    解析单行SSE响应，按引用更新各事件容器。
    注意：data: 后用切片而非split，避免内容本身含冒号被截断。
    """
    line = line.strip()
    if not line:
        return event
    if line.startswith("event:"):
        return line[len("event:") :].strip()
    if line.startswith("data:"):
        data = line[len("data:") :].strip()
        if event == "message":
            messages.append(data)
        elif event == "done":
            done_data.update(json.loads(data))
        elif event == "error":
            error_data.update(json.loads(data))
    return event


# ============================================================
# 接口1：单轮问答 /chat/single
# ============================================================


async def test_single_chat_success(client: AsyncClient):
    """单轮问答正常返回：answer非空，token字段完整且total>0"""
    resp = await client.post("/chat/single", json={"prompt": "你好，请用一句话自我介绍"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["msg"] == "ok"
    data = body["data"]
    assert data["answer"], "answer不应为空"
    assert isinstance(data["prompt_tokens"], int)
    assert isinstance(data["completion_tokens"], int)
    assert isinstance(data["total_tokens"], int)
    assert data["total_tokens"] > 0


async def test_single_missing_prompt_422(client: AsyncClient):
    """缺少必填prompt，FastAPI返回422参数校验错误"""
    resp = await client.post("/chat/single", json={})
    body = resp.json()
    assert body["code"] == 422


# ============================================================
# 接口3：多轮非流式（httpx主） /chat/async_session
# ============================================================


async def test_async_session_first_round(client: AsyncClient, session_id: str):
    """多轮首次会话：正常返回，history_count=2（user+assistant）"""
    resp = await client.post(
        "/chat/async_session",
        json={"session_id": session_id, "prompt": "我叫张三，记住我的名字"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    data = body["data"]
    assert data["session_id"] == session_id
    assert data["answer"], "answer不应为空"
    assert data["history_count"] == 2, f"首次会话应为2条，实际{data['history_count']}"


async def test_async_session_context_memory(client: AsyncClient, session_id: str):
    """多轮上下文记忆：第一轮告知名字，第二轮能正确回答，history_count=4"""
    # 第一轮
    resp1 = await client.post(
        "/chat/async_session",
        json={"session_id": session_id, "prompt": "我叫张三，记住我的名字"},
    )
    assert resp1.status_code == 200
    assert resp1.json()["code"] == 0

    # 第二轮：同session_id提问
    resp2 = await client.post(
        "/chat/async_session",
        json={"session_id": session_id, "prompt": "我叫什么名字？"},
    )
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["code"] == 0
    answer = body2["data"]["answer"]
    assert "张三" in answer, f"上下文记忆失败，回答中未包含'张三'，实际回答：{answer}"
    assert body2["data"]["history_count"] == 4, f"两轮会话应为4条，实际{body2['data']['history_count']}"


# ============================================================
# 接口5：SSE流式（httpx主） /chat/session_stream_httpx
# ============================================================


async def test_stream_httpx_normal_flow(client: AsyncClient, session_id: str):
    """
    SSE流式正常输出：
    - 响应头 content-type 含 text/event-stream
    - 收到多个 message 分片
    - 最终收到 done 事件，无 error 事件
    - done.full_answer 等于所有 message 分片拼接
    - done.msg_count = 2（首次会话）
    """
    async with client.stream(
        "POST",
        "/chat/session_stream_httpx",
        json={"session_id": session_id, "prompt": "用三句话介绍Python"},
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        messages = []
        done_data = {}
        error_data = {}
        event = ""

        async for line in resp.aiter_lines():
            event = _parse_sse_line(line, event, messages, done_data, error_data)

        # 无error事件
        assert not error_data, f"流式出现error事件: {error_data}"
        # 至少1个message分片
        assert len(messages) > 0, "未收到任何message分片"
        # 必须收到done事件
        assert done_data, "未收到done事件"
        # done.full_answer == 分片拼接
        full_from_chunks = "".join(messages)
        chunk_normalized = full_from_chunks.replace("\n", "").replace(" ", "")
        done_normalized = done_data["full_answer"].replace("\n", "").replace(" ", "")
        assert chunk_normalized == done_normalized, (
            f"done.full_answer与分片拼接不一致\n"
            f"分片拼接原始长度={len(full_from_chunks)}, done原始长度={len(done_data['full_answer'])}\n"
            f"chunk_norm={repr(chunk_normalized[:100])}\n"
            f"done_norm={repr(done_normalized[:100])}"
        )
        assert done_data["msg_count"] == 2, f"首次流式会话msg_count应为2，实际{done_data['msg_count']}"


# ============================================================
# 跨接口：非流式建立会话 → 流式继续（上下文互通）
# ============================================================


async def test_cross_session_nonstream_to_stream(client: AsyncClient, session_id: str):
    """
    非流式接口建立会话后，流式接口同session_id能读到历史上下文：
    - 非流式告知"我叫李四"
    - 流式提问"我叫什么"，回答包含"李四"
    - done.msg_count = 4（非流式2条 + 流式本轮2条）
    """
    # 第一步：非流式建立会话
    resp1 = await client.post(
        "/chat/async_session",
        json={"session_id": session_id, "prompt": "我叫李四，记住我的名字"},
    )
    assert resp1.status_code == 200
    assert resp1.json()["code"] == 0

    # 第二步：流式接口同session_id提问
    async with client.stream(
        "POST",
        "/chat/session_stream_httpx",
        json={"session_id": session_id, "prompt": "我叫什么名字？"},
    ) as resp:
        assert resp.status_code == 200
        messages = []
        done_data = {}
        error_data = {}
        event = ""

        async for line in resp.aiter_lines():
            event = _parse_sse_line(line, event, messages, done_data, error_data)

        assert not error_data, f"流式出现error事件: {error_data}"
        assert done_data, "未收到done事件"
        full_answer = "".join(messages)
        assert "李四" in full_answer, f"跨接口上下文互通失败，流式回答未包含'李四'，实际：{full_answer}"
        assert done_data["msg_count"] == 4, f"跨接口会话msg_count应为4，实际{done_data['msg_count']}"
