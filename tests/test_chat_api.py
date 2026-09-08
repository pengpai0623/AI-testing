"""
全接口E2E测试 - 主路径验证
前置条件：
  1. uvicorn main:app 已启动（默认 http://127.0.0.1:8000）
  2. Redis 已启动且可连接
  3. LLM API Key 有效（测试会真实调用大模型，产生费用）

运行方式：
  pytest tests/test_chat_api.py -v
  API_BASE_URL=http://your-host:8000 pytest tests/test_chat_api.py -v

注意事项：
1. 限流用例使用Redis滑动窗口，共享Redis状态；多次执行需要等待窗口过期或清空redis key
2. SSE接口：路由层预处理异常(Pydantic/参数校验)直接返回JSON，不走text/event-stream；
   生成器内部异常才会走SSE error事件通道
3. 两层校验区分：
   - Pydantic max_length：HTTP入参单字段超长，code=422，发生在router之前
   - ChatService上下文校验：历史+prompt拼接后总上下文超限，业务错误码，进入service后抛出
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
    """缺少必填prompt，FastAPI Pydantic参数校验返回422"""
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


# ============================================================
# 业务校验：prompt / system_prompt 长度超限校验（Pydantic入参层）
# ============================================================


async def test_single_prompt_too_long(client: AsyncClient):
    """单轮问答 prompt 超出pydantic max_length，请求参数校验失败 code=422"""
    long_prompt = "你好".ljust(5000, "a")
    resp = await client.post("/chat/single", json={"prompt": long_prompt})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 422


async def test_async_session_prompt_too_long(client: AsyncClient, session_id: str):
    """多轮非流式 prompt超出pydantic max_length，参数校验失败 code=422"""
    long_prompt = "测试".ljust(5000, "x")
    resp = await client.post("/chat/async_session", json={"session_id": session_id, "prompt": long_prompt})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 422


async def test_stream_httpx_prompt_too_long_not_sse(client: AsyncClient, session_id: str):
    """
    SSE接口prompt超出pydantic max_length：
    路由预处理阶段直接拦截，返回普通JSON响应，不是text/event-stream流式；
    异常发生在生成器对象迭代之前，不走SSE error事件
    """
    long_prompt = "测试".ljust(5000, "z")
    resp = await client.post("/chat/session_stream_httpx", json={"session_id": session_id, "prompt": long_prompt})
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "text/event-stream" not in content_type
    body = resp.json()
    assert body["code"] == 422


# ============================================================
# Redis分布式滑动窗口限流测试
# 注意：共享Redis key，多轮pytest执行会互相影响；可清空redis或者等待窗口过期
# ============================================================


async def test_chat_rate_limit_trigger(client: AsyncClient):
    """
    高频调用接口触发Redis分布式滑动窗口限流，全局异常处理器转换返回code=42900，http=200
    前置：constants中MAX_REQUEST_PER_WINDOW设置较小，方便e2e触发
    """
    url = "/chat/single"
    payload = {"prompt": "hi"}

    limited = False
    for _ in range(50):
        r = await client.post(url, json=payload)
        j = r.json()
        print(f"rate limit debug resp: {j}")
        if j["code"] == 42900:
            limited = True
            break
    assert limited, "多次请求后未触发限流，检查rate_limiter_dep配置、确认MAX_REQUEST_PER_WINDOW数值"

    # 限流命中后再次请求依然是限流错误
    resp_limited = await client.post(url, json=payload)
    j_limited = resp_limited.json()
    assert j_limited["code"] == 42900
    assert "请求过于频繁" in j_limited["msg"]


# ============================================================
# 全局异常兜底：404、405，校验统一返回JSON格式
# ============================================================


async def test_global_404_not_found(client: AsyncClient):
    """访问不存在路由，校验全局异常处理器统一返回业务JSON结构"""
    resp = await client.post("/chat/not_exist_route", json={"prompt": "hi"})
    body = resp.json()
    assert "code" in body
    assert "msg" in body


async def test_global_405_method_not_allowed(client: AsyncClient):
    """接口请求方法不匹配，校验全局异常处理器统一返回业务JSON结构"""
    resp = await client.get("/chat/single")
    body = resp.json()
    assert "code" in body
    assert "msg" in body
