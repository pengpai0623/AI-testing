import asyncio

from fastapi import APIRouter, FastAPI, HTTPException
from sse_starlette.sse import EventSourceResponse

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.utils import logger
from llmsdk.utils.message_validator import validate_messages
from server.models.chat_models import (
    MessageItem,
    SessionChatRequest,
    SessionChatResponse,
    SingleChatRequest,
    SingleChatResponse,
)

# 1. 实例化一个路由器
router = APIRouter()

# 初始化LLM客户端
llm_client = LLMBaseClient()

# 多轮对话会话存储（内存版，重启丢失，多进程部署会话不共享；生产用Redis）
# 结构：{session_id: [{"role":"user","content":"xxx"}, ...]}
session_store: dict[str, list] = {}


# ========== 接口1：普通单轮问答 ==========
@router.post("/single", response_model=SingleChatResponse, summary="单轮问答")
def chat_single(req: SingleChatRequest):
    """
    单轮问答接口，每次请求独立，不保留上下文
    - **prompt**: 用户提问（必填）
    - **system_prompt**: 系统提示词（可选）
    - **temperature**: 温度参数（可选，默认0.7）
    """
    logger.info(f"[chat/single] 收到请求，temperature:{req.temperature}, prompt长度:{len(req.prompt)}")
    try:
        # 调用llmsdk客户端
        result = llm_client.chat_single(
            prompt=req.prompt,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
        )

        return SingleChatResponse(
            code=0,
            message="success",
            data={
                "answer": result,
                "prompt_tokens": len(req.prompt),  # 简易token估算
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")


# ========== 接口2：同步多轮对话（带会话管理） ==========
@router.post("/session", response_model=SessionChatResponse, summary="同步多轮对话")
def chat_session(req: SessionChatRequest):
    """
    多轮对话接口，通过 session_id 维护上下文，基于requests
    - **session_id**: 会话ID（必填，同一对话保持一致）
    - **prompt**: 本轮用户提问
    - **system_prompt**: 系统提示词（仅首次会话生效）
    """
    session_id = req.session_id
    logger.info(f"[chat/session] 收到请求 session_id={session_id}, prompt长度:{len(req.prompt)}")

    # 获取历史（若不存在则初始化）
    history = session_store.get(session_id, [])

    if not history and req.system_prompt:
        history.append({"role": "system", "content": req.system_prompt})

    # 构造本次完整消息列表（不含助手回复）
    temp_messages = history + [{"role": "user", "content": req.prompt}]

    # 验证消息合法性
    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(f"[chat/session] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"消息格式非法: {e}")

    try:
        logger.info(f"[chat/session] session={session_id} 开始调用大模型")
        answer = llm_client.chat_with_messages(messages=valid_messages, temperature=req.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")

    content = answer.get("content")
    if not content:
        raise HTTPException(status_code=500, detail="大模型返回内容为空")

    # 成功：把user+assistant完整存入会话存储
    session_store[session_id] = temp_messages + [{"role": "assistant", "content": content}]

    return SessionChatResponse(
        code=0,
        message="success",
        data={
            "session_id": session_id,
            "answer": answer,
            "history_count": len(session_store[session_id]),
        },
    )


# ========== 接口3：异步多轮对话（带会话管理） ==========
@router.post("/async_session", response_model=SessionChatResponse, summary="异步多轮对话")
async def async_chat_session(req: SessionChatRequest):
    """
    异步多轮对话接口，通过 session_id 维护上下文，基于httpx
    - **session_id**: 会话ID（必填，同一对话保持一致）
    - **prompt**: 本轮用户提问
    - **system_prompt**: 系统提示词（仅首次会话生效）
    """
    session_id = req.session_id
    logger.info(f"[chat/session] 收到请求 session_id={session_id}, prompt长度:{len(req.prompt)}")

    # 获取历史（若不存在则初始化）
    history = session_store.get(session_id, [])

    if not history and req.system_prompt:
        history.append({"role": "system", "content": req.system_prompt})

    # 构造本次完整消息列表（不含助手回复）
    temp_messages = history + [{"role": "user", "content": req.prompt}]

    # 验证消息合法性
    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(f"[chat/session] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"消息格式非法: {e}")

    try:
        logger.info(f"[chat/session] session={session_id} 开始调用大模型")
        answer = await llm_client.async_chat_with_messages(messages=valid_messages, temperature=req.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM调用失败: {str(e)}")

    content = answer.get("content")
    if not content:
        raise HTTPException(status_code=500, detail="大模型返回内容为空")

    # 成功：把user+assistant完整存入会话存储
    session_store[session_id] = temp_messages + [{"role": "assistant", "content": content}]

    return SessionChatResponse(
        code=0,
        message="success",
        data={
            "session_id": session_id,
            "answer": answer,
            "history_count": len(session_store[session_id]),
        },
    )


# ========== 接口4：流式返回--基于requests ==========
@router.post("/session_stream_requests", summary="多对话SSE流式输出(打字机)")
async def chat_session_stream_with_requests(req: SessionChatRequest):
    """
    SSE流式多轮对话
    基于requests
    客户端接收event: message 的data分片；全部结束推送event: done
    """

    session_id = req.session_id
    logger.info(f"[chat/session_stream] 收到请求, session_id={session_id}, prompt长度:{len(req.prompt)}")

    # 获取历史（若不存在则初始化）
    history = session_store.get(session_id, [])

    if not history and req.system_prompt:
        history.append({"role": "system", "content": req.system_prompt})

    # 构造本次完整消息列表（不含助手回复）
    temp_messages = history + [{"role": "user", "content": req.prompt}]

    # 验证消息合法性
    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(f"[chat/stream] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"消息格式非法: {e}")

    async def stream_generator():
        full_answer = ""
        try:
            sync_iter = await asyncio.to_thread(
                llm_client.chat_stream_messages,
                messages=valid_messages,
                temperature=req.temperature,
            )
            while True:
                try:
                    chunk = await asyncio.to_thread(next, sync_iter)
                except StopIteration:
                    break
                if chunk:
                    full_answer += chunk
                    yield {"event": "message", "data": chunk}

            yield {"event": "done", "data": full_answer}

        except Exception as exc:
            logger.error(f"[chat/session_stream] 调用大模型异常 {exc}", exc_info=True)
            yield {"event": "error", "data": str(exc)}
        finally:
            session_store[session_id] = temp_messages + [{"role": "assistant", "content": full_answer}]
            logger.info(
                f"[chat/session_stream] session {session_id} 会话保存完成，总长度 {len(session_store[session_id])}"
            )

    return EventSourceResponse(stream_generator())


# ========== 接口5：流式返回--基于httpx ==========
@router.post("/session_stream_httpx", summary="多对话SSE流式输出(打字机)")
async def chat_session_stream_with_httpx(req: SessionChatRequest):
    """
    SSE流式多轮对话
    基于httpx
    客户端接收event: message 的data分片；全部结束推送event: done
    """

    session_id = req.session_id
    logger.info(f"[chat/session_stream] 收到请求, session_id={session_id}, prompt长度:{len(req.prompt)}")

    # 获取历史（若不存在则初始化）
    history = session_store.get(session_id, [])

    if not history and req.system_prompt:
        history.append({"role": "system", "content": req.system_prompt})

    # 构造本次完整消息列表（不含助手回复）
    temp_messages = history + [{"role": "user", "content": req.prompt}]

    # 验证消息合法性
    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(f"[chat/stream] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"消息格式非法: {e}")

    async def stream_generator():
        full_answer = ""
        try:
            # 异步生成器正确的调用方式是async for
            # async for 调用异步生成器的 __anext__() 方法
            # 真正的await是在_async_request_stream_messages的client.stream()以及aiter_lines()
            async for chunk in llm_client.async_chat_stream_messages(
                messages=valid_messages, temperature=req.temperature
            ):
                if chunk:
                    full_answer += chunk
                    yield {"event": "message", "data": chunk}

            yield {"event": "done", "data": full_answer}
        except Exception as exc:
            logger.error(f"调用大模型异常 {exc}", exc_info=True)
            yield {"event": "error", "data": str(exc)}
        finally:
            session_store[session_id] = temp_messages + [{"role": "assistant", "content": full_answer}]

    return EventSourceResponse(stream_generator())


# ========== 健康检查 ==========
@router.get("/health", summary="健康检查")
def health_check():
    logger.info("[health] 健康检查请求")
    return {"status": "ok", "service": "llm-api"}
