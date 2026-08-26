import asyncio
import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.utils import logger
from llmsdk.utils.constants import CODE_OK, ERR_LLM_HTTP, ERR_MSG_VALIDATE
from llmsdk.utils.exceptions import (
    ClientDisconnectError,
    LLMBaseError,
    LLMHttpError,
    MessageValidateError,
)
from server.models.chat_models import (
    MessageItem,
    SessionChatRequest,
    SessionChatResponse,
    SingleChatRequest,
    SingleChatResponse,
)
from server.schemas.common_resp import ApiResponse

router = APIRouter()

llm_client = LLMBaseClient()

# 内存会话存储，仅开发使用；uvicorn多worker下失效，生产务必替换Redis
session_store: dict[str, list] = {}


# ========== 接口1：普通单轮问答 ==========
@router.post("/single", summary="单轮问答")
def chat_single(req: SingleChatRequest):
    """
    单轮问答接口，每次请求独立，不保留上下文
    - prompt: 用户提问（必填）
    - system_prompt: 系统提示词（可选）
    - temperature: 模型温度参数（可选）
    """
    logger.info(f"[chat/single] 收到请求, temperature={req.temperature}, prompt_len={len(req.prompt)}")
    logger.info("[chat/single] 开始调用llm_client.chat_single")

    result = llm_client.chat_single(
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )
    logger.info(f"[chat/single] LLM返回完成，answer_len={len(result['content'])}")

    resp = SingleChatResponse(
        answer=result["content"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
    )
    logger.info("[chat/single] 接口处理完毕，准备返回")
    return ApiResponse(code=CODE_OK, msg="ok", data=resp)


# ========== 接口2：同步多轮对话（requests） ==========
@router.post("/session", summary="同步多轮对话")
def chat_session(req: SessionChatRequest):
    """
    多轮对话接口，session_id维护上下文
    - session_id: 会话ID
    - prompt: 用户本轮提问
    - system_prompt: 仅首次会话生效
    """
    session_id = req.session_id
    logger.info(f"[chat/session] 收到请求 session_id={session_id}, prompt_len={len(req.prompt)}")

    history = session_store.get(session_id, [])
    logger.info(f"[chat/session] session={session_id} 当前历史消息数={len(history)}")

    if not history and req.system_prompt:
        logger.info(f"[chat/session] session={session_id} 首次会话写入system_prompt")
        history.append({"role": "system", "content": req.system_prompt})
    elif history and req.system_prompt:
        # 会话已存在，忽略新传入system_prompt，本会话不支持动态更新system
        logger.warning(
            f"[chat/session_stream_httpx] session={session_id} 会话已存在，忽略传入的system_prompt，如需变更请使用新session_id"
        )

    temp_messages = history + [{"role": "user", "content": req.prompt}]
    logger.info(f"[chat/session] session={session_id} 待校验消息总数={len(temp_messages)}")

    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(f"[chat/session] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}")
    except Exception as e:
        logger.exception(f"[chat/session] session={session_id} 消息校验失败 err={repr(e)}")
        raise MessageValidateError(code=ERR_MSG_VALIDATE, msg=f"消息格式非法: {str(e)}") from e

    logger.info(f"[chat/session] session={session_id} 开始调用大模型 chat_with_messages")
    answer = llm_client.chat_with_messages(messages=valid_messages, temperature=req.temperature)
    logger.info(f"[chat/session] session={session_id} 大模型调用完成")

    content = answer.get("content")
    if not content:
        logger.error(f"[chat/session] session={session_id} 大模型返回content为空")
        raise LLMHttpError(code=ERR_LLM_HTTP, msg="大模型返回内容为空")

    # 调用成功才更新会话
    session_store[session_id] = temp_messages + [{"role": "assistant", "content": content}]
    logger.info(f"[chat/session] session={session_id} 会话已更新，总消息数={len(session_store[session_id])}")

    resp = SessionChatResponse(
        session_id=session_id,
        answer=content,
        history_count=len(session_store[session_id]),
        prompt_tokens=answer["prompt_tokens"],
        completion_tokens=answer["completion_tokens"],
        total_tokens=answer["total_tokens"],
    )
    return ApiResponse(code=CODE_OK, msg="ok", data=resp)


# ========== 接口3：异步多轮对话（httpx） ==========
@router.post("/async_session", summary="异步多轮对话")
async def async_chat_session(req: SessionChatRequest):
    """
    异步多轮对话接口，session_id维护上下文
    """
    session_id = req.session_id
    logger.info(f"[chat/async_session] 收到请求 session_id={session_id}, prompt_len={len(req.prompt)}")

    history = session_store.get(session_id, [])
    logger.info(f"[chat/async_session] session={session_id} 当前历史消息数={len(history)}")

    if not history and req.system_prompt:
        logger.info(f"[chat/async_session] session={session_id} 首次会话写入system_prompt")
        history.append({"role": "system", "content": req.system_prompt})
    elif history and req.system_prompt:
        # 会话已存在，忽略新传入system_prompt，本会话不支持动态更新system
        logger.warning(
            f"[chat/session_stream_httpx] session={session_id} 会话已存在，忽略传入的system_prompt，如需变更请使用新session_id"
        )

    temp_messages = history + [{"role": "user", "content": req.prompt}]
    logger.info(f"[chat/async_session] session={session_id} 待校验消息总数={len(temp_messages)}")

    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(f"[chat/async_session] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}")
    except Exception as e:
        logger.exception(f"[chat/async_session] session={session_id} 消息校验失败 err={repr(e)}")
        raise MessageValidateError(code=ERR_MSG_VALIDATE, msg=f"消息格式非法: {str(e)}") from e

    logger.info(f"[chat/async_session] session={session_id} 开始调用 async_chat_with_messages")
    answer = await llm_client.async_chat_with_messages(messages=valid_messages, temperature=req.temperature)
    logger.info(f"[chat/async_session] session={session_id} 大模型异步调用完成")

    content = answer.get("content")
    if not content:
        logger.error(f"[chat/async_session] session={session_id} 大模型返回content为空")
        raise LLMHttpError(code=ERR_LLM_HTTP, msg="大模型返回内容为空")

    session_store[session_id] = temp_messages + [{"role": "assistant", "content": content}]
    logger.info(f"[chat/async_session] session={session_id} 会话更新完成，总消息数={len(session_store[session_id])}")

    resp = SessionChatResponse(
        session_id=session_id,
        answer=content,
        history_count=len(session_store[session_id]),
        prompt_tokens=answer["prompt_tokens"],
        completion_tokens=answer["completion_tokens"],
        total_tokens=answer["total_tokens"],
    )
    return ApiResponse(code=CODE_OK, msg="ok", data=resp)


# ========== 接口4：SSE流式（底层同步 requests） ==========
@router.post("/session_stream_requests", summary="SSE流式-同步requests底层")
async def chat_session_stream_with_requests(req: SessionChatRequest):
    """
    SSE流式输出，打字机效果；底层为同步requests，使用asyncio.to_thread避免阻塞事件循环
    事件：message分片 / done结束 / error异常
    """
    session_id = req.session_id
    logger.info(f"[chat/session_stream_requests] 收到请求, session_id={session_id}, prompt_len={len(req.prompt)}")

    history = session_store.get(session_id, [])
    logger.info(f"[chat/session_stream_requests] session={session_id} 当前历史消息数={len(history)}")

    if not history and req.system_prompt:
        logger.info(f"[chat/session_stream_requests] session={session_id} 首次会话写入system_prompt")
        history.append({"role": "system", "content": req.system_prompt})
    elif history and req.system_prompt:
        # 会话已存在，忽略新传入system_prompt，本会话不支持动态更新system
        logger.warning(
            f"[chat/session_stream_httpx] session={session_id} 会话已存在，忽略传入的system_prompt，如需变更请使用新session_id"
        )

    temp_messages = history + [{"role": "user", "content": req.prompt}]
    logger.info(f"[chat/session_stream_requests] session={session_id} 待校验消息总数={len(temp_messages)}")

    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(
            f"[chat/session_stream_requests] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}"
        )
    except Exception as e:
        logger.exception(f"[chat/session_stream_requests] session={session_id} 消息校验失败 err={repr(e)}")
        raise MessageValidateError(code=ERR_MSG_VALIDATE, msg=f"消息格式非法: {str(e)}") from e

    async def stream_generator():
        full_answer = ""
        chunk_count = 0
        logger.info(f"[chat/session_stream_requests] session={session_id} 启动SSE生成器，开始拉取流式分片")

        def _safe_next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        try:
            sync_iter = await asyncio.to_thread(
                llm_client.chat_stream_messages,
                messages=valid_messages,
                temperature=req.temperature,
            )
            while True:
                chunk = await asyncio.to_thread(_safe_next, sync_iter)
                if chunk is None:
                    logger.info(
                        f"[chat/session_stream_requests] session={session_id} 流式迭代结束，分片总数={chunk_count}"
                    )
                    break
                if chunk:
                    chunk_count += 1
                    full_answer += chunk
                    yield {"event": "message", "data": chunk}

            # 完整正常结束，推送done并保存会话
            yield {"event": "done", "data": full_answer}
            logger.info(
                f"[chat/session_stream_requests] session={session_id} 推送done事件，完整回答长度={len(full_answer)}"
            )
            session_store[session_id] = temp_messages + [{"role": "assistant", "content": full_answer}]
            logger.info(
                f"[chat/session_stream_requests] session {session_id} 会话保存完成，历史条数 {len(session_store[session_id])}"
            )

        except ClientDisconnectError:
            # 客户端断联，连接已关闭，不推送error事件，丢弃本轮会话
            logger.info(f"[chat/session_stream_requests] session={session_id} 客户端断开连接，本轮会话丢弃")
            return

        except LLMBaseError as exc:
            # 业务异常：LLMNetworkError / LLMHttpError / LLMValueError等
            err_data = json.dumps({"code": exc.code, "msg": exc.msg})
            yield {"event": "error", "data": err_data}

        except Exception as exc:
            # 兜底未知异常
            logger.exception(f"[chat/session_stream_requests] session={session_id} SSE生成异常 err={repr(exc)}")
            err_data = json.dumps({"code": 500, "msg": "流式服务内部未知错误"})
            yield {"event": "error", "data": err_data}

    return EventSourceResponse(stream_generator())


# ========== 接口5：SSE流式（底层异步 httpx） ==========
@router.post("/session_stream_httpx", summary="SSE流式-异步httpx底层")
async def chat_session_stream_with_httpx(req: SessionChatRequest):
    """
    SSE流式输出，底层使用异步httpx
    """
    session_id = req.session_id
    logger.info(f"[chat/session_stream_httpx] 收到请求, session_id={session_id}, prompt_len={len(req.prompt)}")

    history = session_store.get(session_id, [])
    logger.info(f"[chat/session_stream_httpx] session={session_id} 当前历史消息数={len(history)}")

    if not history and req.system_prompt:
        logger.info(f"[chat/session_stream_httpx] session={session_id} 首次会话写入system_prompt")
        history.append({"role": "system", "content": req.system_prompt})
    elif history and req.system_prompt:
        # 会话已存在，忽略新传入system_prompt，本会话不支持动态更新system
        logger.warning(
            f"[chat/session_stream_httpx] session={session_id} 会话已存在，忽略传入的system_prompt，如需变更请使用新session_id"
        )

    temp_messages = history + [{"role": "user", "content": req.prompt}]
    logger.info(f"[chat/session_stream_httpx] session={session_id} 待校验消息总数={len(temp_messages)}")

    try:
        valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
        logger.info(
            f"[chat/session_stream_httpx] session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}"
        )
    except Exception as e:
        logger.exception(f"[chat/session_stream_httpx] session={session_id} 消息校验失败 err={repr(e)}")
        raise MessageValidateError(code=ERR_MSG_VALIDATE, msg=f"消息格式非法: {str(e)}") from e

    async def stream_generator():
        full_answer = ""
        chunk_count = 0
        logger.info(f"[chat/session_stream_httpx] session={session_id} 启动SSE生成器，开始拉取流式分片")
        try:
            # 异步生成器正确的调用方式是async for
            # async for 调用异步生成器的 __anext__() 方法
            # 真正的await是在_async_request_stream_messages的client.stream()以及aiter_lines()
            async for chunk in llm_client.async_chat_stream_messages(
                messages=valid_messages, temperature=req.temperature
            ):
                if chunk:
                    chunk_count += 1
                    full_answer += chunk
                    yield {"event": "message", "data": chunk}

            # 完整正常结束，推送done并保存会话
            yield {"event": "done", "data": full_answer}
            logger.info(
                f"[chat/session_stream_httpx] session={session_id} 推送done事件，完整回答长度={len(full_answer)},分片数量={chunk_count}"
            )
            session_store[session_id] = temp_messages + [{"role": "assistant", "content": full_answer}]
            logger.info(
                f"[chat/session_stream_httpx] session {session_id} 会话保存完成，历史条数 {len(session_store[session_id])}"
            )

        except ClientDisconnectError:
            # 客户端断联，连接已关闭，不推送error事件，丢弃本轮会话
            logger.info(f"[chat/session_stream_httpx] session={session_id} 客户端断开连接，本轮会话丢弃")
            return

        except LLMBaseError as exc:
            # 业务异常：LLMNetworkError / LLMHttpError / LLMValueError等
            err_data = json.dumps({"code": exc.code, "msg": exc.msg})
            yield {"event": "error", "data": err_data}

        except Exception as exc:
            logger.exception(f"[chat/session_stream_httpx] session={session_id} SSE生成异常 err={repr(exc)}")
            err_data = json.dumps({"code": 500, "msg": "流式服务内部未知错误"})
            yield {"event": "error", "data": err_data}

    return EventSourceResponse(stream_generator())


# ========== 健康检查 ==========
@router.get("/health", summary="服务健康检查")
def health_check():
    logger.info("[chat/health] 健康检查请求")
    return {"status": "ok", "service": "llm-api"}
