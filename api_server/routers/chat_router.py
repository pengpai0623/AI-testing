from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from api_server.common.response import ApiResponse
from api_server.models.chat_models import (
    SessionChatRequest,
    SessionChatResponse,
    SingleChatRequest,
    SingleChatResponse,
)
from api_server.service.chat_service import ChatService
from llmsdk.utils import logger
from llmsdk.utils.constants import CODE_OK

router = APIRouter()
chat_service = ChatService()


# ========== 接口1：普通单轮问答 ==========
@router.post("/single", summary="单轮问答")
def chat_single(req: SingleChatRequest):
    """
    单轮问答接口，每次请求独立，不保留上下文
    - prompt: 用户提问（必填）
    - system_prompt: 系统提示词（可选）
    - temperature: 模型温度参数（可选）
    """
    logger.info(f"[chat_router/single] 收到请求, temperature={req.temperature}, prompt_len={len(req.prompt)}")
    result = chat_service.chat_single(
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
    )
    resp = SingleChatResponse(
        answer=result["content"],
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
    )
    logger.info("[chat_router/single] 接口处理完毕，准备返回")
    return ApiResponse(code=CODE_OK, msg="ok", data=resp)


# ========== 接口2：异步多轮对话（requests兜底） ==========
@router.post("/session", summary="异步多轮对话‑基于requests实现【兜底】")
async def chat_session(req: SessionChatRequest):
    """
    多轮对话接口，session_id维护上下文
    - session_id: 会话ID
    - prompt: 用户本轮提问
    - system_prompt: 仅首次会话生效
    > 兜底接口，业务优先使用 /async_session（httpx异步版本）
    """
    session_id = req.session_id
    logger.info(f"[chat_router/session] 收到请求 session_id={session_id}, prompt_len={len(req.prompt)}")
    result, msg_count = await chat_service.chat_session(
        session_id=session_id, prompt=req.prompt, system_prompt=req.system_prompt, temperature=req.temperature
    )
    resp = SessionChatResponse(
        session_id=session_id,
        answer=result["content"],
        history_count=msg_count,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
    )
    logger.info("[chat_router/session] 接口处理完毕，准备返回")
    return ApiResponse(code=CODE_OK, msg="ok", data=resp)


# ========== 接口3：异步多轮对话（httpx） ==========
@router.post("/async_session", summary="异步多轮对话‑基于httpx实现【主】")
async def async_chat_session(req: SessionChatRequest):
    """
    异步多轮对话接口，session_id维护上下文
    - session_id: 会话ID
    - prompt: 用户本轮提问
    - system_prompt: 仅首次会话生效
    """
    session_id = req.session_id
    logger.info(f"[chat_router/async_session] 收到请求 session_id={session_id}, prompt_len={len(req.prompt)}")
    answer, msg_count = await chat_service.async_chat_session(
        session_id=session_id, prompt=req.prompt, system_prompt=req.system_prompt, temperature=req.temperature
    )
    resp = SessionChatResponse(
        session_id=session_id,
        answer=answer["content"],
        history_count=msg_count,
        prompt_tokens=answer["prompt_tokens"],
        completion_tokens=answer["completion_tokens"],
        total_tokens=answer["total_tokens"],
    )
    logger.info("[chat_router/async_session] 接口处理完毕，准备返回")
    return ApiResponse(code=CODE_OK, msg="ok", data=resp)


# ========== 接口4：SSE流式（底层同步 requests） ==========
@router.post("/session_stream_requests", summary="SSE流式‑同步requests底层")
async def chat_session_stream_with_requests(req: SessionChatRequest):
    """
    SSE流式输出，打字机效果；底层为同步requests，使用asyncio.to_thread避免阻塞事件循环
    事件：message分片 / done结束 / error异常
    """
    session_id = req.session_id
    logger.info(
        f"[chat_router/session_stream_requests] 收到请求, session_id={session_id}, prompt_len={len(req.prompt)}"
    )
    # router return之前执行公共预处理，异常走全局异常JSON返回
    trimmed_messages = await chat_service.build_chat_prepare_messages(
        session_id=session_id, prompt=req.prompt, system_prompt=req.system_prompt
    )
    generator = chat_service.chat_session_stream_requests(
        session_id=session_id,
        temperature=req.temperature,
        trimmed_messages=trimmed_messages,
    )
    return EventSourceResponse(generator)


# ========== 接口5：SSE流式（底层异步 httpx） ==========
@router.post("/session_stream_httpx", summary="SSE流式‑异步httpx底层【主】")
async def chat_session_stream_with_httpx(req: SessionChatRequest):
    """
    SSE流式输出，底层使用异步httpx
    事件：message分片 / done结束 / error异常
    """
    session_id = req.session_id
    logger.info(f"[chat_router/session_stream_httpx] 收到请求, session_id={session_id}, prompt_len={len(req.prompt)}")
    trimmed_messages = await chat_service.build_chat_prepare_messages(
        session_id=session_id, prompt=req.prompt, system_prompt=req.system_prompt
    )
    # 无需await，直接返回生成器给EventSourceResponse，内部迭代
    generator = chat_service.chat_session_stream_httpx(
        session_id=session_id,
        temperature=req.temperature,
        trimmed_messages=trimmed_messages,
    )
    return EventSourceResponse(generator)


# ========== 健康检查 ==========
@router.get("/health", summary="服务健康检查")
def health_check():
    logger.info("[chat_router/health] 健康检查请求")
    return ApiResponse(code=CODE_OK, msg="ok", data={"status": "ok", "service": "llm-api"})
