import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api_server.common.limiter import get_client_ip, rate_limiter_dep
from api_server.common.response import ApiResponse
from api_server.routers import chat_router
from llmsdk.utils import logger
from llmsdk.utils.constants import (
    CODE_OK,
    CODE_SERVER_ERROR,
    CODE_VALIDATE_ERROR,
    ERR_HTTP_BAD_HTTP_STATUS,
    ERR_RATE_LIMIT,
)
from llmsdk.utils.exceptions import LLMBaseError, LLMSSEParseError


class RequestLogMiddleware(BaseHTTPMiddleware):
    # 中间件处理，记录请求耗时、path、client_ip，把公共字段绑定到 loguru extra 上下文。
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        client_ip = client_ip = get_client_ip(request)
        user_id = request.headers.get("X-User-Id", "anonymous")

        logger.info(
            f"[REQUEST START] method={request.method} path={request.url.path} "
            f"client_ip={client_ip} user_id={user_id}"
        )
        try:
            # 非流式接口，会持续到大模型调用返回结束，所以接口耗时完整准确，异常也可被全局异常捕获
            # 流式，仅包括构造 EventSourceResponse 对象耗时，不包含大模型调用、chunk产出时间，所以接口完整耗时有很大误差；
            # 统计耗时、异常处理需在stream_generator内部实现
            response = await call_next(request)
        except Exception as e:
            cost_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception(
                f"[REQUEST ERROR] path={request.url.path} user_id={user_id} cost={cost_ms}ms err={repr(e)}"
            )
            raise

        cost_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            f"[REQUEST END] path={request.url.path} status_code={response.status_code} "
            f"user_id={user_id} cost={cost_ms}ms"
        )
        return response


app = FastAPI(
    title="LLM API 服务",
    description="基于 FastAPI + llmsdk 的大模型接口服务",
    version="1.0.0",
)


# # 避免 to_thread排队，设置更大的线程池
# @app.on_event("startup")
# async def startup():
#     loop = asyncio.get_running_loop()
#     loop.set_default_executor(ThreadPoolExecutor(max_workers=64))


# 1.捕获Pydantic请求校验异常（422）
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("请求参数校验失败", exc_info=exc)
    return JSONResponse(
        content=ApiResponse(code=CODE_VALIDATE_ERROR, msg=f"参数校验错误:{exc.errors()}", data=None).model_dump(),
        status_code=200,  # 业务通过code区分，http状态码统一200；也可以保留422看团队规范
    )


# 2.捕获FastAPI抛出的HTTPException（包含429限流、404、405等）
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """捕获FastAPI抛出的HTTPException，包含429限流、404、405等"""
    client_ip = get_client_ip(request)

    if exc.status_code == 429:
        # 限流：使用独立业务错误码 ERR_RATE_LIMIT
        logger.warning(f"[RATE_LIMIT] ip={client_ip}, msg={exc.detail}")
        resp = ApiResponse(code=ERR_RATE_LIMIT, msg=exc.detail, data=None)
        return JSONResponse(status_code=200, content=resp.model_dump())

    # 其余404/405等http异常，统一使用公共业务错误码，detail保留原始信息给到前端
    logger.warning(f"[HTTP_EXC] http_status={exc.status_code}, detail={exc.detail}, ip={client_ip}")
    resp = ApiResponse(
        code=ERR_HTTP_BAD_HTTP_STATUS, msg=f"http请求异常：{exc.detail}", data={"http_status": exc.status_code}
    )
    return JSONResponse(status_code=200, content=resp.model_dump())


# 3.捕获自定义业务异常 /LLMBaseError及其子类
@app.exception_handler(LLMBaseError)
async def biz_exception_handler(request: Request, exc: LLMBaseError):
    logger.error(f"业务异常 code={exc.code}, msg={exc.msg}", exc_info=exc)
    return JSONResponse(
        content=ApiResponse(code=exc.code, msg=exc.msg, data=None).model_dump(),
        status_code=200,
    )


# 4.兜底捕获全部未处理Exception如原生异常等（包括500未知错误）
@app.exception_handler(Exception)
async def global_unknown_exception_handler(request: Request, exc: Exception):
    logger.exception("服务器未知异常")
    return JSONResponse(
        content=ApiResponse(code=CODE_SERVER_ERROR, msg="服务器内部错误，请联系管理员", data=None).model_dump(),
        status_code=200,
    )


# 全局注册中间件
app.add_middleware(RequestLogMiddleware)

"""
HTTP Request → FastAPI → 匹配 /chat/*路由
    → rate_limiter_dep（内存限流，429直接返回JSON）
    → Pydantic解析请求模型
    → 进入router接口函数
        → 调用chat_service业务层
            → service内部完成prompt/system_prompt长度校验、会话处理、LLM调用
"""

app.include_router(
    chat_router.router, prefix="/chat", tags=["同/异步多轮对话及流式返回"], dependencies=[Depends(rate_limiter_dep)]
)


if __name__ == "__main__":
    import uvicorn

    """
    api_server.main:app：模块server.main里面的app对象
    host="0.0.0.0"：允许局域网其他机器访问；写127.0.0.1只能本机访问
    reload=True：开发模式，生产环境一定要关闭
    uvicorn api_server.main:app --host 0.0.0.0 --port 8000 --reload
    """
    uvicorn.run("api_server.main:app", host="0.0.0.0", port=8000, reload=True)
