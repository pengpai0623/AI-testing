import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from llmsdk.utils import logger
from llmsdk.utils.constants import CODE_OK, CODE_SERVER_ERROR, CODE_VALIDATE_ERROR
from llmsdk.utils.exceptions import LLMBaseError, LLMSSEParseError
from server.routers import chat_router
from server.schemas.common_resp import ApiResponse


class RequestLogMiddleware(BaseHTTPMiddleware):
    # 中间件处理，记录请求耗时、path、client_ip，把公共字段绑定到 loguru extra 上下文。
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else ""
        user_id = request.headers.get("X-User-Id", "anonymous")

        logger.info(
            f"[REQUEST START] method={request.method} path={request.url.path} "
            f"client_ip={client_ip} user_id={user_id}"
        )
        try:
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


# 1.捕获Pydantic请求校验异常（422）
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("请求参数校验失败", exc_info=exc)
    return JSONResponse(
        content=ApiResponse(code=CODE_VALIDATE_ERROR, msg=f"参数校验错误:{exc.errors()}", data=None).model_dump(),
        status_code=200,  # 业务通过code区分，http状态码统一200；也可以保留422看团队规范
    )


# 2.捕获自定义业务异常 /LLMBaseError
@app.exception_handler(LLMBaseError)
async def biz_exception_handler(request: Request, exc: LLMBaseError):
    logger.error(f"业务异常 code={exc.code}, msg={exc.msg}", exc_info=exc)
    return JSONResponse(
        content=ApiResponse(code=exc.code, msg=exc.msg, data=None).model_dump(),
        status_code=200,
    )


# 3.兜底捕获全部未处理Exception（500未知错误）
@app.exception_handler(Exception)
async def global_unknown_exception_handler(request: Request, exc: Exception):
    logger.exception("服务器未知异常")
    return JSONResponse(
        content=ApiResponse(code=CODE_SERVER_ERROR, msg="服务器内部错误，请联系管理员", data=None).model_dump(),
        status_code=200,
    )


# ✅全局注册中间件
app.add_middleware(RequestLogMiddleware)

app.include_router(chat_router.router, prefix="/chat", tags=["同/异步多轮对话及流式返回"])


if __name__ == "__main__":
    import uvicorn

    """
    server.main:app：模块server.main里面的app对象
    host="0.0.0.0"：允许局域网其他机器访问；写127.0.0.1只能本机访问
    reload=True：开发模式，生产环境一定要关闭
    """
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
