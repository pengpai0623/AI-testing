import time

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from llmsdk.utils import logger
from server.routers import chat_router


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
