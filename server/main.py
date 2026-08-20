from fastapi import FastAPI, HTTPException

from llmsdk.utils import logger
from server.routers import chat_router

app = FastAPI(
    title="LLM API 服务",
    description="基于 FastAPI + llmsdk 的大模型接口服务",
    version="1.0.0",
)

app.include_router(chat_router.router, prefix="/chat", tags=["同/异步多轮对话及流式返回"])

if __name__ == "__main__":
    import uvicorn

    """
    server.main:app：模块server.main里面的app对象
    host="0.0.0.0"：允许局域网其他机器访问；写127.0.0.1只能本机访问
    reload=True：开发模式，生产环境一定要关闭
    """
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
