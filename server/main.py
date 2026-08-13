from fastapi import FastAPI, HTTPException

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

app = FastAPI(
    title="LLM API 服务",
    description="基于 FastAPI + llmsdk 的大模型接口服务",
    version="1.0.0",
)

# 初始化LLM客户端
llm_client = LLMBaseClient()

# 多轮对话会话存储（内存版，重启丢失，多进程部署会话不共享；生产用Redis）
# 结构：{session_id: [{"role":"user","content":"xxx"}, ...]}
session_store: dict[str, list] = {}


# ========== 接口1：普通单轮问答 ==========
@app.post("/chat/single", response_model=SingleChatResponse, summary="单轮问答")
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


# ========== 接口2：多轮对话（带会话管理） ==========
@app.post("/chat/session", response_model=SessionChatResponse, summary="多轮对话")
def chat_session(req: SessionChatRequest):
    """
    多轮对话接口，通过 session_id 维护上下文
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


# ========== 健康检查 ==========
@app.get("/health", summary="健康检查")
def health_check():
    logger.info("[health] 健康检查请求")
    return {"status": "ok", "service": "llm-api"}


if __name__ == "__main__":
    import uvicorn

    """
    server.main:app：模块server.main里面的app对象
    host="0.0.0.0"：允许局域网其他机器访问；写127.0.0.1只能本机访问
    reload=True：开发模式，生产环境一定要关闭
    """
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
