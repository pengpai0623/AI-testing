import asyncio
import time
from typing import Dict, List

from fastapi import HTTPException, Request

from llmsdk.utils import logger
from llmsdk.utils.constants import MAX_REQUEST_PER_WINDOW, WINDOW_SECONDS
from llmsdk.utils.session_redis import SessionRedis


def get_client_ip(request: Request) -> str:
    """获取真实客户端IP"""
    try:
        x_forwarded = request.headers.get("x-forwarded-for")
        if x_forwarded:
            parts = x_forwarded.split(",")
            if parts:
                return parts[0].strip()
    except Exception as e:
        logger.warning(f"[PARSE_X_FORWARDED_FOR_FAILED] error={str(e)}")
    return request.client.host if request.client else "unknown"


session_redis: SessionRedis = SessionRedis()


async def rate_limiter_dep(request: Request):
    """FastAPI依赖：Redis分布式滑动窗口限流，底层同步redis‑py，to_thread避免阻塞事件循环"""
    ip = get_client_ip(request)
    redis_key = f"rate_limit:{ip}"
    now_ts = time.time()
    window_start_ts = now_ts - WINDOW_SECONDS

    try:
        # 在子线程执行同步redis pipeline，不阻塞asyncio事件循环
        def _redis_limit_logic():
            pipe = session_redis.client.pipeline()
            # pipe.multi()  # 上锁，pipeline自动开启事务, 不会被其他请求插队, 解决竞态问题
            pipe.zremrangebyscore(redis_key, 0, window_start_ts)
            pipe.zcard(redis_key)
            pipe.zadd(redis_key, {now_ts: now_ts})
            pipe.expire(redis_key, WINDOW_SECONDS)
            return pipe.execute()

        _, current_count, _, _ = await asyncio.to_thread(_redis_limit_logic)

        if current_count >= MAX_REQUEST_PER_WINDOW:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")

    except HTTPException:
        raise
    except Exception as e:
        # 只有Redis/IO等底层异常才降级放行
        logger.error(f"[RATE_LIMIT_REDIS_ERROR] skip rate limit, err={str(e)}")

    return True
