import json
from typing import Any, Dict, List

import redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from llmsdk.utils.constants import (
    ERR_REDIS_CONNECTION,
    ERR_REDIS_ERROR,
    ERR_REDIS_TIMEOUT,
    REDIS_URL,
    SESSION_TTL_SEC,
)
from llmsdk.utils.exceptions import (
    JsonParseError,
    RedisConnectionError,
    RedisError,
    RedisTimeoutError,
)
from llmsdk.utils.logger import logger


class SessionRedis:
    """
    redis 存储多轮对话上下文，支持多实例部署。
    实例化SessionRedis()的时候，不会做连接测试。只有第一次调用get/set才会真正发生网络连接
    """

    def __init__(self, redis_url: str = REDIS_URL):
        self.client = redis.from_url(redis_url)
        self.prefix = "llm:session:"

    def get_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话上下文
        :param session_id: 会话ID
        :return: 消息列表，若不存在返回空列表 []
        """
        key = self.prefix + session_id
        try:
            data = self.client.get(key)
        except ConnectionError as e:
            raise RedisConnectionError(ERR_REDIS_CONNECTION, msg=f"Redis连接异常: {e}") from e
        except TimeoutError as e:
            raise RedisTimeoutError(ERR_REDIS_TIMEOUT, msg=f"Redis请求超时: {e}") from e
        except RedisError as e:
            raise RedisError(ERR_REDIS_ERROR, msg=f"Redis操作异常: {e}") from e

        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                logger.error(f"session {session_id} redis数据解析失败: {e}")
                # 如果数据损坏，业务显式报错，前端感知会话损坏
                raise JsonParseError(ERR_REDIS_ERROR, f"会话数据解析损坏: {session_id}") from e
        return []

    def set_session(self, session_id: str, messages: List[Dict[str, Any]]):
        """
        设置会话上下文，带过期时间
        :param session_id: 会话ID
        :param messages: 消息列表
        """
        key = self.prefix + session_id
        try:
            payload = json.dumps(messages, ensure_ascii=False)
            self.client.setex(key, SESSION_TTL_SEC, payload)
        except json.JSONDecodeError as e:
            raise JsonParseError(ERR_REDIS_ERROR, msg=f"Redis数据序列化失败: {e}") from e
        except ConnectionError as e:
            raise RedisConnectionError(ERR_REDIS_CONNECTION, msg=f"Redis连接异常: {e}") from e
        except TimeoutError as e:
            raise RedisTimeoutError(ERR_REDIS_TIMEOUT, msg=f"Redis请求超时: {e}") from e
        except RedisError as e:
            raise RedisError(ERR_REDIS_ERROR, msg=f"Redis操作异常: {e}") from e

    def delete_session(self, session_id: str):
        """
        删除会话上下文
        :param session_id: 会话ID
        """
        key = self.prefix + session_id
        try:
            self.client.delete(key)
        except ConnectionError as e:
            raise RedisConnectionError(ERR_REDIS_CONNECTION, msg=f"Redis连接异常: {e}") from e
        except TimeoutError as e:
            raise RedisTimeoutError(ERR_REDIS_TIMEOUT, msg=f"Redis请求超时: {e}") from e
        except RedisError as e:
            raise RedisError(ERR_REDIS_ERROR, msg=f"Redis操作异常: {e}") from e


if __name__ == "__main__":
    r = SessionRedis()
    r.set_session("chat_01", [{"role": "user", "content": "Hello"}])
    print(r.get_session("chat_01"))
    r.delete_session("chat_01")
