from llmsdk.utils.exceptions import RedisConnectionError, RedisError, RedisTimeoutError
from llmsdk.utils.session_redis import SessionRedis


class ChatDB:
    def __init__(self):
        self._redis = SessionRedis()

    def get_session(self, session_id: str):
        return self._redis.get_session(session_id)

    def set_session(self, session_id: str, messages: list):
        return self._redis.set_session(session_id, messages)

    def safe_set_session_with_watch(self, session_id: str, messages: list):
        return self._redis.safe_set_session_with_watch(session_id, messages)

    def delete_session(self, session_id: str):
        return self._redis.delete_session(session_id)
