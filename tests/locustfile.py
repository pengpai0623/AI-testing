"""
Day26 全链路压测脚本
用法：
  无头模式：
  .\\ai-env\\Scripts\\locust.exe -f tests/locustfile.py --host=http://127.0.0.1:8000  -u 20 -r 1 -t 10m --headless --csv=tests/result
"""

import random
import uuid
from collections import defaultdict

from locust import HttpUser, between, events, task

# ============ 线程安全的统计字典 ============
stats = defaultdict(int)


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    total = stats["total"] or 1
    print("\n" + "=" * 60)
    print("压测结果汇总")
    print("=" * 60)
    print(f"  总请求数:             {stats['total']}")
    print(f"  成功(code=0):        {stats['success']}  ({stats['success']/total*100:.1f}%)")
    print(f"  预期限流(42900):     {stats['rate_limited']}  ({stats['rate_limited']/total*100:.1f}%)")
    print(f"  参数校验(422):       {stats['param_error']}")
    print(f"  其他业务错误:         {stats['biz_error']}")
    print(f"  网络/HTTP错误:       {stats['http_error']}")
    print("-" * 60)
    real_fail = stats["biz_error"] + stats["http_error"]
    print(f"  真实失败率(不含限流): {real_fail/total*100:.2f}%")
    print("=" * 60 + "\n")


class ChatUser(HttpUser):
    """模拟前端用户访问对话接口"""

    wait_time = between(1, 3)

    def on_start(self):
        self.session_id = f"locust-{uuid.uuid4().hex[:12]}"

    def _handle(self, resp):
        """统一处理响应，根据 code 动态重命名 locust 统计条目"""
        stats["total"] += 1
        try:
            body = resp.json()
            code = body.get("code", -1)

            if code == 0:
                stats["success"] += 1
                resp.request_meta["name"] += " (成功)"
                resp.success()
            elif code == 42900:
                stats["rate_limited"] += 1
                resp.request_meta["name"] += " (限流)"
                resp.success()  # 限流是预期行为，不计入 locust 失败
            elif code == 422:
                stats["param_error"] += 1
                resp.request_meta["name"] += " (参数错误)"
                resp.failure(f"code=422: {body.get('msg', '')[:100]}")
            else:
                stats["biz_error"] += 1
                resp.request_meta["name"] += f" (业务错误{code})"
                resp.failure(f"code={code}: {body.get('msg', '')[:100]}")
        except Exception as e:
            stats["http_error"] += 1
            resp.request_meta["name"] += " (网络错误)"
            resp.failure(f"parse/http error: {e}")

    @task(3)
    def health(self):
        with self.client.get("/chat/health", name="/chat/health", catch_response=True) as resp:
            self._handle(resp)

    @task(2)
    def single_chat(self):
        prompt = random.choice(
            [
                "你好，用一句话介绍自己",
                "Python 和 Java 有什么区别？一句话",
                "写一个快速排序的思路",
                "深圳今天天气怎么样？简短回答",
            ]
        )
        with self.client.post(
            "/chat/single",
            json={"prompt": prompt, "temperature": 0.7},
            name="/chat/single",
            catch_response=True,
            timeout=120,
        ) as resp:
            self._handle(resp)

    @task(1)
    def multi_chat_session(self):
        prompt = random.choice(
            [
                "一句话介绍事务循环",
                "httpx相较于requests的优势是什么，一句话回答",
                "什么是悲观锁？一句话解释",
            ]
        )
        with self.client.post(
            "/chat/session",
            json={"session_id": self.session_id, "prompt": prompt, "temperature": 0.7},
            name="/chat/session",
            catch_response=True,
            timeout=120,
        ) as resp:
            self._handle(resp)

    @task(1)
    def multi_chat_async_session(self):
        prompt = random.choice(
            [
                "fastapi的优势是什么，一句话回答",
                "用三句话介绍 Redis",
                "什么是乐观锁？一句话解释",
            ]
        )
        with self.client.post(
            "/chat/async_session",
            json={"session_id": self.session_id, "prompt": prompt, "temperature": 0.7},
            name="/chat/async_session",
            catch_response=True,
            timeout=120,
        ) as resp:
            self._handle(resp)
