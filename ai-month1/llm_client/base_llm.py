import json
import os
from typing import Dict, Generator, List, Optional

import requests
from dotenv import load_dotenv
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

# 加载环境变量
load_dotenv()

# 重试全局配置
MAX_RETRY_TIMES = 3  # 最大重试3次
RETRY_WAIT_SEC = 2  # 每次重试间隔2秒


def retry_log_callback(retry_state: RetryCallState):
    """每次重试触发时打印日志"""

    exc = retry_state.outcome.exception()
    print(f"【LLM重试】第{retry_state.attempt_number}次重试，等待{RETRY_WAIT_SEC}s，异常：{exc}")


class LLMBaseClient:
    def __init__(self):
        self.api_key = os.getenv("DOUBAO_API_KEY")
        self.endpoint = os.getenv("DOUBAO_ENDPOINT")
        self.model_name = os.getenv("DOUBAO_MODEL")
        # 转为int全局默认超时时间
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", 60))

        if not self.api_key:
            raise ValueError("DOUBAO_API_KEY 未配置，请检查.env文件")

        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _request_stream_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: int,
        temperature: float,
    ) -> Generator[str, None, None]:
        """
        流式 SSE（Server-Sent Events 服务端推送事件）
        和 websocket 简单区分
        SSE：HTTP 协议，服务端单向发消息，适合大模型流式，简单轻量；
        WebSocket：双向收发，适合企微实时聊天收发消息。

        大模型一边生成文字，一边把内容切成无数小分片（chunk）实时推送。
        浏览器、客户端持续长连接接收数据，不用等全部结束。
        传输协议：HTTP 长连接，服务端单向不断发数据，客户端只收不发；
        大模型流式本质：遵循 SSE 规范，一行一条data: {json分片}\n\n；
        终止标志：最后会返回data: [DONE]代表流式结束。

        流式输出实例
        data: {"choices":[{"delta":{"content":"你"}}]}
        data: {"choices":[{"delta":{}}]} # 有可能为空行
        data: {"choices":[{"delta":{"content":"好"}}]}
        data: [DONE]
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages不能为空列表，必须传入合法对话消息")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
            "stream": True,  # 以 SSE 多行data:xxx分片持续推送内容
        }

        try:
            resp = requests.post(
                url=self.endpoint,
                headers=self.headers,
                json=request_body,
                timeout=timeout,
                stream=True,  # requests流式开关
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise ConnectionError(f"流式请求超时: {e}")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"流式网络连接失败: {e}")
        except requests.exceptions.HTTPError as e:
            code = resp.status_code if "resp" in locals() else 0
            raise RuntimeError(f"流式HTTP异常 {code}: {e}")

        for raw_line in resp.iter_lines():  # 按行流式读取响应，不会一次性加载全部响应到内存；
            """
            长连接持续接收服务端源源不断的字节流；
            内部不断缓存字节，碰到换行符就切出一行，以 bytes 抛出给循环；
            没有完整一行时继续阻塞等待网络数据，不会提前返回；
            只处理以data:开头的有效行，其他注释、心跳行全部丢弃；
            传输结束（收到[DONE]、连接关闭），循环正常终止。
            """
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")  # 强制utf8解码
            if not line or not line.startswith("data:"):
                continue

            _, _, data_str = line.partition("data:")
            data_str = data_str.strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except json.JSONDecodeError:
                continue

    @retry(
        stop=stop_after_attempt(MAX_RETRY_TIMES),
        wait=wait_fixed(RETRY_WAIT_SEC),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
        before_sleep=retry_log_callback,
    )
    def _request_messages(self, messages: List[Dict[str, str]], timeout: int, temperature: float) -> Dict:
        """底层私有请求方法：接收标准OpenAI messages数组，发起http请求"""

        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages不能为空列表，必须传入合法对话消息")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
        }

        try:
            resp = requests.post(url=self.endpoint, headers=self.headers, json=request_body, timeout=timeout)
            resp.raise_for_status()  # 非2xx抛出HTTPError
            resp_data = resp.json()

            content = resp_data["choices"][0]["message"]["content"]
            token_usage = resp_data["usage"]

            return {
                "status": "success",
                "content": content,
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"],
            }

        except requests.exceptions.Timeout:
            return {"status": "timeout", "content": "请求大模型超时，请稍后重试"}
        except requests.exceptions.ConnectionError:
            return {"status": "conn_error", "content": "网络连接失败，无法访问模型接口"}
        except requests.exceptions.HTTPError as http_err:
            resp_text = resp.text if "resp" in locals() else ""
            status_code = resp.status_code if "resp" in locals() else 0

            if status_code == 429:
                # 文案提示限流，单独返回，绝不重试
                return {"status": "limit_429", "content": "接口调用触发限流，请降低调用频率，稍后重试"}
            elif status_code in [400, 401, 403]:
                # 参数、密钥错误，重试没用
                return {"status": "http_err", "content": f"接口异常{status_code}：{str(http_err)}，响应：{resp_text}"}
            elif status_code in [502, 503]:
                # 临时服务故障，抛出异常触发重试
                raise ConnectionError(f"服务临时不可用 {status_code}")
            else:
                return {"status": "http_err", "content": f"接口异常：{str(http_err)}，响应内容：{resp_text}"}
        except Exception as e:
            return {"status": "unknown_err", "content": f"未知异常：{str(e)}"}

    def chat_single(
        self, prompt: str, timeout: Optional[int] = None, temperature: float = 0.7, system_prompt: Optional[str] = None
    ) -> Dict:
        """
        兼容Day1老接口：单轮对话，自动拼装OpenAI messages
        :param prompt: 用户当前提问
        :param timeout: 请求超时时间，不传则使用实例全局默认超时
        :param temperature: 随机性 0~2，越小回答越固定
        :param system_prompt: 系统角色设定，可为空
        :return: 统一标准化返回字典
        """
        # 不传timeout就用类初始化的全局超时
        use_timeout = timeout if timeout is not None else self.timeout

        messages: List[Dict[str, str]] = []
        # ✅ 修复：system必须是role=system，不能是user
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._request_messages(messages, use_timeout, temperature)

    def chat_with_messages(
        self, messages: List[Dict[str, str]], timeout: Optional[int] = None, temperature: float = 0.7
    ) -> Dict:
        """
        多轮对话入口：外部直接传入完整OpenAI格式messages数组
        :param messages: 标准OpenAI消息列表，包含system/user/assistant
        :param timeout: 超时，为空使用全局默认
        :param temperature: 随机系数
        """
        use_timeout = timeout if timeout is not None else self.timeout
        return self._request_messages(messages, use_timeout, temperature)

    def chat_stream_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:

        use_timeout = timeout if timeout is not None else self.timeout
        # yield from 自动遍历内部生成器，把内部所有产出的值一层层抛给外层调用者
        # yield from X == for i in X: yield i；
        yield from self._request_stream_messages(messages, use_timeout, temperature)


if __name__ == "__main__":
    llm = LLMBaseClient()
    stream_gen = llm.chat_stream_messages(
        [
            {"role": "system", "content": "你是作家"},
            {"role": "user", "content": "今天是2026.7.29，写一篇 400 字深圳夏日散文"},
        ]
    )
    full_text = ""
    print("流式输出：")
    for word in stream_gen:
        full_text += word
        print("这次的流式输出是：", word, flush=True)
    print(f"\n完整结果：{full_text}")
