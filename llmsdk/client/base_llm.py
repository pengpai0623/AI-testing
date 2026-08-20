import json
from typing import AsyncGenerator, Dict, Generator, List, Optional

import httpx
import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from llmsdk.config.settings import (
    DOUBAO_API_KEY,
    DOUBAO_ENDPOINT,
    DOUBAO_MODEL,
    REQUEST_TIMEOUT,
)
from llmsdk.utils.common import struct_retry_log
from llmsdk.utils.constants import MAX_RETRY_TIMES, RETRY_WAIT_SEC
from llmsdk.utils.exceptions import LLMHttpError, LLMNetworkError, LLMSSEParseError


class LLMBaseClient:
    def __init__(self):
        self.api_key = DOUBAO_API_KEY
        self.endpoint = DOUBAO_ENDPOINT
        self.model_name = DOUBAO_MODEL
        self.timeout = REQUEST_TIMEOUT

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_stream_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: int,
        temperature: float,
    ) -> Generator[str, None, None]:
        # 同步流式处理
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages不能为空列表")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        resp = None  # 初始化，避免异常处理中未定义
        try:
            resp = requests.post(
                url=self.endpoint,
                headers=self.headers,
                json=request_body,
                timeout=timeout,
                stream=True,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout as e:
            raise LLMNetworkError(f"流式超时：{e}")
        except requests.exceptions.ConnectionError as e:
            raise LLMNetworkError(f"流式连接失败：{e}")
        except requests.exceptions.HTTPError as e:
            code = resp.status_code if resp is not None else 0
            raise LLMHttpError(f"流式HTTP异常 {code}: {e}")

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line or not line.startswith("data:"):
                continue

            _, _, data_str = line.partition("data:")
            data_str = data_str.strip()
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError as e:
                raise LLMSSEParseError(f"SSE JSON解析失败：{e}")

            if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta

    async def _async_request_stream_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: int,
        temperature: float,
    ) -> AsyncGenerator[str, None]:
        # 异步流式处理
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages不能为空列表")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                # 这里使用 .stream() 方法，且必须 await
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=self.headers,
                    json=request_body,
                ) as response:
                    response.raise_for_status()

                    # 使用 aiter_lines() 异步迭代每一行，而不是同步的 iter_lines
                    async for raw_line in response.aiter_lines():
                        if not raw_line:
                            continue
                        if not raw_line.startswith("data:"):
                            continue

                        _, _, data_str = raw_line.partition("data:")
                        data_str = data_str.strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError as e:
                            raise LLMSSEParseError(f"SSE JSON解析失败：{e}")

                        if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta  # 在异步生成器中，yield 不需要 await

            except httpx.TimeoutException as e:
                raise LLMNetworkError(f"流式超时：{e}")
            except httpx.ConnectError as e:
                raise LLMNetworkError(f"流式连接失败：{e}")
            except httpx.HTTPStatusError as e:
                raise LLMHttpError(f"流式HTTP异常 {e.response.status_code}: {e}")

    @retry(
        stop=stop_after_attempt(MAX_RETRY_TIMES),
        wait=wait_fixed(RETRY_WAIT_SEC),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError)),
        before_sleep=struct_retry_log,
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
            resp = requests.post(
                url=self.endpoint,
                headers=self.headers,
                json=request_body,
                timeout=timeout,
            )
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
            return {"status": "timeout", "content": "请求超时"}
        except requests.exceptions.ConnectionError:
            return {"status": "conn_error", "content": "网络连接失败"}
        except requests.exceptions.HTTPError as http_err:
            resp_text = resp.text if resp else ""
            status_code = resp.status_code if resp else 0
            if status_code == 429:
                return {"status": "limit_429", "content": "接口限流"}
            elif status_code in [400, 401, 403]:
                return {
                    "status": "http_err",
                    "content": f"{status_code} 权限/参数错误：{resp_text}",
                }
            elif status_code in [502, 503]:
                raise ConnectionError("服务临时故障，触发重试")
            else:
                return {
                    "status": "http_err",
                    "content": f"{status_code} {str(http_err)}",
                }
        except Exception as e:
            return {"status": "unknown_err", "content": f"未知异常：{str(e)}"}

    @retry(
        stop=stop_after_attempt(MAX_RETRY_TIMES),
        wait=wait_fixed(RETRY_WAIT_SEC),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        before_sleep=struct_retry_log,
    )
    async def _async_request_messages(self, messages: List[Dict[str, str]], timeout: int, temperature: float) -> Dict:
        """底层异步私有请求方法：接收标准OpenAI messages数组，发起http请求"""

        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("messages不能为空列表，必须传入合法对话消息")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                # client.post返回协程对象，需要await等待
                resp = await client.post(
                    url=self.endpoint,
                    headers=self.headers,
                    json=request_body,
                    timeout=timeout,
                )
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

            except httpx.TimeoutException:
                return {"status": "timeout", "content": "请求超时"}
            except httpx.ConnectError:
                return {"status": "conn_error", "content": "网络连接失败"}
            except httpx.HTTPStatusError as http_err:
                status_code = http_err.response.status_code
                resp_text = http_err.response.text
                if status_code == 429:
                    return {"status": "limit_429", "content": "接口限流"}
                elif status_code in [400, 401, 403]:
                    return {
                        "status": "http_err",
                        "content": f"{status_code} 权限/参数错误：{resp_text}",
                    }
                elif status_code in [502, 503]:
                    raise ConnectionError("服务临时故障，触发重试")
                else:
                    return {
                        "status": "http_err",
                        "content": f"{status_code} {str(http_err)}",
                    }
            except Exception as e:
                return {"status": "unknown_err", "content": f"未知异常：{str(e)}"}

    def chat_single(
        self,
        prompt: str,
        timeout: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
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
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        print("messages", messages)

        return self._request_messages(messages, use_timeout, temperature)

    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Dict:
        """
        多轮对话入口：外部直接传入完整OpenAI格式messages数组
        :param messages: 标准OpenAI消息列表，包含system/user/assistant
        :param timeout: 超时，为空使用全局默认
        :param temperature: 随机系数
        """
        use_timeout = timeout if timeout is not None else self.timeout
        return self._request_messages(messages, use_timeout, temperature)

    async def async_chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Dict:
        """
        多轮对话入口：外部直接传入完整OpenAI格式messages数组
        :param messages: 标准OpenAI消息列表，包含system/user/assistant
        :param timeout: 超时，为空使用全局默认
        :param temperature: 随机系数
        """
        use_timeout = timeout if timeout is not None else self.timeout
        return await self._async_request_messages(messages, use_timeout, temperature)

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

    async def async_chat_stream_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:

        use_timeout = timeout if timeout is not None else self.timeout

        async for chunk in self._async_request_stream_messages(messages, use_timeout, temperature):
            yield chunk
