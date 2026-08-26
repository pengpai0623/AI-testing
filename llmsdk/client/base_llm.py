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
from llmsdk.utils.constants import (
    ERR_CLIENT_DISCONNECT,
    ERR_CONNECT,
    ERR_LLM_HTTP,
    ERR_LLM_NETWORK,
    ERR_SSE_PARSE,
    ERR_VALUE,
    MAX_RETRY_TIMES,
    RETRY_WAIT_SEC,
)
from llmsdk.utils.exceptions import (
    ClientDisconnectError,
    LLMConnectionError,
    LLMHttpError,
    LLMNetworkError,
    LLMSSEParseError,
    LLMValueError,
)


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
        """
        同步流式底层
        注意：生成器函数 @retry 装饰器不生效；流式中途失败不会自动重试，重试逻辑交给上层业务
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise LLMValueError(code=ERR_VALUE, msg="messages不能为空列表")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        resp = None
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
            raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"流式超时：{str(e)}") from e
        except requests.exceptions.ConnectionError as e:
            raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"流式连接失败：{str(e)}") from e
        except requests.exceptions.HTTPError as e:
            code = resp.status_code if resp is not None else 0
            raise LLMHttpError(code=ERR_LLM_HTTP, msg=f"流式HTTP异常 {code}") from e
        except requests.exceptions.ChunkedEncodingError as e:
            # 客户端主动断开TCP连接
            raise ClientDisconnectError(code=ERR_CLIENT_DISCONNECT, msg="客户端断开流式连接") from e

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
                raise LLMSSEParseError(code=ERR_SSE_PARSE, msg=f"SSE JSON解析失败：{str(e)}") from e

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
        """
        异步流式处理
        注意：异步生成器tenacity装饰器不生效，流式迭代中异常不会自动重试
        """
        if not isinstance(messages, list) or len(messages) == 0:
            raise LLMValueError(code=ERR_VALUE, msg="messages不能为空列表")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    headers=self.headers,
                    json=request_body,
                ) as response:
                    response.raise_for_status()

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
                            raise LLMSSEParseError(code=ERR_SSE_PARSE, msg=f"SSE JSON解析失败：{str(e)}") from e

                        if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta

            except httpx.TimeoutException as e:
                raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"流式超时：{str(e)}") from e
            except httpx.ConnectError as e:
                raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"流式连接失败：{str(e)}") from e
            except httpx.ReadError as e:
                # 对端关闭连接，客户端断开
                raise ClientDisconnectError(code=ERR_CLIENT_DISCONNECT, msg="客户端断开异步流式连接") from e
            except httpx.HTTPStatusError as e:
                raise LLMHttpError(
                    code=ERR_LLM_HTTP,
                    msg=f"流式HTTP异常 {e.response.status_code}: {str(e)}",
                ) from e

    @retry(
        stop=stop_after_attempt(MAX_RETRY_TIMES),
        wait=wait_fixed(RETRY_WAIT_SEC),
        retry=retry_if_exception_type((LLMNetworkError, LLMConnectionError)),
        before_sleep=struct_retry_log,
    )
    def _request_messages(self, messages: List[Dict[str, str]], timeout: int, temperature: float) -> Dict:
        """底层同步请求：成功返回 {content, prompt_tokens...}，失败直接raise自定义异常"""
        if not isinstance(messages, list) or len(messages) == 0:
            raise LLMValueError(code=ERR_VALUE, msg="messages不能为空列表，必须传入合法对话消息")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
        }
        resp = None
        try:
            resp = requests.post(
                url=self.endpoint,
                headers=self.headers,
                json=request_body,
                timeout=timeout,
            )
            resp.raise_for_status()
            resp_data = resp.json()

            # 防御：外部返回结构缺失key，封装解析异常
            try:
                content = resp_data["choices"][0]["message"]["content"]
                token_usage = resp_data["usage"]
            except (KeyError, IndexError) as e:
                raise LLMSSEParseError(code=ERR_SSE_PARSE, msg=f"大模型返回数据结构异常: {str(e)}") from e

            return {
                "content": content,
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"],
            }

        except requests.exceptions.Timeout as e:
            raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"请求超时 {str(e)}") from e
        except requests.exceptions.ConnectionError as e:
            raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"连接失败 {str(e)}") from e
        except requests.exceptions.HTTPError as http_err:
            status_code = resp.status_code if resp else 0
            resp_text = resp.text if resp else ""
            if status_code in (502, 503):
                raise LLMConnectionError(code=ERR_CONNECT, msg=f"服务临时故障 {status_code}") from http_err
            raise LLMHttpError(code=ERR_LLM_HTTP, msg=f"HTTP状态码{status_code}, detail:{resp_text}") from http_err
        except Exception as e:
            raise LLMHttpError(code=ERR_LLM_HTTP, msg=f"请求未知异常 {str(e)}") from e

    # @retry(
    #     stop=stop_after_attempt(MAX_RETRY_TIMES),
    #     wait=wait_fixed(RETRY_WAIT_SEC),
    #     retry=retry_if_exception_type((LLMNetworkError, LLMConnectionError)),
    #     before_sleep=struct_retry_log,
    # )
    async def _async_request_messages(self, messages: List[Dict[str, str]], timeout: int, temperature: float) -> Dict:
        """底层异步请求：成功返回 {content, prompt_tokens...}，失败直接raise自定义异常"""
        if not isinstance(messages, list) or len(messages) == 0:
            raise LLMValueError(code=ERR_VALUE, msg="messages不能为空列表，必须传入合法对话消息")

        request_body = {
            "model": self.model_name,
            "temperature": temperature,
            "messages": messages,
        }
        resp = None
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(
                    url=self.endpoint,
                    headers=self.headers,
                    json=request_body,
                    timeout=timeout,
                )
                resp.raise_for_status()
                resp_data = resp.json()

                try:
                    content = resp_data["choices"][0]["message"]["content"]
                    token_usage = resp_data["usage"]
                except (KeyError, IndexError) as e:
                    raise LLMSSEParseError(code=ERR_SSE_PARSE, msg=f"异步大模型返回数据结构异常: {str(e)}") from e

                return {
                    "content": content,
                    "prompt_tokens": token_usage["prompt_tokens"],
                    "completion_tokens": token_usage["completion_tokens"],
                    "total_tokens": token_usage["total_tokens"],
                }

            except httpx.TimeoutException as e:
                raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"异步请求超时 {str(e)}") from e
            except httpx.ConnectError as e:
                raise LLMNetworkError(code=ERR_LLM_NETWORK, msg=f"异步连接失败 {str(e)}") from e
            except httpx.HTTPStatusError as http_err:
                status_code = http_err.response.status_code
                resp_text = http_err.response.text
                if status_code in (502, 503):
                    raise LLMConnectionError(code=ERR_CONNECT, msg=f"异步服务临时故障 {status_code}") from http_err
                raise LLMHttpError(
                    code=ERR_LLM_HTTP,
                    msg=f"异步HTTP状态码{status_code}, detail:{resp_text}",
                ) from http_err
            except Exception as e:
                raise LLMHttpError(code=ERR_LLM_HTTP, msg=f"异步请求未知异常 {str(e)}") from e

    def chat_single(
        self,
        prompt: str,
        timeout: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """单轮对话，自动拼装messages；成功返回 {content, prompt_tokens...}，异常直接raise"""
        use_timeout = timeout if timeout is not None else self.timeout

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._request_messages(messages, use_timeout, temperature)

    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        timeout: Optional[int] = None,
        temperature: float = 0.7,
    ) -> Dict:
        """多轮对话入口，传入完整messages；成功返回字典，异常直接raise"""
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
