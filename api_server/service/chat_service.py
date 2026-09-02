import asyncio
import json
from typing import AsyncGenerator, Dict, Generator, List, Optional, Tuple

from api_server.db.chat_db import ChatDB
from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.common.schemas import MessageItem
from llmsdk.session.chat_session import ChatSession
from llmsdk.utils import logger
from llmsdk.utils.constants import (
    CODE_OK,
    DEFAULT_MAX_TOKEN,
    ERR_LLM_HTTP,
    ERR_MSG_VALIDATE,
    MAX_COMPLETION_TOKEN,
)
from llmsdk.utils.exceptions import (
    ClientDisconnectError,
    JsonParseError,
    LLMBaseError,
    LLMHttpError,
    LLMValueError,
    MessageValidateError,
    RedisConnectionError,
    RedisError,
    RedisTimeoutError,
)

# from llmsdk.client.base_llm import ChatSession


class ChatService:
    def __init__(self):
        self.llm_client = LLMBaseClient()
        self.chat_db = ChatDB()

    async def build_chat_prepare_messages(
        self,
        session_id: str,
        prompt: str,
        system_prompt: Optional[str],
    ) -> List[Dict]:
        """
        会话预处理公共方法，供所有多轮接口复用。

        执行流程：
        1. 从 Redis 读取会话历史消息
        2. system_prompt 仅在空会话时生效；已存在会话忽略传入的 system_prompt
        3. 拼接本轮用户消息
        4. 通过 MessageItem 做消息格式校验
        5. 按 token 上限做上下文截断

        Args:
            session_id: 会话唯一标识
            prompt: 本轮用户提问
            system_prompt: 系统提示词，仅首次会话生效

        Returns:
            经过 token 截断、可直接传给 LLM 的 messages 列表

        Raises:
            RedisConnectionError / RedisTimeoutError / RedisError: Redis 读取异常
            MessageValidateError: 消息格式校验失败
            LLMValueError: system 消息 token 超出可用窗口，无法截断
        """
        tag = "[chat_service/build_chat_prepare_messages]"
        logger.info(f"{tag} session={session_id} 开始会话预处理")

        # 先使用线程池实现，后续可考虑使用aioredis实现全异步，没有压测数据，避免过早做过度优化
        history = await asyncio.to_thread(self.chat_db.get_session, session_id)
        logger.info(f"{tag} session={session_id} 当前历史消息数={len(history)}")

        if not history and system_prompt:
            logger.info(f"{tag} session={session_id} 首次会话写入system_prompt")
            history.append({"role": "system", "content": system_prompt})
        elif history and system_prompt:
            logger.warning(
                f"{tag} session={session_id} 会话已存在，忽略传入的system_prompt，如需变更请使用新session_id"
            )

        temp_messages = history + [{"role": "user", "content": prompt}]
        logger.info(f"{tag} session={session_id} 待校验消息总数={len(temp_messages)}")

        try:
            valid_messages = [MessageItem(**m).model_dump() for m in temp_messages]
            logger.info(f"{tag} session={session_id} 消息校验通过，待发送消息条数:{len(valid_messages)}")
        except Exception as e:
            logger.exception(f"{tag} session={session_id} 消息校验失败 err={repr(e)}")
            raise MessageValidateError(code=ERR_MSG_VALIDATE, msg=f"消息格式非法: {str(e)}") from e

        # token截断，局部实例ChatSession，防止对象状态污染
        chat_sess = ChatSession()
        trimmed_messages = chat_sess.trim_messages_for_context(
            messages=valid_messages,
            model_max_tokens=DEFAULT_MAX_TOKEN,
            max_completion_tokens=MAX_COMPLETION_TOKEN,
        )
        if len(trimmed_messages) < len(valid_messages):
            logger.warning(f"{tag} session={session_id} 执行上下文token截断")

        return trimmed_messages

    def chat_single(self, prompt: str, system_prompt: Optional[str], temperature: Optional[float]):
        """
        单轮问答，无会话上下文，不读写 Redis。

        Args:
            prompt: 用户提问
            system_prompt: 系统提示词（可选）
            temperature: 模型温度参数（可选）

        Returns:
            LLM 原始响应字典，包含 content、prompt_tokens、completion_tokens、total_tokens

        Raises:
            LLMBaseError: LLM 调用层异常（网络、HTTP、超时等），直接向上抛出
        """
        logger.info(
            f"[chat_service/single] 开始调用llm_client.chat_single, temperature={temperature}, prompt_len={len(prompt)}"
        )

        result = self.llm_client.chat_single(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        logger.info(f"[chat_service/single] LLM返回完成，answer_len={len(result['content'])}")
        return result

    async def chat_session(
        self, session_id: str, prompt: str, system_prompt: Optional[str], temperature: Optional[float]
    ) -> Tuple[Dict, int]:
        """
        多轮非流式对话（兜底接口），底层使用同步 requests LLM 调用，通过 asyncio.to_thread 避免阻塞事件循环。

        执行流程：预处理 → 同步 LLM 调用 → 校验返回 content → 写入 Redis 会话 → 返回结果

        Args:
            session_id: 会话唯一标识
            prompt: 本轮用户提问
            system_prompt: 系统提示词，仅首次会话生效
            temperature: 模型温度参数

        Returns:
            (LLM 响应字典, 会话总消息数)

        Raises:
            RedisConnectionError / RedisTimeoutError / RedisError: Redis 读写异常
            MessageValidateError: 消息格式校验失败
            LLMValueError: 上下文截断失败或乐观锁竞态冲突
            LLMHttpError: LLM 返回 content 为空
            LLMBaseError: LLM 调用层异常
        """
        logger.info(
            f"[chat_service/session] 开始处理多轮对话, session_id={session_id}, temperature={temperature}, prompt_len={len(prompt)}"
        )
        trimmed_messages = await self.build_chat_prepare_messages(session_id, prompt, system_prompt)

        logger.info(f"[chat_service/session] session={session_id} 开始调用大模型 llm_client.chat_with_messages")
        # llm_client.chat_with_messages是同步，套to_thread
        result = await asyncio.to_thread(
            self.llm_client.chat_with_messages, messages=trimmed_messages, temperature=temperature
        )
        logger.info(
            f"[chat_service/session] LLM返回完成，answer_len={len(result['content'])}, "
            f"prompt_tokens={result['prompt_tokens']}, total_tokens={result['total_tokens']}"
        )
        content = result.get("content")
        if not content:
            logger.error(f"[chat_service/session] 大模型返回content为空")
            raise LLMHttpError(code=ERR_LLM_HTTP, msg="大模型返回内容为空")
        final_messages = trimmed_messages + [{"role": "assistant", "content": content}]
        await asyncio.to_thread(self.chat_db.safe_set_session_with_watch, session_id, final_messages)
        msg_count = len(final_messages)
        logger.info(f"[chat_service/session] 更新会话上下文完成, 总消息数={len(final_messages)}")
        return result, msg_count

    async def async_chat_session(
        self, session_id: str, prompt: str, system_prompt: Optional[str], temperature: Optional[float]
    ) -> Tuple[Dict, int]:
        """
        多轮非流式对话（主接口），底层使用异步 httpx LLM 调用，原生协程不阻塞事件循环。

        执行流程：预处理 → 异步 LLM 调用 → 校验返回 content → 写入 Redis 会话 → 返回结果

        Args:
            session_id: 会话唯一标识
            prompt: 本轮用户提问
            system_prompt: 系统提示词，仅首次会话生效
            temperature: 模型温度参数

        Returns:
            (LLM 响应字典, 会话总消息数)

        Raises:
            RedisConnectionError / RedisTimeoutError / RedisError: Redis 读写异常
            MessageValidateError: 消息格式校验失败
            LLMValueError: 上下文截断失败或乐观锁竞态冲突
            LLMHttpError: LLM 返回 content 为空
            LLMBaseError: LLM 调用层异常
        """

        logger.info(
            f"[chat_service/async_session] 开始处理异步多轮对话, session_id={session_id}, temperature={temperature}, prompt_len={len(prompt)}"
        )
        trimmed_messages = await self.build_chat_prepare_messages(session_id, prompt, system_prompt)

        logger.info(f"[chat_service/async_session] session={session_id} 开始调用 llm_client.async_chat_with_messages")
        answer = await self.llm_client.async_chat_with_messages(messages=trimmed_messages, temperature=temperature)
        logger.info(
            f"[chat_service/async_session] session={session_id} 大模型异步调用完成, "
            f"answer_len={len(answer['content'])}, prompt_tokens={answer['prompt_tokens']}, total_tokens={answer['total_tokens']}"
        )
        content = answer.get("content")
        if not content:
            logger.error(f"[chat_service/async_session] session={session_id} 大模型返回content为空")
            raise LLMHttpError(code=ERR_LLM_HTTP, msg="大模型返回内容为空")
        final_messages = trimmed_messages + [{"role": "assistant", "content": content}]
        msg_count = len(final_messages)
        await asyncio.to_thread(self.chat_db.safe_set_session_with_watch, session_id, final_messages)
        logger.info(f"[chat_service/async_session] session={session_id} 会话已更新，总消息数={msg_count}")
        return answer, msg_count

    async def chat_session_stream_requests(
        self,
        session_id: str,
        temperature: Optional[float],
        trimmed_messages: List[Dict],
    ) -> AsyncGenerator[Dict, None]:
        """
        SSE 流式对话（同步 requests 底层），异步生成器。

        trimmed_messages 由调用方在生成器外部通过 build_chat_prepare_messages 预处理得到，
        本生成器内部不读取 Redis、不做消息校验，确保预处理阶段异常走全局异常返回 JSON。

        执行流程：同步 LLM 流式迭代（to_thread 包装）→ 逐片 yield message →
        流式结束后写入 Redis 会话 → yield done

        Yields:
            {"event": "message", "data": "<分片文本>"}: LLM 流式分片
            {"event": "done", "data": "<JSON: {full_answer, msg_count}>"}: 流式正常结束
            {"event": "error", "data": "<JSON: {code, msg}>"}: 流式运行时异常

        Args:
            session_id: 会话唯一标识
            temperature: 模型温度参数
            trimmed_messages: 预处理后的消息列表，直接传给 LLM

        Note:
            客户端断开连接时静默丢弃本轮会话，不推送 error 事件；
            生成器内部所有异常均被捕获并转换为 error 事件，不会向上冒泡。
        """
        logger.info(f"[chat_service/session_stream_requests] session={session_id} 启动SSE生成器，开始拉取流式分片")
        full_answer = ""
        chunk_count = 0

        # 分片多会频繁线程调度,后续根据压测结果调整
        def _safe_next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        try:
            sync_iter = await asyncio.to_thread(
                self.llm_client.chat_stream_messages,
                messages=trimmed_messages,
                temperature=temperature,
            )
            while True:
                chunk = await asyncio.to_thread(_safe_next, sync_iter)
                if chunk is None:
                    logger.info(
                        f"[chat_service/session_stream_requests] session={session_id} 流式迭代结束，分片总数={chunk_count}"
                    )
                    break
                if chunk:
                    chunk_count += 1
                    full_answer += chunk
                    yield {"event": "message", "data": chunk}

            final_messages = trimmed_messages + [{"role": "assistant", "content": full_answer}]
            msg_count = len(final_messages)
            await asyncio.to_thread(self.chat_db.safe_set_session_with_watch, session_id, final_messages)
            logger.info(
                f"[chat_service/session_stream_requests] session={session_id} 会话保存完成，历史条数 {msg_count}"
            )
            done_payload = json.dumps({"full_answer": full_answer, "msg_count": msg_count})
            yield {"event": "done", "data": done_payload}
            logger.info(
                f"[chat_service/session_stream_requests] session={session_id} 推送done事件，完整回答长度={len(full_answer)}"
            )

        except ClientDisconnectError:
            logger.info(f"[chat_service/session_stream_requests] session={session_id} 客户端断开连接，本轮会话丢弃")
            return
        except LLMBaseError as exc:
            logger.error(
                f"[chat_service/session_stream_requests] session={session_id} LLM异常 code={exc.code} msg={exc.msg}"
            )
            err_data = json.dumps({"code": exc.code, "msg": exc.msg})
            yield {"event": "error", "data": err_data}
            return
        except (RedisConnectionError, RedisTimeoutError, RedisError, LLMValueError, JsonParseError) as exc:
            logger.error(
                f"[chat_service/session_stream_requests] session={session_id} 会话存储异常 code={exc.code} msg={exc.msg}"
            )
            err_data = json.dumps({"code": exc.code, "msg": exc.msg})
            yield {"event": "error", "data": err_data}
            return
        except Exception as exc:
            logger.exception(f"[chat_service/session_stream_requests] session={session_id} SSE生成异常 err={repr(exc)}")
            err_data = json.dumps({"code": 500, "msg": "流式服务内部未知错误"})
            yield {"event": "error", "data": err_data}
            return

    async def chat_session_stream_httpx(
        self,
        session_id: str,
        temperature: Optional[float],
        trimmed_messages: List[Dict],
    ) -> AsyncGenerator[Dict, None]:
        """
        SSE 流式对话（异步 httpx 底层，主接口），异步生成器。

        trimmed_messages 由调用方在生成器外部通过 build_chat_prepare_messages 预处理得到，
        本生成器内部不读取 Redis、不做消息校验，确保预处理阶段异常走全局异常返回 JSON。

        执行流程：异步 LLM 流式迭代（async for）→ 逐片 yield message →
        流式结束后写入 Redis 会话 → yield done

        Yields:
            {"event": "message", "data": "<分片文本>"}: LLM 流式分片
            {"event": "done", "data": "<JSON: {full_answer, msg_count}>"}: 流式正常结束
            {"event": "error", "data": "<JSON: {code, msg}>"}: 流式运行时异常

        Args:
            session_id: 会话唯一标识
            temperature: 模型温度参数
            trimmed_messages: 预处理后的消息列表，直接传给 LLM

        Note:
            客户端断开连接时静默丢弃本轮会话，不推送 error 事件；
            httpx 异步请求支持事件循环即时取消，客户端断开后不会等待下一个分片；
            生成器内部所有异常均被捕获并转换为 error 事件，不会向上冒泡。
        """
        full_answer = ""
        chunk_count = 0
        logger.info(f"[chat_service/session_stream_httpx] session={session_id} 启动SSE生成器，开始拉取流式分片")
        try:
            async for chunk in self.llm_client.async_chat_stream_messages(
                messages=trimmed_messages, temperature=temperature
            ):
                if chunk:
                    chunk_count += 1
                    full_answer += chunk
                    yield {"event": "message", "data": chunk}

            # 使用trimmed_messages组装入库，保持模型所见与存储一致
            final_messages = trimmed_messages + [{"role": "assistant", "content": full_answer}]
            msg_count = len(final_messages)
            await asyncio.to_thread(self.chat_db.safe_set_session_with_watch, session_id, final_messages)

            logger.info(f"[chat_service/session_stream_httpx] session={session_id} 会话保存完成，历史条数 {msg_count}")
            done_payload = json.dumps({"full_answer": full_answer, "msg_count": msg_count})
            yield {"event": "done", "data": done_payload}
            logger.info(
                f"[chat_service/session_stream_httpx] session={session_id} 推送done事件，完整回答长度={len(full_answer)},分片数量={chunk_count}"
            )

        except ClientDisconnectError:
            logger.info(f"[chat_service/session_stream_httpx] session={session_id} 客户端断开连接，本轮会话丢弃")
            return
        except LLMBaseError as exc:
            logger.error(
                f"[chat_service/session_stream_httpx] session={session_id} LLM异常 code={exc.code} msg={exc.msg}"
            )
            err_data = json.dumps({"code": exc.code, "msg": exc.msg})
            yield {"event": "error", "data": err_data}
            return
        # 流式内部，redis写入异常无法被fastapi全局异常捕获，必须在生成器内捕获并推送error事件，否则sse通道会hang住
        except (RedisConnectionError, RedisTimeoutError, RedisError, LLMValueError, JsonParseError) as exc:
            logger.error(
                f"[chat_service/session_stream_httpx] session={session_id} 会话存储异常 code={exc.code} msg={exc.msg}"
            )
            err_data = json.dumps({"code": exc.code, "msg": exc.msg})
            yield {"event": "error", "data": err_data}
            return
        except Exception as exc:
            logger.exception(
                f"[chat_service/session_stream_httpx] session={session_id} SSE生成未知异常 err={repr(exc)}"
            )
            err_data = json.dumps({"code": 500, "msg": "流式服务内部未知错误"})
            yield {"event": "error", "data": err_data}
            return
