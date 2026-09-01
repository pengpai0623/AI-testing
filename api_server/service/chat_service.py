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
        会话预处理公共逻辑：
        1.读取redis历史会话
        2.system_prompt规则处理（仅空会话生效）
        3.拼接本轮用户消息
        4.MessageItem格式校验
        5.token上下文截断
        返回：经过token截断、可直接传给LLM的messages列表

        异常：Redis系列异常 / MessageValidateError / LLMValueError，全部向上抛出，交给上层全局异常处理器
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
        单轮问答，无会话上下文
        业务日志全部放在service；底层异常直接向上抛出，不捕获
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
        非流式同步llm调用，多轮对话接口
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
        非流式异步，异步多轮对话接口，session_id维护上下文
        - session_id: 会话ID
        - prompt: 用户本轮提问
        - system_prompt: 仅首次会话生效
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
        流式同步，SSE流式，底层同步requests实现；
        trimmed_messages由router调用build_chat_prepare_messages得到，不在本生成器读取redis
        yield事件：message / done / error
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
        流式异步，SSE流式，底层异步httpx；全部业务收敛service层
        yield事件：message / done / error
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
