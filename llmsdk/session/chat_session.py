from typing import Dict, Generator, List, Optional

import tiktoken

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.utils.constants import DEFAULT_MAX_TOKEN, ERR_VALUE
from llmsdk.utils.exceptions import LLMValueError
from llmsdk.utils.logger import logger


class ChatSession:
    """
    多轮对话上下文管理 + 上下文截断优化
    每个实例 = 独立聊天窗口，隔离上下文。
    """

    def __init__(self, system_content: str = "", max_token_limit=DEFAULT_MAX_TOKEN):
        self.system_content = system_content or "你是通用AI助手，回答简洁清晰"
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_content}]
        self.max_token_limit = max_token_limit

    def _add_user_msg(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def _add_assistant_msg(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def count_tokens_for_messages(self, messages: List[Dict[str, str]], model: str = "gpt-4o") -> int:
        """
        估算 messages 的 prompt token 数量（包含格式开销）。
        注意：基于 cl100k_base 编码计算，与豆包官方实际计数存在差异，仅做本地预估使用。
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def trim_messages_for_context(
        self,
        messages: List[Dict[str, str]],
        model_max_tokens: int,
        max_completion_tokens: int,
        model: str = "gpt-4o",
        keep_system: bool = True,
    ) -> List[Dict[str, str]]:
        """
        上下文截断：超过窗口限制时，删除最早的非 system 对话，直到满足 token 限制。
        :raises LLMValueError: system 消息本身超出可用空间，无法截断
        """
        system_messages = []
        chat_messages = []
        for msg in messages:
            if keep_system and msg.get("role") == "system":
                system_messages.append(msg)
            else:
                chat_messages.append(msg)

        system_tokens = self.count_tokens_for_messages(system_messages, model) if system_messages else 0
        available_for_prompt = model_max_tokens - max_completion_tokens

        if system_tokens > available_for_prompt:
            raise LLMValueError(
                code=ERR_VALUE,
                msg=f"System messages alone ({system_tokens} tokens) exceed the available prompt limit "
                f"({available_for_prompt}) after reserving {max_completion_tokens} tokens for completion.",
            )

        chat_tokens = self.count_tokens_for_messages(chat_messages, model) if chat_messages else 0

        prompt_token = chat_tokens + system_tokens
        remaining_tokens = available_for_prompt - prompt_token
        logger.info(f"当前messages prompt_token为{prompt_token}, 当前上下文剩余可用token为{remaining_tokens}")

        if chat_tokens <= available_for_prompt - system_tokens:
            return messages

        left, right = 0, len(chat_messages)
        while left < right:
            mid = (left + right) // 2
            remaining = chat_messages[mid:]
            remaining_tokens = self.count_tokens_for_messages(remaining, model)
            if remaining_tokens <= available_for_prompt - system_tokens:
                right = mid
            else:
                left = mid + 1

        trimmed_chat = chat_messages[left:]
        return system_messages + trimmed_chat

    def clear_history(self):
        self.messages = [msg for msg in self.messages if msg.get("role") == "system"]

    def reset_system(self, new_system: str):
        self.system_content = new_system
        self.messages = [{"role": "system", "content": new_system}]

    def get_all_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def chat(
        self,
        user_input: str,
        llm: LLMBaseClient,
        timeout: Optional[int] = None,
        max_completion_tokens: int = 300,
        temperature: float = 0.7,
    ) -> Dict:
        """
        会话主聊天方法
        :param user_input: 用户提问
        :param llm: 外部传入 LLM 实例
        :param timeout: 超时，不传使用 LLM 内置默认超时
        :param temperature: 随机性
        :raises LLMBaseError: 截断失败 / 大模型调用异常，抛出前自动回滚本次 user 消息
        """
        self._add_user_msg(user_input)
        try:
            self.messages = self.trim_messages_for_context(
                self.messages,
                model_max_tokens=self.max_token_limit,
                max_completion_tokens=max_completion_tokens,
            )
            resp = llm.chat_with_messages(messages=self.messages, timeout=timeout, temperature=temperature)
        except Exception as e:
            # 任何异常都回滚刚插入的 user 消息，再向上抛出
            self.messages.pop()
            raise e from e

        self._add_assistant_msg(resp["content"])
        return resp

    def chat_stream(
        self,
        user_input: str,
        llm: LLMBaseClient,
        timeout: Optional[int] = None,
        max_completion_tokens: int = 300,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        流式输出，迭代生成器实时拿分片；迭代结束后读取self.stream_last_full_text获取完整回答，避免return与生成器冲突
        调用示例：
        llm = LLMBaseClient()
        session = ChatSession(system_content="你叫小助手，记住用户信息", max_token_limit=4000)
        full_answer = ""
        print("AI：", end="", flush=True)
        stream_gen = session.chat_stream(user_input="我叫张三", llm=llm)
        for chunk in stream_gen:
            print(chunk, flush=True)
            full_answer += chunk
        print(f"\n流式拼接完整结果：{full_answer}")
        """
        self._add_user_msg(user_input)
        self.messages = self.trim_messages_for_context(
            self.messages,
            model_max_tokens=self.max_token_limit,
            max_completion_tokens=max_completion_tokens,
        )

        full_text = ""
        try:
            resp_generator = llm.chat_stream_messages(messages=self.messages, timeout=timeout, temperature=temperature)
            for word in resp_generator:
                full_text += word
                yield word
            self._add_assistant_msg(full_text)
        except Exception as e:
            logger.error(f"流式异常：{repr(e)}，回滚用户消息")
            self.messages.pop()
            raise e from e


if __name__ == "__main__":
    llm = LLMBaseClient()
    session = ChatSession(system_content="你叫小助手，记住用户信息", max_token_limit=4000)
    full_answer = ""
    print("AI：", end="", flush=True)
    stream_gen = session.chat_stream(user_input="我叫张三", llm=llm)
    for chunk in stream_gen:
        print(chunk, flush=True)
        full_answer += chunk
    print(f"\n流式拼接完整结果：{full_answer}")

    res2 = session.chat(user_input="我叫什么名字", llm=llm)
    print("完整消息列表", session.get_all_messages())

    session.clear_history()
