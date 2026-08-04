from typing import Dict, Generator, List, Optional

import tiktoken

from llmsdk.client.base_llm import LLMBaseClient
from llmsdk.utils.constants import (
    CHINESE_TOKEN_RATIO,
    CUT_PAIR_PER_TIME,
    DEFAULT_MAX_TOKEN,
    DEFAULT_TEMPERATURE,
    MIN_KEEP_MSG_NUM,
)
from llmsdk.utils.logger import logger


class ChatSession:
    """
    多轮对话上下文管理 + 上下文截断优化
    每个实例 = 独立聊天窗口，隔离上下文。

    一, 完整执行顺序
        1, 初始化指定system_content角色, 记录初始messages(仅仅是 -> [
            {"role": "system", "content": self.system_content}
        ])
        2, 首次发送user_input, 收到assistant后将user_input & assistant append messages
        3, 后续发送(完整)messages + (新)user_input, AI 即会记住之前内容
        4, 记录总 token 超限自动删掉最早的历史消息，防止超出模型窗口上限
    二, 标准 OpenAI Chat 消息格式
        messages 是有序列表，聊天顺序严格从上到下，不能乱序、不能角色错乱。
        {
            "model": "接入点ID",
            "messages": [
                {"role": "system", "content": "角色内容"},
                {"role": "user", "content": "用户第1轮问题"},
                {"role": "assistant", "content": "AI第1轮回答"},
                {"role": "user", "content": "用户第2轮问题"},
                {"role": "assistant", "content": "AI第2轮回答"}
            ],
            "temperature": 0.7,
            "stream": false
    }
    三, 踩过的坑
    一定要明确需求后再进行开发，否则代码会十分混乱，如每个实例 = 独立聊天窗口，隔离上下文。每个实例只需初始化一次即可，参考测试代码，后续设置init方法时要格外注意
    """

    def __init__(self, system_content: str = "", max_token_limit=DEFAULT_MAX_TOKEN):
        # 禁止system为空，兜底默认角色
        self.system_content = system_content or "你是通用AI助手，回答简洁清晰"
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_content}]
        self.max_token_limit = max_token_limit

    def _add_user_msg(self, content: str):
        """追加user消息"""
        self.messages.append({"role": "user", "content": content})

    def _add_assistant_msg(self, content: str):
        """追加AI返回内容"""
        self.messages.append({"role": "assistant", "content": content})

    def count_tokens_for_messages(self, messages: List[Dict[str, str]], model: str = "gpt-4o") -> int:
        """
        计算 messages 列表的 prompt token（包括AI回答消耗） 数量（包含格式开销）。
        这个计算结果与 API 响应的 usage.prompt_tokens 基本一致。
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3
        tokens_per_name = 1  # 如果消息有 name 字段，额外加 1（一般不用）
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # 在 prompt 末尾附加的 assistant 引导标记
        return num_tokens

    def trim_messages_for_context(
        self,
        messages: List[Dict[str, str]],
        model_max_tokens: int,  # 模型最大上下文，例如 gpt-4o 为 128000
        max_completion_tokens: int,  # 本次回答预留的最大 token 数
        model: str = "gpt-4o",  # 不需要，先保留
        keep_system: bool = True,
    ) -> List[Dict[str, str]]:
        """
        如果 prompt_tokens + max_completion_tokens 超过 model_max_tokens，
        则从最前面开始删除非 system 的对话（保留 system 和最新对话），
        直到满足限制。返回截断后的 messages 列表。
        """
        # 1. 分离 system 消息和普通对话
        system_messages = []
        chat_messages = []
        for msg in messages:
            if keep_system and msg.get("role") == "system":
                system_messages.append(msg)
            else:
                chat_messages.append(msg)

        # 2. 计算 system 的 token 数
        system_tokens = self.count_tokens_for_messages(system_messages, model) if system_messages else 0

        # 3. 为 prompt 预留的最大 token 数（总上下文 - 预留的 completion）
        available_for_prompt = model_max_tokens - max_completion_tokens

        if system_tokens > available_for_prompt:
            raise ValueError(
                f"System messages alone ({system_tokens} tokens) exceed the available prompt limit "
                f"({available_for_prompt}) after reserving {max_completion_tokens} tokens for completion."
            )

        # 4. 如果当前 chat 部分已经满足限制，直接返回原始 messages
        chat_tokens = self.count_tokens_for_messages(chat_messages, model) if chat_messages else 0

        prompt_token = chat_tokens + system_tokens
        remaining_tokes = available_for_prompt - prompt_token
        logger.info(f"当前messases prompt_token为{prompt_token}, 当前上下文剩余可用token为{remaining_tokes}")
        if chat_tokens <= available_for_prompt - system_tokens:
            return messages

        # 5. 需要截断：二分查找最小的起始索引，使得 chat_messages[start:] 的 token 数 <= 可用空间
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
        """对外公共方法：清空所有聊天，保留system"""
        self.messages = [msg for msg in self.messages if msg.get("role") == "system"]

    def reset_system(self, new_system: str):
        """更换system角色，全量清空历史"""
        self.system_content = new_system
        self.messages = [{"role": "system", "content": new_system}]

    def get_all_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()  # 防止外部修改 messages

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
        :param llm: 外部传入LLM实例，解除全局硬编码
        :param timeout: 超时，不传使用LLM内置默认超时
        :param temperature: 随机性
        """
        self._add_user_msg(user_input)
        self.messages = self.trim_messages_for_context(
            self.messages,
            model_max_tokens=self.max_token_limit,
            max_completion_tokens=max_completion_tokens,
        )

        # 不传timeout则使用LLM自带全局超时
        resp = llm.chat_with_messages(messages=self.messages, timeout=timeout, temperature=temperature)

        if resp["status"] == "success":
            self._add_assistant_msg(resp["content"])
        else:
            # 请求异常，回滚删掉刚添加的user
            self.messages.pop()
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
            # 正常结束入库
            self._add_assistant_msg(full_text)
        except Exception as e:
            print(f"流式异常：{repr(e)}，回滚用户消息")
            self.messages.pop()
            return


if __name__ == "__main__":
    # 测试代码
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

    # 手动清空历史
    session.clear_history()
