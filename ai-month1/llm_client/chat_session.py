import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional
from llm_client.base_llm import LLMBaseClient


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
    def __init__(self, system_content: str, max_token_limit: int = 6000):
        # 禁止system为空，兜底默认角色
        self.system_content = system_content or "你是通用AI助手，回答简洁清晰"
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_content}
        ]
        self.max_token_limit = max_token_limit

    def _add_user_msg(self, content: str):
        """追加user消息"""
        self.messages.append({"role": "user", "content": content})

    def _add_assistant_msg(self, content: str):
        """追加AI返回内容"""
        self.messages.append({"role": "assistant", "content": content})

    def estimate_total_token(self, messages: List[Dict[str, str]]) -> float:
        """根据当前 messages 整体预估 token，中文1.5token/汉字"""
        total_tokens = 0.0
        for chat_mes in messages:
            content = chat_mes["content"]
            current_chat_tokens = len(content) * 1.5
            total_tokens += current_chat_tokens
        print(f"[Token统计] 当前总预估token：{total_tokens:.1f}")
        return total_tokens

    def _del_history_content(self, del_pair_num: int):
        """删除前面N对对话，返回新消息列表，带边界防护"""
        system_msg = self.messages[0]
        dialog_list = self.messages[1:]
        need_cut_length = del_pair_num * 2

        # 对话条数不足，不删除
        if len(dialog_list) < need_cut_length:
            print("对话数量不足，无法继续裁剪")
            return self.messages

        remain_dialog = dialog_list[need_cut_length:]
        new_messages = [system_msg] + remain_dialog
        return new_messages

    def _cut_history_auto(self):
        """token超限自动裁剪，每次删最早1对，禁止删到只剩system"""
        while True:
            current_tokens = self.estimate_total_token(self.messages)
            if current_tokens < self.max_token_limit:
                break

            # 只剩system+1对对话，不再裁剪兜底
            if len(self.messages) <= 3:
                print("对话只剩最少一轮，停止裁剪避免无上下文")
                break

            print(f"Token超限{current_tokens:.1f}/{self.max_token_limit}，删除最早1轮对话")
            # 必须赋值覆盖
            self.messages = self._del_history_content(del_pair_num=1)

    def clear_history(self):
        """对外公共方法：清空所有聊天，保留system"""
        self.messages = self.messages[:1]

    def reset_system(self, new_system: str):
        """更换system角色，全量清空历史"""
        self.system_content = new_system
        self.messages = [{"role": "system", "content": new_system}]

    def get_all_messages(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def chat(
        self,
        user_input: str,
        llm: LLMBaseClient,
        timeout: Optional[int] = None,
        temperature: float = 0.7
    ) -> Dict:
        """
        会话主聊天方法
        :param user_input: 用户提问
        :param llm: 外部传入LLM实例，解除全局硬编码
        :param timeout: 超时，不传使用LLM内置默认超时
        :param temperature: 随机性
        """
        self._add_user_msg(user_input)
        self._cut_history_auto()

        # 不传timeout则使用LLM自带全局超时
        resp = llm.chat_with_messages(
            messages=self.messages,
            timeout=timeout,
            temperature=temperature
        )

        if resp["status"] == "success":
            self._add_assistant_msg(resp["content"])
        else:
            # 请求异常，回滚删掉刚添加的user
            self.messages.pop()
        return resp


if __name__ == "__main__":
    # 测试代码
    llm = LLMBaseClient()
    session = ChatSession(system_content="你叫小助手，记住用户信息", max_token_limit=4000)
    res = session.chat(user_input="我叫张三", llm=llm)
    print(res["content"])
    res2 = session.chat(user_input="我叫什么名字", llm=llm)
    print(res2["content"])
    print("完整消息列表", session.get_all_messages())
    # 手动清空历史
    session.clear_history()
