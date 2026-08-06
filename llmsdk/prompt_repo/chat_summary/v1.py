from llmsdk.prompt_repo.base_template import BasePromptTemplate


class ChatSummaryV1(BasePromptTemplate):
    version = "v1_zero"
    scene = "对话/文本轻量总结"
    template = """
# 角色：文本摘要助手
# 要求
1. 对 {source_text} 做精简总结，控制在 {limit_word}
2. 只保留核心事件、人物、诉求，剔除语气词、重复废话
3. 输出格式：{output_format}
4. 禁止多余开场白、客套话，无有效内容直接返回「无有效信息」
"""


summary_v1 = ChatSummaryV1()
