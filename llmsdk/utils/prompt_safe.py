import re

# 高危劫持关键词正则，辅助过滤，仅前置拦截，不能单独依赖
INJECT_PATTERNS = [
    re.compile(r"忽略(以上|前面|全部|所有).*指令", re.IGNORECASE),
    re.compile(r"忘记.*角色", re.IGNORECASE),
    re.compile(r"覆盖.*系统提示", re.IGNORECASE),
    re.compile(r"输出.*system.*prompt", re.IGNORECASE),
    re.compile(r"你的新角色", re.IGNORECASE),
]

# 特殊分隔符，移除用户输入里容易混淆模型的分割标记
SPLIT_MARKERS = ["---", "===", "***", "````"]


def check_prompt_injection(text: str) -> tuple[bool, str]:
    """
    检测是否疑似prompt注入
    return (是否风险, 清洗后文本)
    """
    if not text:
        return False, text

    # 移除容易混淆模型的分割线
    clean_text = text
    for mark in SPLIT_MARKERS:
        clean_text = clean_text.replace(mark, "")

    # 关键词匹配
    for pat in INJECT_PATTERNS:
        if pat.search(clean_text):
            return True, clean_text

    return False, clean_text


def wrap_user_content(content: str) -> str:
    """
    给不可信用户内容加上隔离标签
    """
    return f"<USER_INPUT>\n{content}\n</USER_INPUT>"
