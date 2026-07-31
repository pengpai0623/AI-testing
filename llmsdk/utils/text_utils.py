import re


def clean_ai_json(raw_text: str) -> str:
    """
    通用AI文本清洗：去除markdown代码块、换行、制表符，返回干净JSON字符串
    全SDK统一调用，base、结构化共用
    """
    content = raw_text.strip()
    md_pattern = r"```(?:json)?\n?([\s\S]*?)\n?```"
    match_result = re.search(md_pattern, content)
    if match_result:
        content = match_result.group(1).strip()
    content = content.replace("\n", "").replace("\t", "")
    return content.strip()
