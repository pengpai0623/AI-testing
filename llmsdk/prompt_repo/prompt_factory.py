# llmsdk/prompt_repo/prompt_factory.py
from llmsdk.utils.constants import PROMPT_CONFIG


# 商品抽取模板工厂
def get_product_extract_template():
    ver = PROMPT_CONFIG["product_extract_version"]
    if ver == "v2_fewshot":
        from .struct_extract import product_extract_prompt_v2

        return product_extract_prompt_v2
    from .struct_extract import product_extract_prompt_v1

    return product_extract_prompt_v1


# 代码分析模板工厂
def get_code_analyze_template():
    ver = PROMPT_CONFIG["code_analyze_version"]
    if ver == "v2_CoT":
        from .code_analyze import code_analyze_cot_v2

        return code_analyze_cot_v2
    from .code_analyze import code_analyze_v1

    return code_analyze_v1


# 对话总结模板工厂
def get_chat_summary_template():
    ver = PROMPT_CONFIG["chat_summary_version"]
    if ver == "v2_CoT":
        from .chat_summary import summary_v2

        return summary_v2
    from .chat_summary import summary_v1

    return summary_v1
