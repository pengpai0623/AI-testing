from llmsdk.prompt_repo.base_template import BasePromptTemplate


# 根据不同场景创建子类，避免重复赋值
class CodeAnalyzeV1(BasePromptTemplate):
    version = "v1"
    scene = "代码推理、改错"
    template = """
# 角色：{role}
# 基础约束规则
{base_rules}
# 固定输出格式要求
{output_format}
# 硬性禁止项（固定内置，无需外部传入）
1. 禁止多余客套开场白、总结废话；
2. 只分析提供的代码，不脑补新增无关功能；
"""


# 全局实例，外部直接导入使用
code_analyze_v1 = CodeAnalyzeV1()

"""
    system_prompt = code_analyze_v1.render(
        role="专业Python代码审计与排错工程师",
        base_rules="精准定位代码语法错误、类型异常、逻辑漏洞，分析底层报错原理",
        output_format="输出分为两大模块：【分步推理过程】、【修复后完整代码】"
    )

    resp = llm.chat_single(
        prompt=f"请分析以下代码存在的问题：\n{bug_code}",
        system_prompt=system_prompt
    )
"""
