from llmsdk.prompt_repo.base_template import BasePromptTemplate


class CodeAnalyzeCoTV2(BasePromptTemplate):
    version = "v2_CoT"
    scene = "代码Bug分析、逻辑改错（CoT分步推理增强）"
    template = """
# 角色：{role}
# 基础约束规则
{base_rules}
# CoT强制分步推理要求
{cot_rules}
# 固定输出格式要求
{output_format}
# 硬性禁止项（固定内置，无需外部传入）
1. 禁止多余客套开场白、总结废话；
2. 不得跳过推理步骤直接给出修复代码；
3. 只分析提供的代码，不脑补新增无关功能；
"""


# 全局单例
code_analyze_cot_v2 = CodeAnalyzeCoTV2()

"""
    system_prompt = code_analyze_cot_v2.render(
        role="专业Python代码审计与排错工程师",
        base_rules="精准定位代码语法错误、类型异常、逻辑漏洞，分析底层报错原理",
        cot_rules='''
            1. 第一步：梳理代码整体功能、入参定义、完整执行流程；
            2. 第二步：逐行解析变量数据类型、运算逻辑；
            3. 第三步：精准定位报错代码行，解释异常产生原因；
            4. 第四步：输出完整修复后的可运行代码；
            ''',
        output_format="输出分为两大模块：【分步推理过程】、【修复后完整代码】"
    )

    resp = llm.chat_single(
        prompt=f"请分析以下代码存在的问题：\n{bug_code}",
        system_prompt=system_prompt
    )
"""
