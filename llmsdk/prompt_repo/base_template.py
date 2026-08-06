class BasePromptTemplate:
    version: str  # 版本号 v1/v2
    scene: str  # 所属业务场景
    template: str  # 原始模板字符串

    def render(self, **kwargs) -> str:
        """动态填充占位符，返回完整prompt文本"""
        return self.template.format(**kwargs)
