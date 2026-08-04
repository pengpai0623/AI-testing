llmsdk/                     # SDK根包名
├── __init__.py             # 包入口，统一对外暴露API
├── client/                 # LLM底层请求层
│   ├── __init__.py
│   └── base_llm.py         # LLMBaseClient 完整代码
├── session/                # 会话上下文管理层
│   ├── __init__.py
│   └── chat_session.py     # ChatSession 完整代码
├── utils/                  # 工具、常量、通用函数
│   ├── __init__.py
│   ├── constants.py        # 全局常量（重试次数、默认超时、token系数）
│   └── exceptions.py       # 自定义异常（替代裸Exception，方便捕获）
└── config/                 # 配置读取、环境变量加载
    ├── __init__.py
    └── settings.py         # 统一读取.env，集中管理所有配置

# 配套外层文件（用于打包、安装）
.env                        # 环境变量（放项目根目录，不打进包内）
requirements.txt            # 依赖清单
main_demo.py                # 整体调用演示入口（用户测试用，不属于SDK包）
README.md                   # 使用文档
