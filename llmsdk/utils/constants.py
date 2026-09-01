# 重试配置
MAX_RETRY_TIMES = 3
RETRY_WAIT_SEC = 2

# Token估算配置
CHINESE_TOKEN_RATIO = 1.5
DEFAULT_MAX_TOKEN = 6000
MIN_KEEP_MSG_NUM = 3
CUT_PAIR_PER_TIME = 1
MAX_COMPLETION_TOKEN = 1000

# 请求默认参数
DEFAULT_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0.7

# prompt本地默认值
PROMPT_CONFIG = {
    "product_extract_version": "v2_fewshot",
    "code_analyze_version": "v2_CoT",
    "chat_summary_version": "v2_CoT",
}

# 结构化专属默认参数
STRUCT_DEFAULT_TEMP = 0.2
STRUCT_DEFAULT_SYSTEM_PROMPT = """
# 角色：商品信息抽取专家
# 分步推理规则
1. 第一步：从原文筛选所有商品、价格、标签相关文字；
2. 第二步：过滤无关修饰词，提取纯数值价格；
3. 第三步：整理分类标签列表；
推理完成后，仅输出纯净JSON，无多余文字、解释、markdown代码块，key为name/price/tags，无多余文字。
# 参考标准样例（严格模仿此格式输出）
示例1
输入：无线蓝牙耳机269元，标签数码、耳机
输出：{"name":"无线蓝牙耳机","price":269,"tags":["数码","耳机"]}

示例2
输入：运动手环99，适用健身、睡眠监测
输出：{"name":"运动手环","price":99,"tags":["健身","睡眠监测"]}
# 按上面样例格式处理用户输入
"""

# Web接口统一返回码（ApiResponse 的 code）
# 0/400/422/500：web 层 http 业务码
CODE_OK = 0
CODE_PARAM_ERROR = 400
CODE_VALIDATE_ERROR = 422
CODE_SERVER_ERROR = 500

# SDK业务异常码 LLMBaseError子类
# 1xxx：大模型底层链路类错误（网络、http、sse、配置）
ERR_ENV_CONFIG = 1001  # EnvConfigError 配置/环境变量错误
ERR_LLM_NETWORK = 1002  # LLMNetworkError 网络超时连接失败
ERR_LLM_HTTP = 1003  # LLMHttpError 上游4xx/5xx
ERR_SSE_PARSE = 1004  # LLMSSEParseError SSE分片解析失败

# 4xxx：结构化输出解析类错误（json 解析、pydantic 校验、参数数值）
ERR_JSON_PARSE = 4001  # JsonParseError json解析失败
ERR_PYDANTIC_VALIDATE = 4002  # PydanticValidateError 字段校验失败
ERR_VALUE = 4003  # LLMValueError 数值非法
ERR_CONNECT = 4004  # LLMConnectionError 连接业务异常
ERR_MSG_VALIDATE = 4005  # MessageValidateError 消息列表校验失败
ERR_CLIENT_DISCONNECT = 4006  # ClientDisconnectError 客户端主动断开SSE流式连接，属于正常结束，不是服务故障

# 5xxx：Redis会话管理类错误
ERR_REDIS_CONNECTION = 5001  # RedisConnectionError Redis连接异常
ERR_REDIS_TIMEOUT = 5002  # RedisTimeoutError Redis请求超时
ERR_REDIS_ERROR = 5003  # RedisError Redis操作异常

# redis 会话管理
REDIS_URL = "redis://127.0.0.1:6379/0"
SESSION_TTL_SEC = 3600 * 24 * 1  # 会话过期时间，单位秒，默认7天
