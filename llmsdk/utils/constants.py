# 重试配置
MAX_RETRY_TIMES = 3
RETRY_WAIT_SEC = 2

# Token估算配置
CHINESE_TOKEN_RATIO = 1.5
DEFAULT_MAX_TOKEN = 6000
MIN_KEEP_MSG_NUM = 3
CUT_PAIR_PER_TIME = 1

# 请求默认参数
DEFAULT_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0.7

# 结构化专属默认参数
STRUCT_DEFAULT_TEMP = 0.2
STRUCT_DEFAULT_SYSTEM_PROMPT = (
    "你是信息抽取助手。严格遵守规则："
    "1. 只返回JSON字符串，绝对不要增加任何解释、前言、后语、换行说明；"
    "2. 不要用markdown代码块包裹；"
    "3. 必须包含name、price、tags三个字段；"
    "4. price必须是纯数字，不能带元、¥等符号；"
    "5. tags是字符串列表。用户输入商品文案，仅输出合法JSON。"
)
