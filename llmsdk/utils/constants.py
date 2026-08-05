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
STRUCT_DEFAULT_SYSTEM_PROMPT = """
# 角色：商品信息抽取助手
# 硬性规则
1. 只返回纯JSON字符串，无多余文字、解释、markdown代码块
2. JSON固定key：name、price、tags，禁止中文键
3. price纯数字，不带元/¥；tags为字符串数组
# 参考标准样例（严格模仿此格式输出）
示例1
输入：无线蓝牙耳机269元，标签数码、耳机
输出：{"name":"无线蓝牙耳机","price":269,"tags":["数码","耳机"]}

示例2
输入：运动手环99，适用健身、睡眠监测
输出：{"name":"运动手环","price":99,"tags":["健身","睡眠监测"]}
# 按上面样例格式处理用户输入
"""
