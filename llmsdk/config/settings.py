import os

from dotenv import load_dotenv

from llmsdk.utils.exceptions import EnvConfigError

# 加载项目根目录.env
load_dotenv()

# 大模型接口配置
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "")
DOUBAO_ENDPOINT = os.getenv("DOUBAO_ENDPOINT", "")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "")

# 请求超时
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 60))


# 配置校验
def check_config():
    missing = []
    if not DOUBAO_API_KEY:
        missing.append("DOUBAO_API_KEY")
    if not DOUBAO_ENDPOINT:
        missing.append("DOUBAO_ENDPOINT")
    if not DOUBAO_MODEL:
        missing.append("DOUBAO_MODEL")
    if missing:
        raise EnvConfigError(f".env缺失必填配置：{','.join(missing)}")


# 初始化自动校验
check_config()
