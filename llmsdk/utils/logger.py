# logger.py
import logging
import sys

# 1. 获取根记录器，清空所有已有的 handlers（防止重复或默认行为）
root_logger = logging.getLogger()
root_logger.handlers.clear()

# 2. 设置日志级别为 INFO（DEBUG 及以下将被忽略）
root_logger.setLevel(logging.INFO)

# 3. 创建一个控制台处理器（输出到终端）
console_handler = logging.StreamHandler(sys.stdout)  # 或 sys.stderr，通常用 stdout
console_handler.setLevel(logging.INFO)  # 与 root 一致

# 4. 设置日志格式（包含时间、级别、模块名、消息）
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)

# 5. 将处理器添加到根记录器
root_logger.addHandler(console_handler)

# 6. 导出 logger 对象供全局使用
logger = root_logger
