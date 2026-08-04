# logger.py
import logging
import sys

root_logger = logging.getLogger()
root_logger.handlers.clear()

root_logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)  # 或 sys.stderr，通常用 stdout
console_handler.setLevel(logging.INFO)  # 与 root 一致

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)

root_logger.addHandler(console_handler)

logger = root_logger
