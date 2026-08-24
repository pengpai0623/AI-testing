import logging
import sys
from typing import Optional


def setup_logger():
    app_log = logging.getLogger("llm_sdk")
    # 防止重复添加handler（多次import不会重复打印）
    if app_log.handlers:
        return app_log

    app_log.setLevel(logging.INFO)
    app_log.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    app_log.addHandler(console_handler)

    # 如果需要文件日志，可以追加FileHandler
    # from logging.handlers import RotatingFileHandler
    # file_handler = RotatingFileHandler("app.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf‑8")
    # file_handler.setFormatter(formatter)
    # app_log.addHandler(file_handler)

    return app_log


logger = setup_logger()
