# -------- 构建阶段：安装依赖 --------
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装编译必要系统库
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 把依赖安装到单独输出目录，后续复制到运行镜像
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# -------- 运行阶段：最终镜像 --------
FROM python:3.11-slim

WORKDIR /app

# 创建普通用户，不使用root运行服务（安全最佳实践）
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 从builder复制已经安装好的python包
COPY --from=builder /install /usr/local

# 拷贝业务源码（排除tests，由.dockerignore控制）
COPY . .

# 修改目录权限
RUN chown -R appuser:appuser /app
USER appuser

# 暴露端口，仅文档提示，不会自动映射
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
