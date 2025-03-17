# 使用 Python 3.11.11 作为基础镜像
FROM python:3.11.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -sSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# 复制 pyproject.toml 文件
COPY pyproject.toml README.md ./

# 复制源代码
COPY src ./src

# 使用 uv 安装依赖
RUN uv pip install -e .

# 如果有需要，安装开发依赖
# RUN uv pip install -e ".[dev]"

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露端口（如果您的应用需要）
EXPOSE 5000

# 设置启动命令
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app:app"]