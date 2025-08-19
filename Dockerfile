# 使用 NVIDIA CUDA 12.1 作为基础镜像
ARG BASE_IMAGE="nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04"
FROM ${BASE_IMAGE}

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    UV_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

# 安装基本依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -sSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# 创建虚拟环境
ENV VIRTUAL_ENV=/app/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 复制项目文件
COPY pyproject.toml .env ./
COPY src ./src

# 使用 uv sync 安装依赖
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-packages \
    --index-url=$PIP_INDEX_URL \
    --trusted-host=$UV_TRUSTED_HOST \
   # 验证python
    && python -c "import src.voice2text.tran.app; print('Voice2Text app is ready!')"


# 暴露端口
EXPOSE 8765

# 设置启动命令
CMD ["python", "-m", "src.voice2text.tran.app"]

