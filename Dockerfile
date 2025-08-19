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

# 安装基本依赖和 Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 pipx 并使用它安装 uv
RUN python3 -m pip install --user pipx \
    && python3 -m pipx ensurepath \
    && export PATH="$HOME/.local/bin:$PATH" \
    && pipx install uv

# 将 pipx 安装的工具路径添加到 PATH
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# 创建虚拟环境
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 复制项目文件
COPY pyproject.toml README.md uv.lock ./
COPY src ./src

# 检查 uv 是否安装成功
RUN which uv && uv --version

# 使用 uv sync 安装依赖
RUN uv sync --index-url=$PIP_INDEX_URL

# 暴露端口
EXPOSE 8765

# 设置启动命令
CMD ["python", "-m", "src.voice2text.tran.app"]