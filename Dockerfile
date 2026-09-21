FROM python:3.12-slim

# 安装必要证书与网络工具
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*

# 通过 pip 安装 uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# 配置环境变量
ENV UV_SYSTEM_PYTHON=1 \
    PYTHONUNBUFFERED=1

# 编译参数：默认不安装 CUDA / PyTorch，保持镜像轻量
ARG WITH_CUDA=false

# 拷贝依赖配置
COPY requirements.txt pyproject.toml ./

# 使用 uv 安装基础运行依赖
RUN uv pip install --system -r requirements.txt

# 特定编译开关：仅在 WITH_CUDA=true 时安装 CUDA PyTorch 与 Laya
RUN if [ "$WITH_CUDA" = "true" ]; then \
        echo "--> 启用 CUDA 编译开关，安装 PyTorch 与 Laya..."; \
        uv pip install --system torch --index-url https://download.pytorch.org/whl/cu124 && \
        uv pip install --system laya; \
    else \
        echo "--> 默认轻量模式（无 CUDA），跳过本地 PyTorch/Laya 安装"; \
    fi

# 拷贝应用代码
COPY . .

# 暴露内部服务端口
EXPOSE 8000

# 默认使用 uv run 启动服务
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
