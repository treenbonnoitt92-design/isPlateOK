#!/usr/bin/env bash
set -e

# ==========================================================
# JevLaya 一键容器化构建与部署脚本
# 默认映射外部端口: 13580 -> 内部 8000
# 镜像名称: jevlaya:latest
# ==========================================================

IMAGE_NAME="jevlaya:latest"
PORT="13580"
WITH_CUDA="false"

# 检查参数
for arg in "$@"; do
  case $arg in
    --cuda|--with-cuda)
      WITH_CUDA="true"
      shift
      ;;
    --help|-h)
      echo "用法: ./deploy.sh [选项]"
      echo "选项:"
      echo "  --cuda, --with-cuda   启用 CUDA/GPU 编译开关构建镜像 (安装 PyTorch 及 Laya)"
      echo "  -h, --help            显示帮助信息"
      exit 0
      ;;
  esac
done

echo "=========================================================="
echo "🚀 开始部署 JevLaya 车牌智能决策系统"
echo "   - 目标镜像: ${IMAGE_NAME}"
echo "   - 对外端口: ${PORT}"
echo "   - CUDA 支持: ${WITH_CUDA}"
echo "=========================================================="

# 1. 检查必要环境
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未检测到 docker 命令，请先安装 Docker！"
    exit 1
fi

COMPOSE_CMD=""
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ 错误: 未检测到 docker compose 或 docker-compose！"
    exit 1
fi

# 2. 检查 .env 配置文件
if [ ! -f ".env" ]; then
    echo "⚠️ 警告: 未找到 .env 配置文件，正在创建模板..."
    cat << 'EOF' > .env
OPENROUTER_API_KEY=
OPENROUTER_API_MODEL="~typesafe/jev-latest"
EOF
    echo "❗ 请在 .env 中填入有效的 OPENROUTER_API_KEY 后重新运行 ./deploy.sh"
    exit 1
fi

# 3. 预构建 Docker 镜像（docker-compose 仅引用镜像，不负责 build）
echo "📦 正在构建 Docker 镜像 (${IMAGE_NAME})..."
if [ "${WITH_CUDA}" = "true" ]; then
    echo "⚡ [编译开关] 启用 CUDA 支持构建镜像..."
    docker build --build-arg WITH_CUDA=true -t "${IMAGE_NAME}" .
else
    echo "🌱 [编译开关] 采用默认轻量模式构建（无 CUDA，仅通过 uv run 运行 Jev 决策）..."
    docker build --build-arg WITH_CUDA=false -t "${IMAGE_NAME}" .
fi

# 4. 启动容器服务
echo "🚀 启动容器服务 (端口映射 13580:8000)..."
${COMPOSE_CMD} down || true
${COMPOSE_CMD} up -d

# 5. 输出状态与访问提示
echo "=========================================================="
echo "✅ 部署完成！"
echo "   - 容器运行状态:"
${COMPOSE_CMD} ps
echo ""
echo "🌐 访问地址: http://localhost:${PORT}"
echo "📜 查看实时日志: ${COMPOSE_CMD} logs -f"
echo "🛑 停止服务: ${COMPOSE_CMD} down"
echo "=========================================================="
