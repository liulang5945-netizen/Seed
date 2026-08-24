# Seed — 容器化部署（API 服务 + 静态前端）
#
# 说明：
#   - Seed 的主形态是 PyQt6 桌面应用（desktop/），Docker 镜像面向"服务端 / 无头"
#     部署场景：只运行 FastAPI 后端 + 构建好的前端静态资源。
#   - 模型检查点（checkpoints/*.pt）体积较大且不入库，需通过 volume 挂载提供，
#     否则后端以无模型的降级模式启动。
#
# 构建 & 运行：
#   docker build -t seed:latest .
#   docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints seed:latest
# 或使用 docker-compose：
#   docker compose up --build

# ---------- Stage 1: 前端构建 ----------
FROM node:22-slim AS frontend-builder
WORKDIR /app/frontend

# 先拷贝依赖清单以利用层缓存
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# 拷贝源码并构建
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python 运行时 ----------
FROM python:3.12-slim AS runtime

# 避免交互式安装 / 生成 .pyc，日志直出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖：curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 先装 torch（CPU wheel）以利用层缓存
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 拷贝打包元数据并安装项目（legacy extras 提供 FastAPI/uvicorn 等 API 依赖）
COPY pyproject.toml README.md ./
COPY seed/ ./seed/
COPY taiji/ ./taiji/
COPY neuroplex/ ./neuroplex/
COPY api/ ./api/
RUN pip install -e ".[legacy]"

# 拷贝前端构建产物（后端以此为静态资源）
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# 训练脚本与数据目录（部分 API 路由会引用）
COPY scripts/ ./scripts/
COPY data/ ./data/

# 运行用户与端口
EXPOSE 8000

# 健康检查：/api/health 由 routes_chat.py 提供
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# 以 0.0.0.0 暴露（桌面模式默认绑定 127.0.0.1，容器内必须用 0.0.0.0）
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
