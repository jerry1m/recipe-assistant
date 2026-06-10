"""
FastAPI 入口 — 参考 multi-agent-ecommerce-system python/main.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.schemas import AskRequest, AskResponse
from src.api.streaming import stream_response
from src.core.config import get_settings
from src.core.utils.logger import configure_logging
from src.core.utils.metrics import MetricsCollector
from src.orchestrator.supervisor import RecipeOrchestrator

logger = structlog.get_logger()
settings = get_settings()

metrics = MetricsCollector()
orchestrator = RecipeOrchestrator(metrics=metrics)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(debug=settings.debug)
    logger.info("app.startup", model=settings.llm_model)
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="Multi-Modal Recipe Assistant",
    description="多模态智能食谱助手 — Router + 4 Worker + Critic + Formatter",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件: 食谱图片
_images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images")
os.makedirs(_images_dir, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=_images_dir), name="recipe_images")


@app.get("/health")
async def health():
    """
    健康检查扩展 — 参考 travel-agent-guide 的 /api/v1/health 设计

    返回所有依赖组件的健康状态:
      - redis: 缓存/会话存储
      - vector_store: 向量索引文件
      - long_term_memory: 长效记忆存储
      - local_model: 本地 LLM 模型
      - llm_api: API 熔断器状态
    """
    import time
    from pathlib import Path

    checks = {}
    status = "ok"

    # ── Redis 检查 ──
    try:
        from src.core.memory import ConversationMemory
        r = ConversationMemory._get_redis()
        if r is not None:
            await r.ping()
            checks["redis"] = True
        else:
            checks["redis"] = False
            status = "degraded"
    except Exception:
        checks["redis"] = False
        status = "degraded"

    # ── 向量存储检查 ──
    try:
        vector_dir = Path(settings.vector_store_path)
        if vector_dir.exists():
            index_files = list(vector_dir.glob("*.index")) + list(vector_dir.glob("*.pkl"))
            checks["vector_store"] = len(index_files) > 0
            if not checks["vector_store"]:
                status = "degraded"
        else:
            checks["vector_store"] = False
            status = "degraded"
    except Exception:
        checks["vector_store"] = False
        status = "degraded"

    # ── 长效记忆检查 ──
    try:
        from src.core.long_term_memory import get_long_term_memory
        ltm = get_long_term_memory()
        stats = ltm.get_stats()
        checks["long_term_memory"] = True
    except Exception:
        checks["long_term_memory"] = False
        status = "degraded"

    # ── 本地模型检查（仅检查是否已加载） ──
    try:
        from src.core.utils.llm import is_local_model_loaded
        checks["local_model"] = is_local_model_loaded()
    except Exception:
        checks["local_model"] = False

    # ── API 熔断器状态 ──
    try:
        from src.core.utils.llm import get_api_breaker_stats
        breaker_stats = get_api_breaker_stats()
        checks["llm_api"] = breaker_stats["available"]
        if not breaker_stats["available"]:
            status = "degraded"
    except Exception:
        checks["llm_api"] = False
        status = "degraded"

    return {
        "status": status,
        "model": settings.llm_model,
        "version": "0.1.0",
        "checks": checks,
        "timestamp": time.time(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """统一问答入口"""
    return await orchestrator.ask(request)


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """SSE 流式问答入口"""
    return await stream_response(request, orchestrator)


@app.get("/metrics/agents")
async def agent_metrics():
    """Agent 运行指标"""
    return metrics.get_agent_stats()


@app.get("/metrics/business")
async def business_metrics():
    """业务指标"""
    return metrics.get_business_stats()


@app.get("/metrics/compression")
async def compression_metrics():
    """LLM 对话压缩指标"""
    from src.core.memory import ConversationMemory
    return ConversationMemory.get_compression_stats()


@app.get("/metrics/long_term_memory")
async def long_term_memory_metrics():
    """长效记忆指标"""
    from src.core.long_term_memory import get_long_term_memory
    return get_long_term_memory().get_stats()


@app.get("/metrics/circuit_breaker")
async def circuit_breaker_metrics():
    """熔断器指标"""
    from src.core.utils.llm import get_api_breaker_stats
    return get_api_breaker_stats()


# ── Frontend SPA (最后注册，避免覆盖 API 路由) ──
_frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
    logger.info("frontend_mounted", path=_frontend_dir)

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
