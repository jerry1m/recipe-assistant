"""
Recipe Supervisor 编排器 — LangGraph 包装层

   [start] → load_memory → router (with context) → worker (by intent)
                                                    ↓ (并行: worker + fallback)
           → critic → {passed: save_memory → formatter → [end]}
                      {failed & retry<max: revision → worker (retry with feedback)}
                      {failed & retry>=max: save_memory → formatter → [end]}

多轮对话: ConversationMemory 按 session_id 存储历史，Router 带上文
重试循环: Critic 不通过时，送反馈回 Worker 重做（最多 2 次），仍失败则走兜底
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from src.api.schemas import AskRequest, AskResponse
from src.core.utils.metrics import MetricsCollector
from src.orchestrator.graph import build_recipe_graph

logger = structlog.get_logger()


class RecipeOrchestrator:
    """编排多 Agent 协同工作 — 包装 LangGraph（多轮对话 + 重试循环）"""

    def __init__(self, metrics: MetricsCollector | None = None):
        self.graph = build_recipe_graph()
        self.metrics = metrics or MetricsCollector()

    async def ask(self, request: AskRequest) -> AskResponse:
        start = time.perf_counter()
        session_id = request.session_id or str(uuid.uuid4())

        logger.info(
            "orchestrator.start",
            query=request.query[:50],
            session_id=session_id,
            has_images=len(request.images) > 0,
        )

        # ── 执行 LangGraph 状态图 ──
        initial_state: dict[str, Any] = {
            "query": request.query,
            "session_id": session_id,
            "images": request.images,
            "files": request.files,
            "intent_hint": request.intent_hint,
            "stream": request.stream,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        final_state = await self.graph.ainvoke(initial_state)

        total_latency = (time.perf_counter() - start) * 1000

        logger.info(
            "orchestrator.complete",
            session_id=session_id,
            intent=final_state.get("intent", ""),
            retry_count=final_state.get("retry_count", 0),
            total_latency_ms=round(total_latency, 1),
        )

        return AskResponse(
            answer=final_state.get("final_response", ""),
            provenance=final_state.get("provenance", []),
            intent=final_state.get("intent", ""),
            confidence=final_state.get("router_confidence", 0.0),
            latency_ms=round(total_latency, 1),
            tokens_used=0,
            disclaimer=None,
        )
