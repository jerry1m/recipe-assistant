"""
SSE 流式输出
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi.responses import StreamingResponse

from src.api.schemas import AskRequest, AskResponse, StreamingEvent
from src.orchestrator.supervisor import RecipeOrchestrator


async def event_stream(
    request: AskRequest,
    orchestrator: RecipeOrchestrator,
) -> AsyncGenerator[str, None]:
    """生成 SSE 事件流"""

    # 发送开始事件
    yield _format_event("start", {"query": request.query})

    # 执行完整流程
    response: AskResponse = await orchestrator.ask(request)

    # 按 token 流式发送（当前简单模拟）
    words = response.answer.split(" ")
    for i, word in enumerate(words):
        yield _format_event("token", {"type": "chunk", "text": word + " "})
        if i % 5 == 0:
            # 每 5 个词发送一次 provenance
            if response.provenance:
                yield _format_event("provenance", {
                    "items": [p.model_dump() for p in response.provenance[:3]]
                })

    # 发送完成事件
    yield _format_event("done", {
        "intent": response.intent,
        "confidence": response.confidence,
        "latency_ms": response.latency_ms,
        "tokens_used": response.tokens_used,
    })


def _format_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_response(
    request: AskRequest,
    orchestrator: RecipeOrchestrator,
) -> StreamingResponse:
    return StreamingResponse(
        event_stream(request, orchestrator),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
