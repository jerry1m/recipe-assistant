"""
快速启动 Demo — agent_trace_demo.py
"""

from __future__ import annotations

import asyncio
import json

from src.api.schemas import AskRequest
from src.core.utils.logger import configure_logging
from src.orchestrator.supervisor import RecipeOrchestrator


async def main():
    configure_logging(debug=True)
    orchestrator = RecipeOrchestrator()

    test_queries = [
        "推荐一道红烧肉的菜谱",
        "鸡肉的热量是多少？",
        "没有鸡蛋可以用什么代替？",
        "你好",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")

        request = AskRequest(query=query, stream=False)
        response = await orchestrator.ask(request)

        print(f"📋 Intent: {response.intent} (confidence: {response.confidence:.2f})")
        print(f"💬 Answer: {response.answer}")
        print(f"📎 Provenance: {len(response.provenance)} items")
        print(f"⏱ Latency: {response.latency_ms}ms")

        if response.disclaimer:
            print(f"⚠️ Disclaimer: {response.disclaimer}")

    # 打印 Agent 统计
    print(f"\n{'='*60}")
    print("📊 Agent Stats:")
    print(json.dumps(orchestrator.metrics.get_agent_stats(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
