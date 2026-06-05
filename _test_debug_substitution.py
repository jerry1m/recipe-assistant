#!/usr/bin/env python3
"""单轮测试：Substitution 意图 + Critic 重试循环"""
import asyncio
import time
import sys
sys.path.insert(0, '.')

from src.orchestrator.supervisor import RecipeOrchestrator
from src.api.schemas import AskRequest


async def main():
    orch = RecipeOrchestrator()
    print(f"图节点: {list(orch.graph.nodes.keys())}\n")

    print("=" * 60)
    print("【Debug】替换测试：那换成猪肉呢")
    print("=" * 60)
    t0 = time.perf_counter()
    resp = await orch.ask(AskRequest(query='那换成猪肉呢', session_id='debug_sub', stream=False))
    t1 = time.perf_counter()
    print(f"  Intent: {resp.intent}")
    print(f"  Answer: {resp.answer[:200]}...")
    print(f"  Wall: {t1-t0:.1f}s")
    print()
    if resp.answer:
        print("  ✅ 测试通过")
    else:
        print("  ❌ Answer 为空")


if __name__ == "__main__":
    asyncio.run(main())
