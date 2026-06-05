#!/usr/bin/env python3
"""仅跑多轮对话测试（跳过 TextRAG/NutritionSQL/闲聊）"""
import asyncio
import time
import sys
sys.path.insert(0, '.')

from src.orchestrator.supervisor import RecipeOrchestrator
from src.api.schemas import AskRequest
from src.core.memory import ConversationMemory


async def main():
    orch = RecipeOrchestrator()
    print(f"图节点: {list(orch.graph.nodes.keys())}\n")

    sid = "e2e_test_multi"

    # 清理旧会话
    ConversationMemory.clear_all()

    # 第 1 轮：推荐菜
    print("=" * 60)
    print("【多轮测试】第 1 轮：推荐红烧肉")
    print("=" * 60)
    t0 = time.perf_counter()
    r1 = await orch.ask(AskRequest(query='推荐一道红烧肉', session_id=sid, stream=False))
    t1 = time.perf_counter()
    print(f"  Intent: {r1.intent}")
    print(f"  Answer: {r1.answer[:150]}...")
    print(f"  Wall: {t1-t0:.1f}s")
    assert r1.intent == "ingredient_recommend", f"❌ 意图异常: {r1.intent}"
    memory = ConversationMemory.get_or_create(sid)
    print(f"  记忆轮次: {len(memory.turns)}")
    assert len(memory.turns) == 1, "❌ 记忆未保存"
    print("  ✅ 第 1 轮通过\n")

    # 第 2 轮：追问（多轮上下文）
    print("=" * 60)
    print("【多轮测试】第 2 轮：换成猪肉")
    print("=" * 60)
    t0 = time.perf_counter()
    r2 = await orch.ask(AskRequest(query='那换成猪肉呢', session_id=sid, stream=False))
    t1 = time.perf_counter()
    print(f"  Intent: {r2.intent}")
    print(f"  Answer: {r2.answer[:150]}...")
    print(f"  Wall: {t1-t0:.1f}s")
    print(f"  记忆轮次: {len(memory.turns)}")
    assert len(memory.turns) == 2, "❌ 第二轮记忆未保存"
    assert r2.intent in ("substitution", "ingredient_recommend"), f"❌ 意图异常: {r2.intent}"
    # 验证多轮上下文
    context = memory.get_context(last_n=2)
    print(f"  上下文摘要 ({len(context)} chars): {context[:100]}...")
    assert "红烧肉" in context, "❌ 多轮上下文未包含上一轮信息"
    print("  ✅ 第 2 轮通过\n")

    # 第 3 轮：验证对话历史
    print("=" * 60)
    print("【多轮测试】第 3 轮：闲聊（验证记忆轮次）")
    print("=" * 60)
    t0 = time.perf_counter()
    r3 = await orch.ask(AskRequest(query='你好', session_id=sid, stream=False))
    t1 = time.perf_counter()
    print(f"  Intent: {r3.intent}")
    print(f"  Answer: {r3.answer[:150]}...")
    print(f"  Wall: {t1-t0:.1f}s")
    print(f"  记忆轮次: {len(memory.turns)}")
    assert len(memory.turns) == 3, "❌ 第三轮记忆未保存"
    print("  ✅ 第 3 轮通过\n")

    print("=" * 60)
    print("🎉 多轮对话测试全部通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
