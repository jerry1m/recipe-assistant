#!/usr/bin/env python3
"""端到端测试：单轮 + 多轮 + Critic 重试"""
import asyncio
import time
import sys
sys.path.insert(0, '.')

from src.orchestrator.supervisor import RecipeOrchestrator
from src.api.schemas import AskRequest
from src.core.memory import ConversationMemory


async def test_single_turn(orch: RecipeOrchestrator):
    """测试 1: 单轮 TextRAG"""
    print("=" * 60)
    print("【测试 1/4】单轮 TextRAG")
    print("=" * 60)
    t0 = time.perf_counter()
    resp = await orch.ask(AskRequest(query='how to make a cake', stream=False))
    t1 = time.perf_counter()
    print(f"  Intent: {resp.intent}")
    print(f"  Answer: {resp.answer[:150]}...")
    print(f"  Latency: {resp.latency_ms:.0f}ms | Wall: {t1-t0:.1f}s")
    assert resp.answer, "❌ Answer 为空"
    assert resp.intent in ("ingredient_recommend", "step_qa"), f"❌ 意图异常: {resp.intent}"
    assert resp.latency_ms > 0, "❌ Latency 异常"
    print("  ✅ 通过\n")


async def test_nutrition_sql(orch: RecipeOrchestrator):
    """测试 2: 单轮 NutritionSQL + Critic 重试"""
    print("=" * 60)
    print("【测试 2/4】NutritionSQL + Critic 重试")
    print("=" * 60)
    t0 = time.perf_counter()
    resp = await orch.ask(AskRequest(query='how many calories in chicken', stream=False))
    t1 = time.perf_counter()
    print(f"  Intent: {resp.intent}")
    print(f"  Answer: {resp.answer[:150]}...")
    print(f"  Latency: {resp.latency_ms:.0f}ms | Wall: {t1-t0:.1f}s")
    assert resp.intent == "nutrition_filter", f"❌ 意图异常: {resp.intent}"
    print("  ✅ 通过\n")


async def test_chitchat(orch: RecipeOrchestrator):
    """测试 3: 闲聊直出（跳过 Critic 和 Formatter）"""
    print("=" * 60)
    print("【测试 3/4】闲聊直出")
    print("=" * 60)
    t0 = time.perf_counter()
    resp = await orch.ask(AskRequest(query='hello', stream=False))
    t1 = time.perf_counter()
    print(f"  Intent: {resp.intent}")
    print(f"  Answer: {resp.answer[:150]}...")
    print(f"  Latency: {resp.latency_ms:.0f}ms | Wall: {t1-t0:.1f}s")
    assert resp.intent == "chitchat", f"❌ 意图异常: {resp.intent}"
    assert "食谱助手" in resp.answer, "❌ 闲聊回复异常"
    print("  ✅ 通过\n")


async def test_multi_turn(orch: RecipeOrchestrator):
    """测试 4: 多轮对话"""
    print("=" * 60)
    print("【测试 4/4】多轮对话")
    print("=" * 60)

    sid = "e2e_test_session"

    # 清理旧会话
    ConversationMemory.clear_all()

    # 第 1 轮：推荐菜
    print("  --- 第 1 轮：推荐红烧肉 ---")
    t0 = time.perf_counter()
    r1 = await orch.ask(AskRequest(query='推荐一道红烧肉', session_id=sid, stream=False))
    t1 = time.perf_counter()
    print(f"    Intent: {r1.intent}")
    print(f"    Answer: {r1.answer[:120]}...")
    print(f"    Wall: {t1-t0:.1f}s")
    assert r1.intent == "ingredient_recommend", f"❌ 意图异常: {r1.intent}"
    memory = ConversationMemory.get_or_create(sid)
    print(f"    记忆轮次: {len(memory.turns)}")
    assert len(memory.turns) == 1, "❌ 记忆未保存"

    # 第 2 轮：追问（多轮上下文）
    print("  --- 第 2 轮：换成猪肉 ---")
    t0 = time.perf_counter()
    r2 = await orch.ask(AskRequest(query='那换成猪肉呢', session_id=sid, stream=False))
    t1 = time.perf_counter()
    print(f"    Intent: {r2.intent}")
    print(f"    Answer: {r2.answer[:120]}...")
    print(f"    Wall: {t1-t0:.1f}s")
    print(f"    记忆轮次: {len(memory.turns)}")
    assert len(memory.turns) == 2, "❌ 第二轮记忆未保存"
    # 这里的意图应为 substitution（上一轮是菜，这轮问换食材）
    # 或者 ingredient_recommend（也是合理）
    assert r2.intent in ("substitution", "ingredient_recommend"), f"❌ 意图异常: {r2.intent}"

    # 验证多轮上下文
    context = memory.get_context(last_n=2)
    print(f"    上下文摘要 ({len(context)} chars): {context[:100]}...")
    assert "红烧肉" in context, "❌ 多轮上下文未包含上一轮信息"

    print("  ✅ 多轮对话通过\n")


async def main():
    orch = RecipeOrchestrator()
    print(f"图节点: {list(orch.graph.nodes.keys())}\n")

    await test_single_turn(orch)
    await test_nutrition_sql(orch)
    await test_chitchat(orch)
    await test_multi_turn(orch)

    print("=" * 60)
    print("🎉 全部端到端测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
