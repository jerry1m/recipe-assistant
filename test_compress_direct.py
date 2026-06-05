"""
直接测试 LLM 对话压缩（方案 A）—— 不经过完整 API 管道

流程：
  1. 直接构造 ConversationMemory，注入多轮对话
  2. 手动调用 compress_if_needed()，记录压缩前后 token
  3. 输出压缩指标
"""

import asyncio
import json
import time
import sys

sys.path.insert(0, ".")

from src.core.memory import ConversationMemory
from src.api.schemas import TurnRecord


async def main():
    # 清理已有数据
    mem = await ConversationMemory.get_or_create("test_direct_compress")
    await mem.clear()

    # 手动注入多轮对话（模拟长对话）
    queries = [
        "I want to cook Braised Pork Belly, any recommendations?",
        "What ingredients do I need for this dish?",
        "Can you give me the step-by-step instructions?",
        "I want to reduce sugar, what can I substitute it with?",
        "How many calories does this dish have?",
        "Suggest a light and healthy dish instead.",
        "Recommend a low-fat soup",
        "How long should I simmer this soup?",
        "What vegetables go well with this soup?",
    ]
    answers = [
        "Here are some pork belly recipes: 1. Braised Pork Belly (Hong Shao Rou) - classic Chinese red braised pork belly with soy sauce and caramelized sugar. 2. Crispy Pork Belly - oven roasted with crackling skin. 3. Pork Belly Stir Fry - sliced thin with vegetables.",
        "You'll need: 500g pork belly, 2 tbsp light soy sauce, 1 tbsp dark soy sauce, 30g rock sugar, 2 star anise, 1 cinnamon stick, 3 slices ginger, 2 spring onions, 500ml water.",
        "Step 1: Blanch pork belly in boiling water for 3 mins. Step 2: Cut into 3cm cubes. Step 3: Caramelize sugar in oil until golden. Step 4: Add pork and coat with sugar. Step 5: Add soy sauces, spices, and water. Step 6: Simmer for 1.5 hours until tender.",
        "You can substitute rock sugar with: 1. Monk fruit sweetener (0 calories) 2. Erythritol (70% sweetness) 3. Stevia (200x sweeter, use less). For savory dishes like braised pork, monk fruit works best as it has no bitter aftertaste.",
        "A serving of Braised Pork Belly (150g) contains approximately: 380 calories, 28g fat, 12g protein, 15g carbs. It's relatively high in calories due to the fatty cut of meat.",
        "Try Steamed Fish with ginger and spring onions - only 180 calories per serving, high protein and very light. Or a simple Stir-fried Bok Choy with garlic - 50 calories per serving.",
        "A great low-fat soup is Miso Soup with tofu and seaweed - only 40 calories per bowl. Also try Hot and Sour Soup - 60 calories, or Chicken Vegetable Soup - 80 calories.",
        "For most soups, simmer on low heat for 20-30 minutes after bringing to a boil. For bone-based soups, simmer 1-2 hours. Don't boil aggressively as it makes the soup cloudy.",
        "For Miso Soup: seaweed, tofu, green onions, and mushrooms. For Chicken Soup: carrots, celery, onions, and leeks. For Vegetable Soup: zucchini, tomatoes, spinach, and bell peppers.",
    ]

    turns = []
    for i, (q, a) in enumerate(zip(queries, answers)):
        turn = TurnRecord(
            query=q,
            intent="ingredient_recommend",
            slots={},
            answer=a,
            provenance=[],
            timestamp=time.time(),
        )
        turns.append(turn)

    # 直接写入 Redis
    r = ConversationMemory._get_redis()
    assert r is not None, "Redis must be available"
    
    # 清空已有的 turns
    await r.delete(mem._turns_key, mem._summary_key, mem._meta_key)
    
    # 写入所有轮次
    for turn in turns:
        await r.rpush(mem._turns_key, turn.model_dump_json())
    
    # 设置 TTL
    await r.expire(mem._turns_key, 86400)
    
    count = await r.llen(mem._turns_key)
    print(f"✅ 已写入 {count} 轮对话到 Redis")
    
    # 设置较低的 max_tokens 强制触发压缩
    await mem._store_summary("")  # 清空已有摘要
    await mem._store_compressed_count(0)
    
    # 调用压缩
    print("\n🔄 调用 compress_if_needed(max_tokens=800) ...")
    result = await mem.compress_if_needed(max_tokens=800)
    print(f"   结果: {json.dumps(result, ensure_ascii=False, default=str)}")
    
    # 获取上下文验证
    ctx = await mem.get_context(max_tokens=800)
    print(f"\n📝 压缩后的上下文 ({_estimate_tokens(ctx)} tokens):")
    print(f"   {ctx[:200]}...")
    
    # 再次压缩（应检测到已压缩到最新）
    print("\n🔄 再次压缩 (compress_if_needed)...")
    result2 = await mem.compress_if_needed(max_tokens=800)
    print(f"   结果: {json.dumps(result2, ensure_ascii=False, default=str)}")
    
    # 查看全局统计
    print("\n📊 全局压缩统计:")
    stats = ConversationMemory.get_compression_stats()
    print(f"   {json.dumps(stats, ensure_ascii=False, indent=2)}")
    
    # 计算比例
    if stats["total_compression_calls"] > 0:
        ratio = stats["overall_compression_ratio"]
        saved = stats["tokens_saved"]
        orig = stats["total_original_tokens"]
        comp = stats["total_compressed_tokens"]
        print(f"\n{'='*50}")
        print(f"📈 压缩效果总结:")
        print(f"   压缩前: {orig} tokens")
        print(f"   压缩后: {comp} tokens")
        print(f"   压缩率: {ratio} ({ratio*100:.1f}%)")
        print(f"   节省:   {saved} tokens ({saved/max(1,orig)*100:.1f}%)")
        print(f"   调用次数: {stats['total_compression_calls']}")
        print(f"   压缩轮次: {stats['total_turns_compressed']}")
        print(f"{'='*50}")
    else:
        print("\n⚠️  未触发压缩")
        # 检查是否因为压缩预算不足
        turns_loaded = await mem._load_turns()
        print(f"   当前轮次数: {len(turns_loaded)}")
        summary = await mem._load_summary()
        print(f"   当前摘要: '{summary[:80] if summary else '(空)'}'")

    await r.aclose()


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)


if __name__ == "__main__":
    asyncio.run(main())
