"""
综合基准测试 — 为项目描述提供量化指标

测试维度:
  1. 检索架构 (HybridRetriever): 延迟 P50/P95/P99 + 预热效果
  2. 意图识别 (RouterAgent): 多类别准确率
  3. Text-to-SQL 安全: 攻击拦截率 + 正常查询通过率
  4. CLIP 多模态: 文本→菜名检索质量 + 延迟
  5. 上下文压缩: 压缩率 + 质量
  6. 端到端流程 (Orchestrator): 完整流程延迟

用法:
  CUDA_VISIBLE_DEVICES=1 python -m src.eval.run_benchmark
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ── 1. 检索架构测试 ──

async def bench_retrieval() -> dict[str, Any]:
    """测试 HybridRetriever 延迟分布 + 预热效果"""
    from src.core.retrievers.hybrid import HybridRetriever

    retriever = HybridRetriever()

    # ===== 预热 =====
    warmup_queries = ["红烧肉的做法", "番茄炒蛋", "麻婆豆腐", "鱼香肉丝", "宫保鸡丁"]
    print("\n=== 预热检索器 (5轮) ===")
    for q in warmup_queries:
        t0 = time.perf_counter()
        await retriever.retrieve(q)
        print(f"  预热 {q[:10]}: {(time.perf_counter()-t0)*1000:.1f}ms")

    # ===== 正式测试 =====
    # 注意：所有食谱数据是英文的（recipes_real.json），
    # 中文查询因中-英跨语言差异，BM25+FAISS 分数普遍偏低（常为 0.000），
    # 这属于客观存在的跨语言检索挑战，并非系统缺陷。
    # 英文查询能充分测试语义检索能力，中文查询反映实际跨语言场景。
    test_queries = [
        # ── 英文查询（与数据语言匹配，测试完整检索链路）──
        "Roasted chicken with vegetables",
        "Chocolate cake recipe",
        "Pasta with tomato sauce",
        "beef stew recipe",
        "healthy breakfast ideas",
        "how to make bread",
        "chicken soup recipe",
        "vegetable salad",
        "quick dinner recipes",
        "gluten free dessert",
        "Italian pasta dishes",
        "summer BBQ ideas",
        "low carb meals",
        # ── 中文查询（跨语言场景，测试实际用户画像）──
        "红烧肉的做法",
        "番茄炒蛋的热量是多少",
        "没有鸡蛋可以用什么代替",
        "麻婆豆腐怎么做",
        "推荐一道低热量的素菜",
        "鸡肉的蛋白质含量高吗",
        "鱼香肉丝没有豆瓣酱可以用什么",
        "蛋白质含量最高的菜有哪些",
        "低卡晚餐推荐",
        "素食主义者能吃什么",
        "糖醋排骨的做法步骤",
        "清蒸鲈鱼需要什么材料",
        "牛肉炖土豆怎么做",
    ]

    latencies = []
    print("\n=== 正式检索测试 (26 queries) ===")
    for i, q in enumerate(test_queries):
        t0 = time.perf_counter()
        chunks = await retriever.retrieve(q)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        top_scores = [f"{c.score:.3f}" for c in chunks[:3]]
        print(f"  [{i+1:02d}] {q[:20]:20s} → {elapsed:8.1f}ms  top3={top_scores}")

    latencies.sort()
    n = len(latencies)
    avg = sum(latencies) / n
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[int(n * 0.99)]
    cold = sum(latencies[:3]) / 3        # 前3次（含冷启动）
    warm = sum(latencies[-5:]) / 5       # 后5次（热缓存）
    first = latencies[0]
    last = latencies[-1]

    return {
        "retrieval": {
            "total_queries": n,
            "avg_latency_ms": round(avg, 1),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
            "warm_latency_ms": round(warm, 1),
            "cold_start_ms": round(cold, 1),
            "min_ms": round(first, 1),
            "max_ms": round(last, 1),
            "bm25_weight": 0.3,
            "vector_weight": 0.5,
            "rerank_weight": 0.2,
        }
    }


# ── 2. 意图识别测试 ──

async def bench_intent() -> dict[str, Any]:
    """测试 RouterAgent 多类别意图识别准确率"""
    from src.agents.router import RouterAgent

    agent = RouterAgent()

    test_cases = [
        # ingredient_recommend (推荐)
        ("推荐一道红烧肉的菜谱", "ingredient_recommend"),
        ("推荐一道低热量的素菜", "ingredient_recommend"),
        ("有什么好吃的菜推荐", "ingredient_recommend"),
        ("推荐几道适合夏天的菜", "ingredient_recommend"),
        ("我想吃辣的，有什么推荐", "ingredient_recommend"),
        # step_qa (做法/步骤/制作)
        ("麻婆豆腐怎么做", "step_qa"),
        ("红烧肉的制作步骤", "step_qa"),
        ("如何制作番茄炒蛋", "step_qa"),
        ("蛋糕的烘焙方法", "step_qa"),
        ("糖醋排骨的做法", "step_qa"),
        ("清蒸鲈鱼需要什么材料", "step_qa"),
        ("电饭煲怎么蒸蛋糕", "step_qa"),
        ("烤鸡翅的温度和时间", "step_qa"),
        # nutrition_filter (营养/热量)
        ("番茄炒蛋的热量是多少", "nutrition_filter"),
        ("鸡肉的蛋白质含量高吗", "nutrition_filter"),
        ("蛋白质含量最高的菜有哪些", "nutrition_filter"),
        ("这个菜有多少卡路里", "nutrition_filter"),
        ("糖尿病人能吃什么", "nutrition_filter"),
        # substitution (替代)
        ("没有鸡蛋可以用什么代替", "substitution"),
        ("鱼香肉丝没有豆瓣酱可以用什么", "substitution"),
        ("不吃辣可以用什么代替辣椒", "substitution"),
        ("没有烤箱怎么做蛋糕", "substitution"),
        # image_search (图片)
        ("我想看看红烧肉的照片", "image_search"),
        ("有没有麻婆豆腐的图片", "image_search"),
        ("展示一下提拉米苏的样子", "image_search"),
        # chitchat (闲聊)
        ("你好，请问能帮我做什么", "chitchat"),
        ("谢谢", "chitchat"),
        ("再见", "chitchat"),
        # pdf_parse (PDF) — 注意：需传 files 参数模拟上传文件，
        # 否则 RouterAgent 的 "if files:" 检查不通过，永远走不到 pdf_parse
        ("帮我解析这个PDF文件", "pdf_parse"),
        ("读取这个文档", "pdf_parse"),
    ]

    results = []
    latencies = []
    by_category: dict[str, dict] = {}

    print("\n=== 意图识别测试 (30 queries, 7 categories) ===")
    for query, expected in test_cases:
        t0 = time.perf_counter()
        # 修复：pdf_parse 测试需传 files 模拟上传文件，
        # 否则 RouterAgent 中 "if files:" 永不成立
        kwargs = {"query": query}
        if expected == "pdf_parse":
            kwargs["files"] = ["mock_pdf_base64_content_for_testing"]
        result = await agent.run(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        # RouterResult has intent (IntentType enum) and confidence
        actual_intent = result.intent.value if hasattr(result, 'intent') else "unknown"
        confidence = result.confidence if hasattr(result, 'confidence') else 0.0

        match = actual_intent == expected
        results.append({
            "query": query,
            "expected": expected,
            "actual": actual_intent,
            "match": match,
            "confidence": confidence,
            "latency_ms": round(elapsed, 1),
        })

        marker = "✅" if match else "❌"
        print(f"  {marker} {query[:24]:24s} → {actual_intent:24s} (期望: {expected}) [{elapsed:.0f}ms]")

        if expected not in by_category:
            by_category[expected] = {"total": 0, "correct": 0, "latencies": []}
        by_category[expected]["total"] += 1
        by_category[expected]["latencies"].append(elapsed)
        if match:
            by_category[expected]["correct"] += 1

    total = len(results)
    correct = sum(1 for r in results if r["match"])

    category_stats = {}
    for cat, stats in sorted(by_category.items()):
        acc = stats["correct"] / stats["total"] * 100
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
        category_stats[cat] = {
            "accuracy_pct": round(acc, 1),
            "total": stats["total"],
            "correct": stats["correct"],
            "avg_latency_ms": round(avg_lat, 1),
        }

    latencies.sort()
    n = len(latencies)

    return {
        "intent_recognition": {
            "total_queries": total,
            "overall_accuracy_pct": round(correct / total * 100, 1),
            "correct": correct,
            "wrong": total - correct,
            "p50_latency_ms": round(latencies[int(n * 0.50)], 1),
            "p95_latency_ms": round(latencies[int(n * 0.95)], 1),
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "by_category": category_stats,
        }
    }


# ── 3. Text-to-SQL 安全测试 ──

async def bench_sql() -> dict[str, Any]:
    """测试 SQL Validator 安全拦截 + Sandbox 执行"""
    from src.core.sql.validator import SQLValidator
    from src.core.sql.sandbox import SQLSandbox

    validator = SQLValidator()
    sandbox = SQLSandbox()

    attack_queries = [
        ("DROP TABLE recipes", "DROP 语句"),
        ("DELETE FROM nutrition", "DELETE 语句"),
        ("ALTER TABLE ingredients ADD COLUMN hack text", "ALTER 语句"),
        ("INSERT INTO recipes VALUES (1,'hack')", "INSERT 语句"),
        ("UPDATE nutrition SET calories=0 WHERE id=1", "UPDATE 语句"),
        ("SELECT * FROM users", "越权表访问"),
        ("SELECT * FROM recipes; DROP TABLE nutrition", "分号注入"),
        ("SELECT * FROM nutrition WHERE 1=1; --", "SQL 注释"),
        ("SELECT * FROM nutrition UNION SELECT * FROM users", "UNION 注入"),
        ("' OR 1=1 --", "OR 注入"),
        ("SELECT * FROM nutrition WHERE name='' OR 1=1", "恒真条件"),
        ("'; SELECT * FROM nutrition; --", "分号逃逸"),
    ]

    normal_queries = [
        ("SELECT * FROM recipes LIMIT 5", "简单查询"),
        ("SELECT name, cook_time FROM recipes WHERE difficulty = 'easy'", "条件查询"),
        ("SELECT r.name, n.calories FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id LIMIT 10", "JOIN 查询"),
        ("SELECT cuisine, COUNT(*) as cnt FROM recipes GROUP BY cuisine ORDER BY cnt DESC", "聚合查询"),
        ("SELECT name FROM recipes WHERE name LIKE '%chicken%'", "模糊查询"),
        ("SELECT * FROM nutrition ORDER BY calories DESC LIMIT 5", "排序查询"),
    ]

    print("\n=== SQL 安全测试 ===")
    attack_results = []
    for sql, desc in attack_queries:
        is_valid, err = validator.validate(sql)
        safe = not is_valid
        attack_results.append({"sql": sql[:40], "desc": desc, "blocked": safe, "error": err})
        marker = "✅" if safe else "❌"
        print(f"  {marker} [{desc:12s}] {sql[:50]} → {'BLOCKED' if safe else 'PASSED'}")

    normal_results = []
    for sql, desc in normal_queries:
        is_valid, err = validator.validate(sql)
        if not is_valid:
            normal_results.append({"sql": sql[:40], "desc": desc, "passed": False, "error": err})
            print(f"  ❌ [{desc:12s}] {sql[:50]} → VALIDATION FAILED: {err}")
            continue
        success, result = await sandbox.execute(sql)
        passed = isinstance(result, list)
        normal_results.append({
            "sql": sql[:40], "desc": desc,
            "passed": passed,
            "rows": len(result) if isinstance(result, list) else 0,
            "error": result if isinstance(result, str) else None,
        })
        marker = "✅" if passed else "❌"
        print(f"  {marker} [{desc:12s}] {sql[:50]} → {'OK' if passed else str(result)[:40]}")

    attack_block_rate = sum(1 for r in attack_results if r["blocked"]) / len(attack_results) * 100
    normal_pass_rate = sum(1 for r in normal_results if r["passed"]) / len(normal_results) * 100

    return {
        "sql_safety": {
            "attack_queries": len(attack_queries),
            "attack_block_rate_pct": round(attack_block_rate, 1),
            "attack_blocked": sum(1 for r in attack_results if r["blocked"]),
            "normal_queries": len(normal_queries),
            "normal_pass_rate_pct": round(normal_pass_rate, 1),
            "normal_passed": sum(1 for r in normal_results if r["passed"]),
            "details": {
                "attack": attack_results,
                "normal": normal_results,
            }
        }
    }


# ── 4. CLIP 多模态测试 ──

async def bench_clip() -> dict[str, Any]:
    """测试 CLIP 文本→菜名检索质量"""
    from src.core.retrievers.clip_retriever import CLIPRetriever

    retriever = CLIPRetriever()

    test_queries = [
        "Roasted Carrots and Beets",
        "chocolate cake",
        "beef steak",
        "chicken soup",
        "pasta with tomato",
        "fresh salad",
        "fish",
        "vegetable soup",
        "rice",
        "pancake",
    ]

    print("\n=== CLIP 文本→菜名检索测试 ===")
    latencies = []
    all_results = []

    for query in test_queries:
        t0 = time.perf_counter()
        chunks = await retriever.retrieve(query=query, top_k=5)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        names = [c.content for c in chunks]
        scores = [c.score for c in chunks]
        all_results.append({
            "query": query,
            "top5_names": names,
            "top5_scores": [round(s, 4) for s in scores],
            "latency_ms": round(elapsed, 1),
        })

        print(f"  [{elapsed:6.1f}ms] {query:30s} →")
        for name, score in zip(names, scores):
            print(f"             ({score:.4f}) {name}")

    latencies.sort()
    n = len(latencies)

    return {
        "clip_retrieval": {
            "total_queries": n,
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "p50_ms": round(latencies[int(n * 0.50)], 1),
            "p95_ms": round(latencies[int(n * 0.95)], 1),
            "details": all_results,
        }
    }


# ── 5. 上下文压缩测试 ──

async def bench_compression() -> dict[str, Any]:
    """测试 ConversationMemory 上下文压缩效果"""
    from src.core.memory import ConversationMemory

    session_id = f"bench_compress_{int(time.time())}"
    memory = await ConversationMemory.get_or_create(session_id)

    turns_data = [
        ("user", "推荐一道红烧肉的做法"),
        ("assistant", "红烧肉是一道经典的中式菜肴，主要用料包括五花肉、冰糖、酱油、八角、桂皮等。做法如下：1) 五花肉切块焯水；2) 锅中放少许油，加入冰糖炒出糖色；3) 放入五花肉翻炒上色；4) 加入酱油、八角、桂皮和适量水；5) 大火烧开后转小火慢炖40分钟；6) 最后收汁即可。红烧肉的关键在于火候的掌握和糖色的炒制。"),
        ("user", "蛋白质含量高的食物有哪些"),
        ("assistant", "蛋白质含量高的食物主要包括：1) 鸡胸肉(每100g约31g蛋白质)；2) 鸡蛋(每个约6-7g)；3) 牛奶(每100ml约3.3g)；4) 三文鱼(每100g约20g)；5) 豆腐(每100g约8g)；6) 牛肉(每100g约26g)；7) 虾仁(每100g约24g)；8) 希腊酸奶(每100g约10g)；9) 藜麦(每100g约14g)；10) 杏仁(每100g约21g)。对于健身人群，建议每公斤体重摄入1.2-2.0g蛋白质。"),
        ("user", "番茄炒蛋的家常做法"),
        ("assistant", "番茄炒蛋是最经典的家常菜之一。材料：番茄2个、鸡蛋3个、葱、盐、糖、食用油。做法：1) 番茄切块，鸡蛋打散加少许盐；2) 锅中热油，倒入蛋液炒至凝固盛出；3) 锅中再加少许油，放入番茄块翻炒出汁；4) 加入适量糖和盐调味；5) 倒回炒好的鸡蛋，翻炒均匀；6) 撒上葱花即可出锅。"),
        ("user", "烘焙巧克力和普通巧克力有什么区别"),
        ("assistant", "烘焙巧克力（Baking Chocolate）和普通巧克力的主要区别：1) 含糖量：烘焙巧克力含糖量低，普通巧克力含糖量高；2) 可可脂含量：烘焙巧克力通常更高；3) 添加剂：普通巧克力有乳化剂等，烘焙巧克力更纯净；4) 用途：烘焙巧克力用于融化后制作糕点。不建议用普通巧克力替代烘焙巧克力。"),
        ("user", "如何判断牛排的熟度"),
        ("assistant", "判断牛排熟度的方法：1) 触感法：拇指按食指基部≈三分熟，按中指≈五分熟，按无名指≈七分熟，按小指≈全熟；2) 温度计法：三分熟52°C、五分熟57°C、七分熟63°C、全熟68°C+；3) 切面观察颜色。推荐五分熟或七分熟。"),
        ("user", "素食者如何补充蛋白质"),
        ("assistant", "素食者蛋白质来源：1) 豆类：豆腐(8g/100g)、黑豆(36g/100g)、鹰嘴豆(19g/100g)；2) 谷物：藜麦(14g/100g)、燕麦(17g/100g)；3) 坚果：杏仁(21g/100g)、奇亚籽(17g/100g)；4) 菌菇蔬菜。建议豆类+谷物互补氨基酸谱。"),
        ("user", "怎样煮出完美的溏心蛋"),
        ("assistant", "溏心蛋关键在精确控时：冷水下锅煮开后续4-5分钟，或开水下锅6-7分钟，捞出泡冰水。温泉蛋法65°C煮45分钟。技巧：室温蛋防爆裂、立即冰镇、加醋或盐到水中。"),
        ("user", "空气炸锅和烤箱有什么区别"),
        ("assistant", "空气炸锅利用高速热风循环，预热快(3-5分钟)，比烤箱快30-40%，外酥里嫩少油；烤箱加热更均匀适合烘焙。炸锅适合薯条鸡翅，烤箱适合蛋糕面包。建议两者搭配使用。"),
    ]

    print("\n=== 上下文压缩测试 ===")

    for role, content in turns_data:
        if role == "user":
            # 修复：TurnRecord 使用 query 字段（非 role/content），
            # add_turn(query, intent, slots, answer, provenance) 接收 5 个位置参数
            await memory.add_turn(
                query=content,
                intent="step_qa",
                slots={},
                answer="",
                provenance=[],
            )
    print(f"  已添加 {sum(1 for r,_ in turns_data if r == 'user')} 轮用户对话")

    # 获取压缩前的上下文
    context_before = await memory.get_context(max_tokens=4096)
    char_before = len(context_before) if context_before else 0
    token_est_before = char_before // 2
    print(f"  压缩前: {char_before} chars, ~{token_est_before} tokens")

    # 尝试触发压缩 (设置低 max_tokens 强制压缩)
    compressed = await memory.compress_if_needed(max_tokens=512)
    print(f"  压缩触发结果: {compressed}")

    # 获取压缩后的上下文
    context_after = await memory.get_context(max_tokens=4096)
    char_after = len(context_after) if context_after else 0
    token_est_after = char_after // 2

    compression_ratio = 0.0
    if char_before > 0:
        compression_ratio = (1 - char_after / char_before) * 100

    metrics = {}
    if hasattr(ConversationMemory, 'get_compression_metrics'):
        try:
            metrics = ConversationMemory.get_compression_metrics()
        except Exception:
            pass

    print(f"  压缩后: {char_after} chars, ~{token_est_after} tokens")
    print(f"  压缩率: {compression_ratio:.1f}%")
    if metrics:
        print(f"  压缩指标: {json.dumps(metrics, indent=2)}")

    return {
        "context_compression": {
            "total_turns": len(turns_data),
            "chars_before": char_before,
            "chars_after": char_after,
            "tokens_before_est": token_est_before,
            "tokens_after_est": token_est_after,
            "compression_ratio_pct": round(compression_ratio, 1),
            "compression_metrics": metrics,
        }
    }


# ── 6. 端到端流程测试 ──

async def bench_e2e() -> dict[str, Any]:
    """端到端 Orchestrator 流程测试"""
    from src.orchestrator.supervisor import RecipeOrchestrator
    from src.api.schemas import AskRequest

    orchestrator = RecipeOrchestrator()

    e2e_queries = [
        "你好",
        "推荐一道红烧肉的菜谱",
        "麻婆豆腐怎么做",
        "番茄炒蛋的热量是多少",
        "没有鸡蛋可以用什么代替",
        "我想看看红烧肉的照片",
    ]

    print("\n=== 端到端测试 ===")
    latencies = []
    intent_results = []

    for query in e2e_queries:
        t0 = time.perf_counter()
        try:
            response = await orchestrator.ask(AskRequest(query=query, stream=False))
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            answer_preview = response.answer[:60] if response.answer else "(空)"
            intent_results.append({
                "query": query[:20],
                "intent": response.intent,
                "latency_ms": round(elapsed, 1),
                "confidence": response.confidence,
                "answer_preview": answer_preview,
            })
            print(f"  [{elapsed:7.1f}ms] {query[:20]:20s} → {response.intent:24s} | {answer_preview}")
        except Exception as e:
            latencies.append(0)
            print(f"  [  FAILED ] {query[:20]:20s} → {e}")

    valid_latencies = [l for l in latencies if l > 0]
    if valid_latencies:
        valid_latencies.sort()
        n = len(valid_latencies)
        return {
            "e2e": {
                "total_queries": len(e2e_queries),
                "successful": len(valid_latencies),
                "avg_latency_ms": round(sum(valid_latencies) / n, 1),
                "p50_ms": round(valid_latencies[int(n * 0.50)], 1),
                "p95_ms": round(valid_latencies[int(n * 0.95)], 1),
                "details": intent_results,
            }
        }
    return {"e2e": {"error": "all e2e tests failed"}}


# ── 7. Rerank 缓存基准测试 ──

async def bench_rerank_cache() -> dict[str, Any]:
    """测试 Rerank 缓存对英文查询延迟的优化效果"""
    from src.core.retrievers.hybrid import HybridRetriever

    retriever = HybridRetriever()

    # 先跑一遍所有查询预热 + 填充缓存
    warmup_queries = [
        "红烧肉的做法", "番茄炒蛋的热量是多少",
        "Roasted chicken with vegetables",
        "Chocolate cake recipe",
        "Pasta with tomato sauce",
    ]
    for q in warmup_queries:
        await retriever.retrieve(q)

    # 清除预热统计，只保留缓存内容
    cache_stats_before = retriever.get_rerank_cache_stats()

    # ── 冷启动测试：清空缓存后跑英文查询 ──
    retriever.clear_rerank_cache()
    english_queries = [
        "Roasted chicken with vegetables",
        "Chocolate cake recipe",
        "Pasta with tomato sauce",
    ]

    cold_latencies = []
    print("\n=== Rerank 缓存测试 [冷启动-无缓存] ===")
    for q in english_queries:
        t0 = time.perf_counter()
        await retriever.retrieve(q)
        elapsed = (time.perf_counter() - t0) * 1000
        cold_latencies.append(elapsed)
        print(f"  ❄️  {q[:30]:30s} → {elapsed:.1f}ms")

    cold_p50 = sorted(cold_latencies)[len(cold_latencies) // 2]

    # ── 热启动测试：同样查询再跑一次（全缓存命中）──
    hot_latencies = []
    print("\n=== Rerank 缓存测试 [热启动-全命中] ===")
    for q in english_queries:
        t0 = time.perf_counter()
        await retriever.retrieve(q)
        elapsed = (time.perf_counter() - t0) * 1000
        hot_latencies.append(elapsed)
        print(f"  🔥  {q[:30]:30s} → {elapsed:.1f}ms")

    hot_p50 = sorted(hot_latencies)[len(hot_latencies) // 2]
    cache_stats = retriever.get_rerank_cache_stats()

    # ── 变体查询测试：相似但不同的英文查询（部分缓存命中）──
    variant_queries = [
        ("Roasted chicken", "Roasted chicken with vegetables → 变体"),
        ("chocolate cake", "Chocolate cake recipe → 变体"),
        ("pasta sauce", "Pasta with tomato sauce → 变体"),
    ]

    variant_latencies = []
    print("\n=== Rerank 缓存测试 [变体查询-部分命中] ===")
    for q, label in variant_queries:
        t0 = time.perf_counter()
        await retriever.retrieve(q)
        elapsed = (time.perf_counter() - t0) * 1000
        variant_latencies.append(elapsed)
        print(f"  🔶  {label:40s} → {elapsed:.1f}ms")

    variant_p50 = sorted(variant_latencies)[len(variant_latencies) // 2]
    cache_stats_final = retriever.get_rerank_cache_stats()

    return {
        "rerank_cache": {
            "description": "Rerank 缓存对英文查询的加速效果（CrossEncoder 结果缓存）",
            "mechanism": "缓存 (query, chunk_id) → score，避免重复推理",
            "cold_p50_ms": round(cold_p50, 1),
            "hot_p50_ms": round(hot_p50, 1),
            "variant_p50_ms": round(variant_p50, 1),
            "speedup_cold_to_hot": f"{cold_p50 / max(hot_p50, 0.1):.1f}x",
            "cold_latencies_ms": [round(x, 1) for x in cold_latencies],
            "hot_latencies_ms": [round(x, 1) for x in hot_latencies],
            "variant_latencies_ms": [round(x, 1) for x in variant_latencies],
            "cache_hit_rate_pct": cache_stats["hit_rate_pct"],
            "cache_size": cache_stats["cache_size"],
        }
    }


# ── 主入口 ──

async def main():
    print("=" * 70)
    print("  Multi-Modal Recipe Assistant — 综合基准测试")
    print("=" * 70)

    all_results = {}

    try:
        all_results.update(await bench_retrieval())
    except Exception as e:
        print(f"  ❌ 检索测试失败: {e}")
        all_results["retrieval"] = {"error": str(e)}
    print("\n" + "-" * 70)

    try:
        all_results.update(await bench_intent())
    except Exception as e:
        print(f"  ❌ 意图测试失败: {e}")
        all_results["intent_recognition"] = {"error": str(e)}
    print("\n" + "-" * 70)

    try:
        all_results.update(await bench_sql())
    except Exception as e:
        print(f"  ❌ SQL测试失败: {e}")
        all_results["sql_safety"] = {"error": str(e)}
    print("\n" + "-" * 70)

    try:
        all_results.update(await bench_clip())
    except Exception as e:
        print(f"  ❌ CLIP测试失败: {e}")
        all_results["clip_retrieval"] = {"error": str(e)}
    print("\n" + "-" * 70)

    try:
        all_results.update(await bench_compression())
    except Exception as e:
        print(f"  ❌ 压缩测试失败: {e}")
        all_results["context_compression"] = {"error": str(e)}
    print("\n" + "-" * 70)

    try:
        all_results.update(await bench_e2e())
    except Exception as e:
        print(f"  ❌ 端到端测试失败: {e}")
        all_results["e2e"] = {"error": str(e)}
    print("\n" + "-" * 70)

    try:
        all_results.update(await bench_rerank_cache())
    except Exception as e:
        print(f"  ❌ Rerank 缓存测试失败: {e}")
        all_results["rerank_cache"] = {"error": str(e)}

    # ── 输出汇总报告 ──
    report_path = BASE_DIR / "src" / "eval" / "benchmark_report_latest.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 安全取值
    def g(d, *keys):
        v = d
        for k in keys:
            if isinstance(v, dict):
                v = v.get(k, "N/A")
            else:
                return "N/A"
        return v if v is not None else "N/A"

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                     关键指标摘要                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 检索架构 (Hybrid: BM25 + FAISS + Rerank)                        │
│     P50 延迟: {g(all_results, 'retrieval', 'p50_ms'):>8}ms              │
│     P95 延迟: {g(all_results, 'retrieval', 'p95_ms'):>8}ms              │
│     平均延迟: {g(all_results, 'retrieval', 'avg_latency_ms'):>8}ms      │
│     热缓存:   {g(all_results, 'retrieval', 'warm_latency_ms'):>8}ms      │
│                                                                     │
│  🎯 意图识别 (RouterAgent: regex + FC + fallback)                   │
│     总体准确率: {g(all_results, 'intent_recognition', 'overall_accuracy_pct'):>7}%  │
│     平均延迟:   {g(all_results, 'intent_recognition', 'avg_latency_ms'):>8}ms    │
│                                                                     │
│  🔒 Text-to-SQL 安全 (sqlglot AST + 白名单)                         │
│     攻击拦截率: {g(all_results, 'sql_safety', 'attack_block_rate_pct'):>7}%  │
│     正常通过率: {g(all_results, 'sql_safety', 'normal_pass_rate_pct'):>7}%  │
│                                                                     │
│  🖼️  CLIP 多模态检索                                                  │
│     平均延迟: {g(all_results, 'clip_retrieval', 'avg_latency_ms'):>8}ms    │
│                                                                     │
│  💾 上下文压缩 (LLM 摘要)                                             │
│     压缩率: {g(all_results, 'context_compression', 'compression_ratio_pct'):>7}%  │
│                                                                     │
│  ⚡ 端到端流程                                                        │
│     平均延迟: {g(all_results, 'e2e', 'avg_latency_ms'):>8}ms            │
│                                                                     │
│  🚀 Rerank 缓存 (CrossEncoder 结果复用)                              │
│     冷启动 P50: {g(all_results, 'rerank_cache', 'cold_p50_ms'):>8}ms      │
│     热启动 P50: {g(all_results, 'rerank_cache', 'hot_p50_ms'):>8}ms      │
│     加速比:     {g(all_results, 'rerank_cache', 'speedup_cold_to_hot'):>8}    │
│     缓存命中率: {g(all_results, 'rerank_cache', 'cache_hit_rate_pct'):>7}%  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

完整报告: {report_path}
""")


if __name__ == "__main__":
    asyncio.run(main())
