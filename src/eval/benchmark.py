"""
自动化评测执行
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import structlog

from src.api.schemas import AskRequest
from src.core.utils.logger import configure_logging
from src.eval.metrics import faithfulness, mean_reciprocal_rank, recall_at_k
from src.orchestrator.supervisor import RecipeOrchestrator

logger = structlog.get_logger()

BASE_DIR = Path(__file__).resolve().parent


async def run_benchmark(test_file: str = "test_cases.json") -> dict[str, Any]:
    """执行全部测试用例并汇总指标"""
    configure_logging()

    # 加载测试用例
    with open(BASE_DIR / test_file) as f:
        test_cases = json.load(f)

    orchestrator = RecipeOrchestrator()
    results = []
    total_latencies = []

    for tc in test_cases:
        logger.info("benchmark.run", case_id=tc["id"], query=tc["query"])

        request = AskRequest(query=tc["query"], stream=False)
        start = time.perf_counter()
        response = await orchestrator.ask(request)
        latency = (time.perf_counter() - start) * 1000
        total_latencies.append(latency)

        retrieved_chunks = [p.chunk_id for p in response.provenance]

        results.append({
            "case_id": tc["id"],
            "query": tc["query"],
            "expected_intent": tc["intent"],
            "actual_intent": response.intent,
            "intent_match": tc["intent"] == response.intent,
            "recall@10": recall_at_k(retrieved_chunks, tc["expected_chunks"], k=10),
            "mrr": mean_reciprocal_rank(retrieved_chunks, tc["expected_chunks"]),
            "latency_ms": round(latency, 1),
            "confidence": response.confidence,
        })

    # 汇总
    total = len(results)
    intent_accuracy = sum(1 for r in results if r["intent_match"]) / total
    avg_recall = sum(r["recall@10"] for r in results) / total
    avg_mrr = sum(r["mrr"] for r in results) / total
    latencies_sorted = sorted(total_latencies)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]

    report = {
        "summary": {
            "total_cases": total,
            "intent_accuracy": round(intent_accuracy, 4),
            "avg_recall@10": round(avg_recall, 4),
            "avg_mrr": round(avg_mrr, 4),
            "p95_latency_ms": round(p95, 1),
            "avg_latency_ms": round(sum(total_latencies) / total, 1),
        },
        "details": results,
    }

    # 保存报告
    output_path = BASE_DIR / "benchmark_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("benchmark.complete", path=str(output_path))
    return report


if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else "test_cases.json"
    report = asyncio.run(run_benchmark(test_file))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
