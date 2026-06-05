"""
评测指标计算
"""

from __future__ import annotations

from typing import Any


def recall_at_k(retrieved: list[str], expected: list[str], k: int = 10) -> float:
    """Recall@K: 期望文档在前 K 个检索结果中的比例"""
    if not expected:
        return 1.0
    retrieved_set = set(retrieved[:k])
    hit = sum(1 for e in expected if e in retrieved_set)
    return hit / len(expected)


def faithfulness(cited: list[str], actual: list[str]) -> float:
    """Faithfulness: 答案引用在 provenance 中可追溯的比例"""
    if not cited:
        return 1.0
    hit = sum(1 for c in cited if c in actual)
    return hit / len(cited)


def mean_reciprocal_rank(
    retrieved: list[str],
    expected: list[str],
) -> float:
    """MRR: 第一个相关文档的排名倒数"""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0
