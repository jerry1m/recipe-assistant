"""
Critic 质检 Agent — 事实核查 + 合规校验 + 完整性检查
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import CritiqueReason, CriticResult


class CriticAgent(BaseAgent):
    """对 Agent 输出进行质量检查"""

    def __init__(self):
        super().__init__(name="critic", timeout=5.0, max_retries=2)

    async def _execute(self, **kwargs: Any) -> CriticResult:
        draft = kwargs.get("draft", "")
        provenance = kwargs.get("provenance", [])
        intent = kwargs.get("intent", "")
        need_disclaimer = kwargs.get("need_disclaimer", False)

        reasons: list[CritiqueReason] = []
        suggestions: list[str] = []

        # 1. 完整性检查
        if not draft:
            reasons.append(CritiqueReason.INCOMPLETE)
            suggestions.append("回答为空，请重新生成")

        # 2. 合规检查：营养建议是否带免责声明
        # 注意：如果 need_disclaimer=True，说明 Formatter 后续会添加免责声明，
        # Critic 不需要在此检查，避免误判导致重试循环。
        if not need_disclaimer and intent in ("nutrition_filter", "substitution"):
            disclaimer_keywords = [
                "仅供参考", "建议咨询", "免责",
                "for reference", "consult", "disclaimer", "advice",
            ]
            if not any(kw in draft.lower() for kw in disclaimer_keywords):
                reasons.append(CritiqueReason.UNSAFE_ADVICE)
                suggestions.append("营养建议需附带免责声明 / Add nutrition disclaimer")

        # 3. TODO: 幻觉检测（检查答案中的引用是否在 provenance 中）
        # if not all(ref in provenance for ref in extracted_refs):
        #     reasons.append(CritiqueReason.HALLUCINATION)

        passed = len(reasons) == 0

        return CriticResult(
            passed=passed,
            reasons=reasons,
            suggestions=suggestions,
            data={
                "draft_length": len(draft),
                "provenance_count": len(provenance),
            },
        )
