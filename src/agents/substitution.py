"""
食材替换推理 Agent — 本地 LLM 推理替代食材
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import SubstitutionResult

SUBSTITUTION_SYSTEM_PROMPT = """You are a professional chef and recipe expert.
The user wants to substitute an ingredient. Suggest practical alternatives.

For each substitution, provide:
- original: the original ingredient
- alternative: the suggested replacement
- reason: why this substitution works
- ratio: the replacement ratio (e.g. "1:1", "1:2")

Output in JSON format (a list of substitution objects).
If the ingredient is unknown, return an empty list."""


class SubstitutionAgent(BaseAgent):
    """食材替换推理：LLM 推荐替代品 + 解释原因"""

    def __init__(self):
        super().__init__(name="substitution", timeout=60.0, max_retries=1)

    async def _execute(self, **kwargs: Any) -> SubstitutionResult:
        query = kwargs.get("query", "")
        ingredient = kwargs.get("ingredient", "")

        target = ingredient if ingredient else query

        # 调用本地 LLM
        try:
            from src.core.utils.llm import generate
            raw = await generate(
                query=f"I want to substitute '{target}'. Reason: {query}",
                context="",
                system_prompt=SUBSTITUTION_SYSTEM_PROMPT,
                max_new_tokens=512,
                temperature=0.3,
            )
            substitutions = self._parse_substitutions(raw, target)
            explanation = self._build_explanation(substitutions)
        except Exception:
            substitutions = [{
                "original": target,
                "alternative": "[需手动查询] 建议查阅专业替换表",
                "reason": "LLM 暂不可用，请参考常见食材替换规则",
                "ratio": "1:1",
            }]
            explanation = "LLM 推理暂不可用，以上为通用建议。"

        return SubstitutionResult(
            substitutions=substitutions,
            explanation=explanation,
        )

    def _parse_substitutions(self, raw: str, default_ingredient: str) -> list[dict[str, str]]:
        """从 LLM 输出中解析替换列表"""
        # 尝试提取 JSON
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if json_match:
            try:
                items = json.loads(json_match.group(0))
                if isinstance(items, list):
                    return items
            except json.JSONDecodeError:
                pass

        # fallback: 单行结果
        return [{
            "original": default_ingredient,
            "alternative": raw.strip()[:100],
            "reason": "LLM 推荐",
            "ratio": "1:1",
        }]

    def _build_explanation(self, substitutions: list[dict[str, str]]) -> str:
        if not substitutions:
            return "暂未找到合适的替代食材。"
        parts = ["以下是为您推荐的食材替代方案：\n"]
        for s in substitutions:
            orig = s.get("original", "?")
            alt = s.get("alternative", "?")
            reason = s.get("reason", "")
            ratio = s.get("ratio", "1:1")
            parts.append(f"• **{orig}** → **{alt}** (比例 {ratio})")
            if reason:
                parts.append(f"  原因: {reason}")
        return "\n".join(parts)
