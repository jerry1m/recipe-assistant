"""
反思 Agent (Reflection) — 对草稿输出进行 LLM 驱动的质检与修订

参考 travel-agent-guide 的 ReflectionAgent 设计，适配 recipe-assistant 场景。
相比 CriticAgent（规则检查），ReflectionAgent 使用 LLM 做深层语义审查：
- 事实一致性：答案是否忠实于检索结果
- 完整性：是否覆盖用户问题的所有要点
- 安全性：营养建议是否合规
- 可操作性：步骤是否清晰可行
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import ReflectionResult

_REFLECTION_SYSTEM_PROMPT = (
    "You are a strict quality inspector for a recipe assistant. "
    "Review the draft answer against these criteria:\n\n"
    "1. **Factuality**: Does the answer stay faithful to the provided context? "
    "Do NOT fabricate recipe names, steps, or nutrition data.\n"
    "2. **Completeness**: Does it fully address the user's question? "
    "If the question has multiple parts, all should be answered.\n"
    "3. **Safety**: Nutrition advice must include a disclaimer "
    "(e.g., 'consult a professional', '仅供参考').\n"
    "4. **Actionability**: Are cooking steps clear and executable?\n\n"
    "Output format (strict JSON, no other text):\n"
    "{{\n"
    "  \"passed\": true/false,\n"
    "  \"critique\": \"列出具体问题（若无问题则写'无'）\",\n"
    "  \"revised\": \"修订后的完整回答（与passed无关，始终给出优化版本）\",\n"
    "  \"reasons\": [\"问题类别1\", \"问题类别2\"],\n"
    "  \"suggestions\": [\"改进建议1\", \"改进建议2\"]\n"
    "}}\n\n"
    "Criteria:\n"
    "- passed=true 仅当草稿在事实性、完整性、安全性、可操作性方面都达标。\n"
    "- revised 是对 draft 的优化版本，即使 passed=true 也可以润色。\n"
    "- 不要过度挑剔 — 日常食谱问答不需要学术级严谨。"
)


class ReflectionAgent(BaseAgent):
    """LLM 驱动的反思质检 — 比规则 Critic 更深层的语义审查。"""

    def __init__(self):
        super().__init__(name="reflection", timeout=15.0, max_retries=2)

    async def _execute(self, **kwargs: Any) -> ReflectionResult:
        draft = kwargs.get("draft", "")
        query = kwargs.get("query", "")
        context = kwargs.get("context", "")
        provenance = kwargs.get("provenance", [])

        if not draft:
            return ReflectionResult(
                passed=False,
                critique="草稿为空",
                revised="",
                reasons=["incomplete"],
                suggestions=["请先生成回答再质检"],
            )

        # ── 构建审查上下文 ──
        provenance_str = ""
        if provenance:
            items = []
            for p in provenance[:5]:
                if isinstance(p, dict):
                    items.append(f"- {p.get('recipe_id', '?')}: {p.get('snippet', '')[:100]}")
                else:
                    items.append(f"- {p}")
            provenance_str = "\n".join(items)

        user_prompt_parts = [f"## 用户问题\n{query}\n"]
        if context:
            user_prompt_parts.append(f"## 检索上下文\n{context[:1000]}\n")
        if provenance_str:
            user_prompt_parts.append(f"## 引用来源\n{provenance_str}\n")
        user_prompt_parts.append(f"## 草稿回答\n{draft}\n")

        from src.core.utils.llm import generate_structured
        result_text = await generate_structured(
            query="\n".join(user_prompt_parts),
            system_prompt=_REFLECTION_SYSTEM_PROMPT,
            max_new_tokens=1536,
            temperature=0.1,
        )

        return self._parse_result(result_text, draft)

    def _parse_result(self, text: str, draft: str) -> ReflectionResult:
        """解析 LLM 返回的 JSON。"""
        import json
        import re

        # 提取 JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end+1])
                return ReflectionResult(
                    passed=data.get("passed", False),
                    critique=data.get("critique", ""),
                    revised=data.get("revised", draft),
                    reasons=data.get("reasons", []),
                    suggestions=data.get("suggestions", []),
                )
            except json.JSONDecodeError:
                pass

        # 解析失败，返回宽松结果
        return ReflectionResult(
            passed=True,  # 保守：不阻断流程
            critique=f"反思解析失败，原始输出：{text[:200]}",
            revised=draft,
            reasons=[],
            suggestions=[],
        )
