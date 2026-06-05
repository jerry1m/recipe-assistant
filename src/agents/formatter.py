"""
Formatter Agent — 结构化输出 + 免责声明
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import FormatterResult


class FormatterAgent(BaseAgent):
    """最终格式化输出"""

    def __init__(self):
        super().__init__(name="formatter", timeout=3.0, max_retries=1)

    async def _execute(self, **kwargs: Any) -> FormatterResult:
        draft = kwargs.get("draft", "")
        provenance = kwargs.get("provenance", [])
        need_disclaimer = kwargs.get("need_disclaimer", False)

        # 格式化：添加引用标注
        if provenance:
            refs = "\n\n---\n📎 参考来源：\n"
            for item in provenance[:3]:
                if isinstance(item, dict):
                    refs += f"- {item.get('source', 'unknown')} (chunk: {item.get('chunk_id', '?')}, score: {item.get('score', 0):.2f})\n"
                else:
                    refs += f"- {item.source} (chunk: {item.chunk_id}, score: {item.score:.2f})\n"
            draft += refs

        # 免责声明
        disclaimer = ""
        if need_disclaimer:
            disclaimer = "\n\n⚠️ 营养建议仅供参考，具体请以专业营养师意见为准。"
            draft += disclaimer

        return FormatterResult(
            final_response=draft,
            disclaimer=disclaimer,
        )
