"""
图文检索 Agent — 以图搜菜名（CLIP 跨模态）+ 文字兜底
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import ImageSearchResult, Recipe


class ImageSearchAgent(BaseAgent):
    """
    以图搜菜名 Agent。

    - 用户上传图片 → CLIP 编码 → 检索 5000 条菜名中最相似的 Top-K
    - 纯文字输入 → HybridRetriever 兜底
    - 结果格式: "看起来像是「XXX」，需要告诉你怎么做吗？"
    """

    def __init__(self):
        super().__init__(name="image_search", timeout=60.0, max_retries=1)
        self._clip = None
        self._hybrid = None

    @property
    def clip(self):
        if self._clip is None:
            from src.core.retrievers.clip_retriever import CLIPRetriever
            self._clip = CLIPRetriever()
        return self._clip

    @property
    def hybrid(self):
        if self._hybrid is None:
            from src.core.retrievers.hybrid import HybridRetriever
            self._hybrid = HybridRetriever()
        return self._hybrid

    async def _execute(self, **kwargs: Any) -> ImageSearchResult:
        query = kwargs.get("query", "")
        images = kwargs.get("images", [])
        top_k = min(kwargs.get("top_k", 5), 20)

        # ── 有图片 → CLIP 以图搜菜名 ──
        if images:
            return await self._search_by_image(images[0], top_k)

        # ── 无图片、有文字 → HybridRetriever 兜底 ──
        if query.strip():
            return await self._search_by_text(query, top_k)

        return ImageSearchResult(
            recipes=[],
            data={"note": "请上传图片或输入文字描述来搜索菜谱。"},
        )

    async def _search_by_image(self, image_b64: str, top_k: int) -> ImageSearchResult:
        """CLIP 以图搜菜名"""
        try:
            chunks = await self.clip.retrieve(top_k=top_k, image=image_b64)
        except Exception as exc:
            return ImageSearchResult(
                success=False,
                error=f"CLIP 检索失败: {exc}",
                recipes=[],
                data={"note": str(exc)},
            )

        recipes: list[Recipe] = []
        for c in chunks:
            recipes.append(Recipe(
                recipe_id=c.recipe_id,
                name=c.content,
            ))

        return ImageSearchResult(
            recipes=recipes,
            data={"query": "[图片]", "found": len(recipes), "method": "clip"},
        )

    async def _search_by_text(self, query: str, top_k: int) -> ImageSearchResult:
        """文字描述 → HybridRetriever 检索菜名"""
        chunks = await self.hybrid.retrieve(query, top_k=top_k)

        recipes: list[Recipe] = []
        seen = set()
        for c in chunks:
            rid = c.recipe_id
            if rid not in seen:
                seen.add(rid)
                recipes.append(Recipe(
                    recipe_id=rid,
                    name=rid.replace("_", " ").title(),
                ))

        return ImageSearchResult(
            recipes=recipes,
            data={"query": query, "found": len(recipes), "method": "text"},
        )
