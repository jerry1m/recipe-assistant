"""
文本 RAG Agent — 混合检索 + 本地 LLM 生成回答
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import TextRAGResult


# ── 推荐场景专用 system prompt ──
_RECOMMEND_SYSTEM_PROMPT = (
    "You are a professional recipe recommendation assistant. "
    "The user is asking you to RECOMMEND dishes (e.g. '推荐几道菜'). "
    "Your goal is to suggest multiple (3-5) suitable recipes from the provided context.\n\n"
    "Rules:\n"
    "1. Look through ALL provided recipe context carefully. Extract recipe names, "
    "   cuisines, ingredients, and key characteristics.\n"
    "2. Recommend 3-5 dishes that best match the user's request. For each recommendation, "
    "   provide: dish name, why it's a good fit, and key ingredients.\n"
    "3. If you find recipe steps in the context, include a brief cooking tip.\n"
    "4. If the context has too few matching recipes, be honest: say what you found "
    "   and suggest what additional information would help.\n"
    "5. Format recommendations in a clear, scannable way (use bullet points).\n"
    "6. Always answer in the SAME LANGUAGE as the user's question.\n"
    "7. Be enthusiastic and helpful — the user is excited about cooking!"
)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a professional recipe assistant. Answer the user's question "
    "based on the provided recipe context. If the context doesn't contain "
    "enough information, say so honestly. Always cite the recipe names you reference. "
    "Use a friendly and helpful tone. Answer in the same language as the user's question."
)


class TextRAGAgent(BaseAgent):
    """文本问答：检索食谱知识库 + LLM 生成回答"""

    def __init__(self, retriever=None):
        super().__init__(name="text_rag", timeout=60.0, max_retries=1)
        if retriever is not None:
            self.retriever = retriever
        else:
            from src.core.retrievers.hybrid import HybridRetriever
            # 全局共享实例
            if not hasattr(TextRAGAgent, "_shared_retriever"):
                TextRAGAgent._shared_retriever = HybridRetriever()
            self.retriever = TextRAGAgent._shared_retriever

    async def _execute(self, **kwargs: Any) -> TextRAGResult:
        query = kwargs.get("query", "")
        top_k = kwargs.get("top_k", 5)
        slots = kwargs.get("slots", {})
        intent = kwargs.get("intent", "")

        from src.core.utils.llm import generate, translate_query

        # ── 1. 构建检索 query ──
        # 对于菜系推荐 (cuisine_type)，用菜系名构建精准 query
        # 对于普通问答，翻译原 query
        cuisine_type = slots.get("cuisine_type", "")
        is_recommend = (intent == "ingredient_recommend")

        if is_recommend and cuisine_type:
            # 菜系推荐: "Indian cuisine recipe" 比翻译全文更精准
            cuisine_en = await translate_query(cuisine_type)
            retrieval_query = f"{cuisine_en} cuisine recipe delicious traditional"
        else:
            retrieval_query = await translate_query(query)

        # ── 2. 检索（推荐场景取更多候选） ──
        chunks = []
        if self.retriever:
            effective_top_k = top_k * 2 if is_recommend else top_k
            chunks = await self.retriever.retrieve(retrieval_query, top_k=effective_top_k)

        # ── 3. 构建上下文 ──
        context = ""
        if chunks:
            sections = []
            for i, c in enumerate(chunks, 1):
                sections.append(
                    f"[Source {i}] (Recipe: {c.recipe_id}, Section: {c.section})\n{c.content}"
                )
            context = "\n\n".join(sections)

        # ── 4. 选择 system prompt ──
        system_prompt = _RECOMMEND_SYSTEM_PROMPT if is_recommend else _DEFAULT_SYSTEM_PROMPT

        # ── 5. 调用 LLM 生成 ──
        try:
            answer = await generate(
                query=query,
                context=context,
                system_prompt=system_prompt,
                max_new_tokens=512,
                temperature=0.3,
            )
        except Exception:
            # LLM 失败时降级：直接拼接检索结果
            if chunks:
                ans_lines = [f"🔍 找到 {len(chunks)} 个相关片段："]
                seen = set()
                for c in chunks:
                    rid = c.recipe_id
                    if rid not in seen:
                        seen.add(rid)
                        ans_lines.append(f"\n📖 {rid}:")
                    snippet = c.content[:150]
                    ans_lines.append(f"   • {snippet}...")
                answer = "\n".join(ans_lines)
            else:
                answer = "抱歉，没有找到相关食谱信息。请尝试换个问法。"

        return TextRAGResult(
            chunks=chunks,
            answer=answer,
        )
