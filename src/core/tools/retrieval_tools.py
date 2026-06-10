"""
食谱检索工具 — 注册到 ToolRegistry，供 Planner Agent / 外部调用。
"""

from __future__ import annotations

from typing import Any

from src.core.tools.registry import ToolRegistry


async def search_recipes(
    query: str,
    top_k: int = 5,
    cuisine_type: str = "",
) -> list[dict[str, Any]]:
    """根据关键词检索食谱。返回食谱片段列表。"""
    from src.core.retrievers.hybrid import HybridRetriever

    retriever = HybridRetriever()
    if cuisine_type:
        from src.core.utils.llm import translate_query
        cuisine_en = await translate_query(cuisine_type)
        retrieval_query = f"{cuisine_en} cuisine recipe delicious traditional"
    else:
        from src.core.utils.llm import translate_query
        retrieval_query = await translate_query(query)

    from src.api.schemas import Chunk
    chunks: list[Chunk] = await retriever.retrieve(retrieval_query, top_k=top_k)
    return [
        {
            "chunk_id": c.chunk_id,
            "recipe_id": c.recipe_id,
            "content": c.content[:300],
            "score": c.score,
        }
        for c in chunks
    ]


async def search_nutrition(
    food: str,
    nutrient: str = "",
) -> list[dict[str, Any]]:
    """查询食物的营养信息（热量、蛋白质、脂肪等）。"""
    from src.agents.nutrition_sql import NutritionSQLAgent

    agent = NutritionSQLAgent()
    from src.api.schemas import NutritionSQLResult
    result: NutritionSQLResult = await agent.run(query=food, slots={"nutrient": nutrient} if nutrient else {})
    if result.success and result.rows:
        return result.rows
    return []


async def search_substitution(
    ingredient: str,
    dietary_restriction: str = "",
) -> list[dict[str, str]]:
    """查询食材替换方案。"""
    from src.agents.substitution import SubstitutionAgent

    agent = SubstitutionAgent()
    from src.api.schemas import SubstitutionResult
    result: SubstitutionResult = await agent.run(
        query=ingredient,
        slots={"dietary_restriction": dietary_restriction} if dietary_restriction else {},
    )
    return result.substitutions


def register_tools(registry: ToolRegistry) -> None:
    """自动发现入口：将此模块中的所有工具注册到 registry。"""
    registry.register(
        "search_recipes",
        search_recipes,
        description="根据关键词检索食谱知识库，支持按菜系筛选。返回食谱片段列表。",
        json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词（中文/英文皆可）"},
                "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
                "cuisine_type": {"type": "string", "description": "菜系筛选（可选，如川菜、Italian）"},
            },
            "required": ["query"],
        },
    )
    registry.register(
        "search_nutrition",
        search_nutrition,
        description="查询食物/菜品的营养信息，支持按营养素筛选。返回结构化营养数据。",
        json_schema={
            "type": "object",
            "properties": {
                "food": {"type": "string", "description": "食物或菜品名称"},
                "nutrient": {"type": "string", "description": "营养素类型（calories/protein/fat/carbs），不传则返回全部"},
            },
            "required": ["food"],
        },
    )
    registry.register(
        "search_substitution",
        search_substitution,
        description="查询食材替换方案，如过敏原替代、素食替代等。返回替代食材列表。",
        json_schema={
            "type": "object",
            "properties": {
                "ingredient": {"type": "string", "description": "需要替换的食材名称"},
                "dietary_restriction": {"type": "string", "description": "饮食限制原因（如过敏、素食）"},
            },
            "required": ["ingredient"],
        },
    )
