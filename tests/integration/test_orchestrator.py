"""
集成测试 — Agent 编排全链路（英文知识库，英文 query 为主）
"""

import pytest

from src.api.schemas import AskRequest
from src.orchestrator.supervisor import RecipeOrchestrator


@pytest.mark.asyncio
async def test_chitchat_flow():
    orchestrator = RecipeOrchestrator()
    request = AskRequest(query="hello", stream=False)
    response = await orchestrator.ask(request)

    assert response.intent == "chitchat"
    assert response.latency_ms > 0


@pytest.mark.asyncio
async def test_ingredient_recommend_flow():
    orchestrator = RecipeOrchestrator()
    request = AskRequest(query="how to bake a chocolate cake", stream=False)
    response = await orchestrator.ask(request)

    assert response.intent == "ingredient_recommend"


@pytest.mark.asyncio
async def test_nutrition_filter_flow():
    orchestrator = RecipeOrchestrator()
    request = AskRequest(query="how many calories in chicken", stream=False)
    response = await orchestrator.ask(request)

    assert response.intent == "nutrition_filter"


@pytest.mark.asyncio
async def test_substitution_flow():
    orchestrator = RecipeOrchestrator()
    request = AskRequest(query="substitute eggs in baking", stream=False)
    response = await orchestrator.ask(request)

    assert response.intent == "substitution"


@pytest.mark.asyncio
async def test_step_qa_flow():
    orchestrator = RecipeOrchestrator()
    request = AskRequest(query="what temperature to bake cookies", stream=False)
    response = await orchestrator.ask(request)

    assert response.intent == "step_qa"


@pytest.mark.asyncio
async def test_chinese_fallback_flow():
    """中文 query 仍能正确识别（辅助支持）"""
    orchestrator = RecipeOrchestrator()
    request = AskRequest(query="红烧肉的做法", stream=False)
    response = await orchestrator.ask(request)

    assert response.intent == "ingredient_recommend"
