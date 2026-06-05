"""
单元测试 — Schema & Config & Metrics
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import AskRequest, AskResponse, IntentType, ProvenanceItem
from src.core.config import get_settings
from src.core.utils.metrics import MetricsCollector


class TestSchema:
    def test_ask_request_defaults(self):
        req = AskRequest(query="红烧肉怎么做")
        assert req.query == "红烧肉怎么做"
        assert req.stream is True
        assert req.max_tokens == 2048
        assert req.temperature == 0.3

    def test_ask_request_with_images(self):
        req = AskRequest(query="图片搜索", images=["base64data"])
        assert len(req.images) == 1

    def test_ask_request_invalid(self):
        with pytest.raises(ValidationError):
            AskRequest()

    def test_ask_response(self):
        resp = AskResponse(
            answer="建议使用五花肉",
            provenance=[
                ProvenanceItem(chunk_id="c_001", recipe_id="r_001", score=0.95, source="test", snippet="五花肉")
            ],
            intent="substitution",
            confidence=0.87,
            latency_ms=1450,
        )
        assert resp.answer == "建议使用五花肉"
        assert len(resp.provenance) == 1

    def test_intent_enum_values(self):
        assert IntentType.INGREDIENT_RECOMMEND.value == "ingredient_recommend"
        assert IntentType.CHITCHAT.value == "chitchat"


class TestConfig:
    def test_default_values(self):
        settings = get_settings()
        assert settings.app_name == "Multi-Modal Recipe Assistant"
        assert settings.llm_temperature == 0.3
        assert settings.retriever_top_k == 10

    def test_env_prefix(self):
        settings = get_settings()
        assert settings.model_config["env_prefix"] == "RECIPE_"


class TestMetrics:
    def test_record_and_stats(self):
        m = MetricsCollector()
        m.record_agent_call("agent_a", True, 100.0)
        m.record_agent_call("agent_a", True, 200.0)
        m.record_agent_call("agent_a", False, 300.0, "error")

        stats = m.get_agent_stats()
        assert stats["agent_a"]["call_count"] == 3
        assert stats["agent_a"]["success_rate"] == pytest.approx(2 / 3, 0.01)
        assert stats["agent_a"]["avg_latency_ms"] == pytest.approx(200.0, 0.1)

    def test_business_events(self):
        m = MetricsCollector()
        m.record_business_event("search", query="test", hits=5)
        m.record_business_event("search", query="test2", hits=3)

        stats = m.get_business_stats()
        assert stats["search"]["count"] == 2

    def test_error_rate_property(self):
        m = MetricsCollector()
        m.record_agent_call("agent_b", True, 50.0)
        assert m._agent_metrics["agent_b"].success_rate == 1.0
