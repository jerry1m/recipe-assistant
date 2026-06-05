"""
Agent 基类 — 重试/超时/日志/Fallback
直接参考 multi-agent-ecommerce-system 的 agents/base_agent.py
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import AgentResult

logger = structlog.get_logger()


class BaseAgent(ABC):
    """所有 Agent 继承此基类，获得统一的重试/超时/Fallback 能力"""

    def __init__(self, name: str, timeout: float = 10.0, max_retries: int = 2):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._error_count = 0

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """每个 Agent 在此实现核心逻辑"""

    async def run(self, **kwargs: Any) -> AgentResult:
        """公开入口：包裹 _execute 加上计时、重试、Fallback"""
        start = time.perf_counter()
        self._call_count += 1

        try:
            result = await self._retry_execute(**kwargs)
            result.latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "agent.success",
                agent=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except Exception as exc:
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("agent.failed", agent=self.name, error=str(exc))
            return self._fallback(latency_ms, exc)

    async def _retry_execute(self, **kwargs: Any) -> AgentResult:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            reraise=True,
        )
        async def _inner():
            return await self._execute(**kwargs)

        return await _inner()

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        """Agent 失败时返回降级结果"""
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(exc),
            confidence=0.0,
        )

    @property
    def error_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count
