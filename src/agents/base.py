"""
Agent 基类 — 重试/超时/日志/Fallback + Circuit Breaker 熔断保护
直接参考 multi-agent-ecommerce-system 的 agents/base_agent.py
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import AgentResult
from src.core.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = structlog.get_logger()


class BaseAgent(ABC):
    """所有 Agent 继承此基类，获得统一的重试/超时/Fallback 能力"""

    def __init__(self, name: str, timeout: float = 10.0, max_retries: int = 2):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._error_count = 0
        # 每个 Agent 拥有独立的熔断器（默认配置，可被子类覆盖）
        self._breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            half_open_max_calls=2,
            name=f"agent:{name}",
        )

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """每个 Agent 在此实现核心逻辑"""

    async def run(self, **kwargs: Any) -> AgentResult:
        """公开入口：包裹 _execute 加上计时、重试、Fallback + Circuit Breaker"""
        start = time.perf_counter()
        self._call_count += 1

        # 熔断器快速失败检查
        if not self._breaker.is_available:
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "agent.circuit_open",
                agent=self.name,
                state=self._breaker.state.value,
            )
            return self._fallback(
                latency_ms,
                CircuitBreakerOpenError(
                    f"Circuit breaker '{self._breaker.name}' is OPEN, skipping execution"
                ),
            )

        try:
            result = await self._breaker.call(
                lambda: self._retry_execute(**kwargs)
            )
            result.latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "agent.success",
                agent=self.name,
                latency_ms=round(result.latency_ms, 1),
            )
            return result
        except CircuitBreakerOpenError:
            # 熔断器打开时的快速失败
            self._error_count += 1
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("agent.circuit_open_skip", agent=self.name)
            return self._fallback(latency_ms, CircuitBreakerOpenError(f"Circuit breaker OPEN for {self.name}"))
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

    def get_breaker_stats(self) -> dict:
        """获取熔断器统计信息"""
        return self._breaker.get_stats()
