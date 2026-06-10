"""
异步熔断器 — 防止连续调用失败的 LLM/外部服务，支持快速失败

设计参考 travel-agent-guide 的 circuit_breaker.py

状态转移:
  CLOSED → (连续失败 N 次) → OPEN → (超时后) → HALF_OPEN → (成功) → CLOSED
                                                              → (失败) → OPEN
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar

import structlog

T = TypeVar("T")
logger = structlog.get_logger()


class CircuitState(str, Enum):
    CLOSED = "closed"          # 正常状态，请求通过
    OPEN = "open"              # 熔断打开，请求快速失败
    HALF_OPEN = "half_open"    # 半开状态，允许有限请求试探


@dataclass
class CircuitBreaker:
    """
    异步熔断器

    属性:
        failure_threshold: 连续失败次数阈值，达到后熔断打开
        recovery_timeout: 熔断打开后等待秒数，过后进入半开状态
        half_open_max_calls: 半开状态下允许的最大试探请求数
        name: 熔断器名称（日志用）
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    name: str = "default"

    _failures: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # 统计
    _total_calls: int = field(default=0, init=False)
    _successful_calls: int = field(default=0, init=False)
    _failed_calls: int = field(default=0, init=False)
    _rejected_calls: int = field(default=0, init=False)  # 熔断拒绝的调用

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """执行受熔断保护的异步调用"""
        async with self._lock:
            await self._transition_on_call()
            if self._state == CircuitState.OPEN:
                self._rejected_calls += 1
                raise CircuitBreakerOpenError(
                    f"circuit breaker '{self.name}' is OPEN "
                    f"(failures={self._failures}, timeout={self.recovery_timeout}s)"
                )

        self._total_calls += 1
        try:
            result = await fn()
        except Exception as exc:
            async with self._lock:
                await self._on_failure()
            self._failed_calls += 1
            logger.warning(
                "circuit_breaker.failure",
                name=self.name,
                failures=self._failures,
                state=self._state.value,
                error=str(exc),
            )
            raise

        async with self._lock:
            await self._on_success()
        self._successful_calls += 1
        return result

    async def _transition_on_call(self) -> None:
        """检查是否需要从 OPEN → HALF_OPEN 转换"""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "circuit_breaker.half_open",
                    name=self.name,
                    elapsed_seconds=round(elapsed, 1),
                )

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpenError(
                    f"circuit breaker '{self.name}' half-open call limit exceeded"
                )
            self._half_open_calls += 1

    async def _on_success(self) -> None:
        """成功调用：重置失败计数，关闭熔断器"""
        self._failures = 0
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            old_state = self._state
            self._state = CircuitState.CLOSED
            logger.info(
                "circuit_breaker.closed",
                name=self.name,
                recovered_from=old_state.value,
            )

    async def _on_failure(self) -> None:
        """失败调用：递增失败计数，可能打开熔断器"""
        self._failures += 1
        self._last_failure_time = time.monotonic()
        if self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker.opened",
                name=self.name,
                failures=self._failures,
                threshold=self.failure_threshold,
            )
        elif self._state == CircuitState.HALF_OPEN:
            # 半开状态下失败 → 立即回到 OPEN
            self._state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker.re_opened",
                name=self.name,
                half_open_call_failed=True,
            )

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_available(self) -> bool:
        return self._state != CircuitState.OPEN

    def get_stats(self) -> dict:
        """获取熔断器统计信息"""
        return {
            "name": self.name,
            "state": self._state.value,
            "failures": self._failures,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "failed_calls": self._failed_calls,
            "rejected_calls": self._rejected_calls,
            "available": self.is_available,
        }

    def reset(self) -> None:
        """手动重置熔断器"""
        self._failures = 0
        self._last_failure_time = 0.0
        self._state = CircuitState.CLOSED
        self._half_open_calls = 0
        logger.info("circuit_breaker.reset", name=self.name)


class CircuitBreakerOpenError(RuntimeError):
    """熔断器打开时抛出的异常"""
    pass
