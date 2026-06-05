"""
领域适配器抽象接口
集中管理 Prompt 模板、检索权重、评估指标、降级话术
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DomainAdapter(ABC):
    """领域适配器接口 — 迁移至新领域仅需实现此类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """领域名称，如 'recipe', 'fitness_meal'"""
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        """系统级 Prompt"""
        ...

    @abstractmethod
    def retrieval_weights(self) -> dict[str, float]:
        """检索权重 {bm25, vector, rerank}"""
        ...

    @abstractmethod
    def eval_metrics(self) -> list[str]:
        """评估指标列表"""
        ...

    @abstractmethod
    def fallback_message(self) -> str:
        """降级话术"""
        ...
