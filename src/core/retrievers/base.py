"""
检索引擎基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """所有检索器的统一接口"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """检索并返回相关文档块列表"""
        ...
