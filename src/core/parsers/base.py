"""
解析器基类
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """所有文档解析器的统一接口"""

    @abstractmethod
    async def parse(self, source: str | bytes, **kwargs: Any) -> list[dict[str, Any]]:
        """解析文档，返回 Chunk 列表 [{chunk_id, content, section, metadata}]"""
        ...
