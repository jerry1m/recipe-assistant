"""
工具注册中心 — 注册、查找、包自动发现、MCP 兼容目录

参考 travel-agent-guide 的 ToolRegistry 设计，适配 recipe-assistant 场景。
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T", bound=Callable[..., Awaitable[Any]])


@dataclass
class RegisteredTool:
    """已注册的工具元信息"""

    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    json_schema: dict[str, Any] = field(default_factory=dict)
    mcp_compatible: bool = True


class ToolRegistry:
    """注册 Agent 工具，暴露 MCP 兼容的工具目录。"""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        handler: T,
        *,
        description: str = "",
        json_schema: dict[str, Any] | None = None,
        mcp_compatible: bool = True,
    ) -> T:
        desc = description or (inspect.getdoc(handler) or "")
        self._tools[name] = RegisteredTool(
            name=name,
            description=desc.strip(),
            handler=handler,
            json_schema=json_schema or {},
            mcp_compatible=mcp_compatible,
        )
        return handler

    def has(self, name: str) -> bool:
        return name in self._tools

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def list_tools(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def mcp_tool_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.json_schema or {"type": "object", "properties": {}},
            }
            for t in self._tools.values()
            if t.mcp_compatible
        ]

    def discover_package(self, package: str) -> int:
        """扫描包内所有模块，调用 `register_tools(registry: ToolRegistry)` 注册工具。"""
        pkg = importlib.import_module(package)
        path = getattr(pkg, "__path__", None)
        if path is None:
            return 0
        prefix = pkg.__name__ + "."
        count = 0
        for _finder, modname, _ispkg in pkgutil.walk_packages(path, prefix):
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            fn = getattr(mod, "register_tools", None)
            if callable(fn):
                fn(self)
                count += 1
        return count


async def invoke(
    registry: ToolRegistry,
    name: str,
    arguments: Mapping[str, Any] | None = None,
) -> Any:
    """通过注册中心调用工具。"""
    reg = registry.get(name)
    args = dict(arguments or {})
    result = reg.handler(**args)
    if inspect.isawaitable(result):
        return await result
    return result


# ── 全局共享注册中心实例 ──
_global_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """获取全局 ToolRegistry 单例，自动发现已安装的工具包。"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        # 自动发现 recipe_assitant.tools 下的所有工具
        try:
            count = _global_registry.discover_package("src.core.tools")
            if count > 0:
                import structlog
                structlog.get_logger().info(
                    "tool_registry.discovered", tool_count=count
                )
        except Exception:
            pass
    return _global_registry
