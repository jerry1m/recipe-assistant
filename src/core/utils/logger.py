"""
结构化日志 — 集成 structlog + TraceID
"""

from __future__ import annotations

import structlog


def get_logger(**kwargs):
    return structlog.get_logger(**kwargs)


def configure_logging(debug: bool = False):
    """应用启动时调用一次"""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if debug
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
