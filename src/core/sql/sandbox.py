"""
SQL 安全沙箱 — 只读执行 + 超时熔断
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

from src.core.config import get_settings
from src.core.sql.validator import SQLValidator

settings = get_settings()


class SQLSandbox:
    """只读 SQL 执行环境，带超时熔断"""

    def __init__(self, db_url: str = ""):
        self.db_url = db_url or settings.sql_database_url
        self.validator = SQLValidator()
        self.max_rows = settings.sql_max_rows
        self.timeout_ms = settings.sql_query_timeout_ms

    async def execute(self, sql: str) -> tuple[bool, list[dict[str, Any]] | str]:
        """
        安全执行 SQL。
        返回 (success, rows | error_message)
        """
        # 1. 校验
        is_safe, err = self.validator.validate(sql)
        if not is_safe:
            return False, err

        # 2. 只读执行 + 超时
        try:
            rows = await asyncio.wait_for(
                self._run_query(sql),
                timeout=self.timeout_ms / 1000,
            )
            return True, rows[: self.max_rows]
        except asyncio.TimeoutError:
            return False, f"查询超时 ({self.timeout_ms}ms)"
        except Exception as e:
            return False, f"执行失败: {e}"

    async def _run_query(self, sql: str) -> list[dict[str, Any]]:
        """在独立连接中执行只读查询"""
        def _sync_query():
            conn = sqlite3.connect(self.db_url.replace("sqlite:///", ""))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            conn.close()
            return rows

        return await asyncio.to_thread(_sync_query)
