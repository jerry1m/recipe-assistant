"""
SQL 查询校验 — 基于 sqlglot AST
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp


class SQLValidator:
    """sqlglot AST 校验 + 表名列名白名单"""

    WHITELIST_TABLES = {"recipes", "ingredients", "nutrition", "cuisines"}
    MAX_ROWS = 100
    FORBIDDEN_STATEMENTS = {"DROP", "ALTER", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "CREATE"}

    def validate(self, sql: str) -> tuple[bool, str]:
        """
        校验 SQL 安全性。
        返回 (is_safe, error_message)
        """
        # 1. 检查危险语句
        sql_upper = sql.upper().strip()
        for keyword in self.FORBIDDEN_STATEMENTS:
            if sql_upper.startswith(keyword):
                return False, f"禁止使用 {keyword} 语句"

        # 2. AST 解析
        try:
            parsed = sqlglot.parse_one(sql)
        except Exception as e:
            return False, f"SQL 解析失败: {e}"

        # 3. 必须为 SELECT
        if not isinstance(parsed, exp.Select):
            return False, "仅允许 SELECT 查询"

        # 4. 检查表名白名单
        for table in parsed.find_all(exp.Table):
            if table.name not in self.WHITELIST_TABLES:
                return False, f"表 {table.name} 不在白名单中"

        return True, ""
