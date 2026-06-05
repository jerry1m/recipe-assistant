"""
营养查询 Agent — 三路策略（关键词模板 → LLM 翻译 → 混合检索兜底）
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import NutritionSQLResult
from src.core.sql.sandbox import SQLSandbox
from src.core.retrievers.hybrid import HybridRetriever

TABLE_SCHEMA = """
CREATE TABLE recipes (
    recipe_id TEXT PRIMARY KEY, name TEXT, cuisine TEXT, difficulty TEXT,
    prep_time TEXT, cook_time TEXT, servings INTEGER, tags TEXT, description TEXT
);
CREATE TABLE ingredients (
    recipe_id TEXT, name TEXT, amount TEXT, unit TEXT, alternative TEXT
);
CREATE TABLE nutrition (
    recipe_id TEXT PRIMARY KEY, calories REAL, protein REAL, fat REAL,
    carbs REAL, fiber REAL, sodium REAL
);
CREATE TABLE steps (
    recipe_id TEXT, step_number INTEGER, content TEXT
);
"""

# ── 关键词 SQL 模板（0 LLM 延迟）──
# 匹配常见中文营养查询模式，直接生成精准 SQL
_SQL_TEMPLATES: list[tuple[re.Pattern, str, str]] = [
    # 低热量 / 不想热量太高 / 低卡
    (re.compile(r"(热量|卡路里|calorie|kcal|低卡).*(不|太|高|低)"),
     "低热量食谱",
     "SELECT r.recipe_id, r.name, n.calories, n.fat, n.protein, n.carbs "
     "FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id "
     "WHERE n.calories > 0 AND n.calories < 350 "
     "ORDER BY n.calories ASC LIMIT 15"),
    (re.compile(r"(不|不要太|不想).*(热量|卡路里|高卡|胖)"),
     "低热量食谱",
     "SELECT r.recipe_id, r.name, n.calories, n.fat, n.protein, n.carbs "
     "FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id "
     "WHERE n.calories > 0 AND n.calories < 350 "
     "ORDER BY n.calories ASC LIMIT 15"),
    # 高蛋白
    (re.compile(r"高蛋白|蛋白质"),
     "高蛋白食谱",
     "SELECT r.recipe_id, r.name, n.protein, n.calories, n.fat "
     "FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id "
     "WHERE n.protein > 15 ORDER BY n.protein DESC LIMIT 15"),
    # 低脂
    (re.compile(r"低脂|少油|不油腻|清淡|脂肪.*高|脂肪.*太"),
     "低脂食谱",
     "SELECT r.recipe_id, r.name, n.fat, n.calories, n.protein "
     "FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id "
     "WHERE n.fat > 0 AND n.fat < 10 ORDER BY n.fat ASC LIMIT 15"),
    # 低碳水
    (re.compile(r"低碳|少碳水|少糖|无糖"),
     "低碳水食谱",
     "SELECT r.recipe_id, r.name, n.carbs, n.calories, n.fiber "
     "FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id "
     "WHERE n.carbs > 0 AND n.carbs < 15 ORDER BY n.carbs ASC LIMIT 15"),
]

# ── 菜系识别关键词 ──
_CUISINE_KEYWORDS: dict[str, str] = {
    "美国|美式|american|usa": "American",
    "中国|中式|川菜|粤菜|湘菜|chinese": "Chinese",
    "日本|日式|japanese": "Japanese",
    "印度|印式|indian": "Indian",
    "法国|法式|french": "French",
    "意大利|意式|italian": "Italian",
    "墨西哥|墨式|mexican": "Mexican",
    "泰国|泰式|thai": "Thai",
    "韩国|韩式|korean": "Korean",
}

SQL_TRANSLATION_PROMPT = """You translate nutrition questions into SQLite queries.

Schema:
{TABLE_SCHEMA}

Rules:
- Return ONLY the SQL, no explanation.
- Use UPPER for SQL keywords.
- Use COALESCE(nutrition.col, 0) to avoid NULLs.
- Always JOIN nutrition ON recipes.recipe_id = nutrition.recipe_id.
- Include recipe_id and name in SELECT so the caller can identify results.
- ORDER BY the filtered metric DESC, LIMIT 10.
- Only SELECT — no INSERT/UPDATE/DELETE.

Examples:
Q: 低热量的食谱有哪些？
SQL: SELECT r.recipe_id, r.name, n.calories FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id WHERE n.calories < 300 ORDER BY n.calories ASC LIMIT 10

Q: 高蛋白的食谱推荐
SQL: SELECT r.recipe_id, r.name, n.protein FROM recipes r JOIN nutrition n ON r.recipe_id = n.recipe_id WHERE n.protein > 20 ORDER BY n.protein DESC LIMIT 10

Q: {query}
SQL:"""

ANSWER_PROMPT = """You are a nutrition analysis assistant. Answer the user's question based on the query result below.

User question: {query}

SQL query: {sql}
Results ({row_count} rows):
{rows}

Give a clear summary highlighting the relevant nutrition info."""


class NutritionSQLAgent(BaseAgent):
    """
    营养信息查询 — 三路策略：
    1. 关键词 SQL 模板（0 LLM 延迟，快速命中常见模式）
    2. LLM 翻译 → SQL 执行（复杂查询）
    3. 混合检索兜底（LLM 生成回答）
    """

    def __init__(self):
        super().__init__(name="nutrition_sql", timeout=60.0, max_retries=1)
        self.sandbox = SQLSandbox()
        self.retriever = HybridRetriever()

    async def _execute(self, **kwargs: Any) -> NutritionSQLResult:
        query = kwargs.get("query", "")
        slots = kwargs.get("slots", {})

        # ── 方案 A：关键词 SQL 模板（0 LLM 延迟）──
        sql, label = self._match_template(query, slots)
        if sql:
            success, result = await self.sandbox.execute(sql)
            if success and result:
                answer = self._build_template_answer(label, result, query, slots)
                return NutritionSQLResult(sql=sql, rows=result[:20], answer=answer)

        # ── 方案 B：LLM 翻译 → SQL 执行 ──
        sql, sql_ok = await self._try_sql(query)
        if sql_ok:
            success, result = await self.sandbox.execute(sql)
            if success and result:
                rows_text = self._format_rows(result[:10])
                answer = await self._generate_answer(query, sql, result, rows_text, slots)
                return NutritionSQLResult(sql=sql, rows=result[:20], answer=answer)

        # ── 方案 C：混合检索兜底 ──
        answer = await self._fallback_retrieval(query, slots)

        return NutritionSQLResult(sql=sql or "", rows=[], answer=answer)

    # ── 模板匹配 ──

    @staticmethod
    def _detect_cuisine(query: str, slots: dict | None = None) -> str:
        """从 query 或 slots 中检测菜系关键词，返回 SQL WHERE 片段"""
        # 优先从 slots 取（Router 已提取的菜系）
        if slots and slots.get("cuisine_type"):
            cuisine_en = slots["cuisine_type"]
            return f" AND r.cuisine LIKE '%{cuisine_en}%'"
        # 兜底从 query 中匹配
        q_lower = query.lower()
        for pattern, cuisine_en in _CUISINE_KEYWORDS.items():
            if re.search(pattern, q_lower):
                return f" AND r.cuisine LIKE '%{cuisine_en}%'"
        return ""

    def _match_template(self, query: str, slots: dict | None = None) -> tuple[str, str]:
        """关键词模板匹配（菜系感知），返回 (sql, label)"""
        cuisine_filter = self._detect_cuisine(query, slots)
        for pattern, label, base_sql in _SQL_TEMPLATES:
            if pattern.search(query.lower()):
                sql = base_sql
                if cuisine_filter:
                    sql = base_sql.replace(
                        " ORDER BY ",
                        f"{cuisine_filter} ORDER BY ",
                    )
                return sql, label
        return "", ""

    def _build_template_answer(self, label: str, rows: list[dict[str, Any]], query: str, slots: dict | None = None) -> str:
        """模板命中后直接用数据拼接回答（不需要 LLM）"""
        cuisine_filter = self._detect_cuisine(query, slots)
        title = label
        if cuisine_filter:
            cuisine_name = cuisine_filter.replace(" AND r.cuisine LIKE '%", "").replace("%'", "")
            title = f"{cuisine_name} {label}"
        lines = [f"📊 为你找到以下{title}：\n"]
        for i, r in enumerate(rows[:12], 1):
            name = r.get("name", "?")
            cal = r.get("calories", "?")
            parts = []
            if cal != "?" and cal is not None:
                parts.append(f"{cal:.0f} cal")
            for k, unit in [("protein", "g 蛋白质"), ("fat", "g 脂肪"), ("carbs", "g 碳水"), ("fiber", "g 纤维")]:
                v = r.get(k)
                if v is not None and v != "?":
                    parts.append(f"{v:.1f}{unit}")
            info = " | ".join(parts) if parts else ""
            lines.append(f"{i}. **{name}**  {info}")
        return "\n".join(lines)

    # ── LLM → SQL ──

    async def _try_sql(self, query: str) -> tuple[str, bool]:
        """用 LLM 把自然语言翻译成 SQL"""
        prompt = SQL_TRANSLATION_PROMPT.format(TABLE_SCHEMA=TABLE_SCHEMA, query=query)
        try:
            from src.core.utils.llm import generate_structured
            sql = await generate_structured(
                query=prompt,
                context="",
                system_prompt="You are a SQL generator. Return ONLY valid SQLite SQL.",
                max_new_tokens=256,
                temperature=0.1,
            )
            sql = sql.strip().removeprefix("```sql").removesuffix("```").strip()
            if sql.upper().startswith("SELECT"):
                return sql, True
            return "", False
        except Exception:
            return "", False

    async def _generate_answer(
        self, query: str, sql: str, rows: list[dict[str, Any]], rows_text: str, slots: dict | None = None
    ) -> str:
        """用 LLM 根据 SQL 结果生成自然语言回答"""
        from src.core.utils.llm import generate
        prompt = ANSWER_PROMPT.format(
            query=query, sql=sql, row_count=len(rows), rows=rows_text
        )
        try:
            answer = await generate(
                query=query,
                context=rows_text,
                system_prompt="You are a nutrition analysis assistant. Summarize the data clearly.",
                max_new_tokens=384,
                temperature=0.3,
            )
            return answer
        except Exception:
            # LLM 不可用时直接拼结果
            return self._build_template_answer("营养查询", rows, query, slots)

    # ── 混合检索兜底 ──

    async def _fallback_retrieval(self, query: str, slots: dict | None = None) -> str:
        """混合检索 + LLM 生成"""
        # 构建菜系感知的检索 query
        cuisine_filter = self._detect_cuisine(query, slots)
        retrieval_query = query
        if cuisine_filter:
            cuisine_name = cuisine_filter.replace(" AND r.cuisine LIKE '%", "").replace("%'", "")
            retrieval_query = f"{cuisine_name} {query}"

        chunks = await self.retriever.retrieve(retrieval_query, top_k=5)

        sections = ""
        if chunks:
            sections = "\n\n".join(f"[{c.recipe_id}] {c.content}" for c in chunks)

        try:
            from src.core.utils.llm import generate
            answer = await generate(
                query=query,
                context=sections,
                system_prompt=(
                    "Answer based on the recipe context. "
                    "Highlight nutrition info if available."
                ),
                max_new_tokens=512,
                temperature=0.3,
            )
            return answer
        except Exception:
            return "🔍 未找到精确匹配。试试查询具体菜名？"

    @staticmethod
    def _format_rows(rows: list[dict[str, Any]]) -> str:
        """动态格式化查询结果，只展示结果中实际包含的字段"""
        lines = []

    @staticmethod
    def _format_rows(rows: list[dict[str, Any]]) -> str:
        """动态格式化查询结果，只展示结果中实际包含的字段"""
        lines = []
        for r in rows:
            parts = [f"  - {r.get('name', '?')}"]
            for k in ("calories", "protein", "fat", "carbs", "fiber", "sodium"):
                if k in r:
                    v = r[k]
                    unit = " cal" if k == "calories" else "g" if k in ("protein", "fat", "carbs", "fiber") else ""
                    parts.append(f"{k}={v}{unit}")
            lines.append(", ".join(parts))
        return "\n".join(lines)
