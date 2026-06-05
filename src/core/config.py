"""
Pydantic 配置加载 — 适配 recipe-assistant 领域
参考 multi-agent-ecommerce-system 的 config/settings.py 模式
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class RecipeSettings(BaseSettings):
    # ── 应用基础 ──
    app_name: str = "Multi-Modal Recipe Assistant"
    debug: bool = False

    # ── LLM ──
    llm_api_key: str = ""
    llm_base_url: str = "https://api.minimax.chat/v1"
    llm_model: str = "MiniMax-M1"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ── 检索 ──
    retriever_top_k: int = 10
    retriever_rerank_top_k: int = 5
    retriever_bm25_weight: float = 0.3
    retriever_vector_weight: float = 0.5
    retriever_rerank_weight: float = 0.2

    # ── 向量存储 ──
    vector_store_path: str = "src/data/vector_store"
    embedding_model: str = "BAAI/bge-base-zh-v1.5"
    clip_model: str = "openai/clip-vit-base-patch32"

    # ── SQL 沙箱 ──
    sql_max_rows: int = 100
    sql_query_timeout_ms: int = 3000
    sql_database_url: str = "sqlite:///./recipe.db"

    # ── Agent 超时 ──
    agent_timeout_router: float = 3.0
    agent_timeout_text_rag: float = 8.0
    agent_timeout_image_search: float = 6.0
    agent_timeout_pdf_parse: float = 120.0
    agent_timeout_nutrition_sql: float = 5.0
    agent_timeout_substitution: float = 8.0
    agent_timeout_critic: float = 5.0
    agent_timeout_formatter: float = 3.0

    # ── PDF 解析 ──
    pdf_max_pages: int = 50                                 # 最大解析页数
    pdf_parse_method: str = "auto"                          # auto/basic/enhanced

    # ── 限流 ──
    max_llm_calls_per_request: int = 5
    max_retries_per_agent: int = 2

    # ── Redis ──
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_key_prefix: str = "recipe:session:"
    redis_session_ttl: int = 86400              # 会话 TTL（秒）
    redis_max_context_tokens: int = 2048        # 滑动窗口最大 token 数

    model_config = {"env_file": ".env", "env_prefix": "RECIPE_"}


@lru_cache()
def get_settings() -> RecipeSettings:
    return RecipeSettings()
