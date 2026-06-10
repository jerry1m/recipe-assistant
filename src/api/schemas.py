"""
Pydantic 请求/响应模型
参考 multi-agent-ecommerce-system 的 models/schemas.py 模式
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── 枚举 ──

class IntentType(str, Enum):
    INGREDIENT_RECOMMEND = "ingredient_recommend"       # 食材推荐
    IMAGE_SEARCH = "image_search"                       # 图片搜索
    PDF_PARSE = "pdf_parse"                               # PDF 文档解析
    NUTRITION_FILTER = "nutrition_filter"               # 营养过滤
    STEP_QA = "step_qa"                                 # 步骤问答
    SUBSTITUTION = "substitution"                       # 替换推理
    CHITCHAT = "chitchat"                               # 闲聊


class CritiqueReason(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    HALLUCINATION = "hallucination"
    UNSAFE_ADVICE = "unsafe_advice"
    INCOMPLETE = "incomplete"
    CONTRADICTION = "contradiction"
    TIMEOUT = "timeout"


# ── 数据模型 ──

class Ingredient(BaseModel):
    name: str
    amount: str = ""
    unit: str = ""
    alternative: list[str] = Field(default_factory=list)


class NutritionInfo(BaseModel):
    calories: float = 0.0
    protein: float = 0.0
    fat: float = 0.0
    carbs: float = 0.0
    fiber: float = 0.0
    sodium: float = 0.0


class Recipe(BaseModel):
    recipe_id: str
    name: str
    cuisine: str = ""
    difficulty: str = ""
    prep_time: str = ""
    cook_time: str = ""
    servings: int = 0
    ingredients: list[Ingredient] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    nutrition: NutritionInfo = Field(default_factory=NutritionInfo)
    tags: list[str] = Field(default_factory=list)
    image_url: str = ""
    source: str = ""


class Chunk(BaseModel):
    """检索返回的文本块"""
    chunk_id: str
    recipe_id: str
    content: str
    section: str = ""           # ingredients / steps / nutrition / notes
    score: float = 0.0


class ProvenanceItem(BaseModel):
    """溯源信息：答案引用的来源"""
    chunk_id: str
    recipe_id: str = ""
    score: float
    source: str                  # 文件/数据库来源标识
    snippet: str                 # 原文片段


# ── Agent 结果 ──

class AgentResult(BaseModel):
    """所有 Agent 的统一返回基类"""
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class RouterResult(AgentResult):
    agent_name: str = "router"
    intent: IntentType = IntentType.CHITCHAT
    slots: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class TextRAGResult(AgentResult):
    agent_name: str = "text_rag"
    chunks: list[Chunk] = Field(default_factory=list)
    answer: str = ""


class ImageSearchResult(AgentResult):
    agent_name: str = "image_search"
    recipes: list[Recipe] = Field(default_factory=list)


class PDFParseResult(AgentResult):
    """PDF 解析结果"""
    agent_name: str = "pdf_parse"
    text: str = ""                                          # 解析后的全文
    pages: int = 0                                           # 页数
    method: str = ""                                         # 使用的方法
    metadata: dict[str, Any] = Field(default_factory=dict)   # PDF 元数据


class NutritionSQLResult(AgentResult):
    agent_name: str = "nutrition_sql"
    sql: str = ""
    rows: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""


class SubstitutionResult(AgentResult):
    agent_name: str = "substitution"
    substitutions: list[dict[str, str]] = Field(default_factory=list)
    explanation: str = ""


class CriticResult(AgentResult):
    agent_name: str = "critic"
    passed: bool = True
    reasons: list[CritiqueReason] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class FormatterResult(AgentResult):
    agent_name: str = "formatter"
    final_response: str = ""
    disclaimer: str = ""


class PlannerResult(AgentResult):
    """规划 Agent 执行结果"""
    agent_name: str = "planner"
    goal: str = ""
    steps: list[str] = Field(default_factory=list)
    step_outputs: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class ReflectionResult(AgentResult):
    """反思质检结果"""
    agent_name: str = "reflection"
    passed: bool = True
    critique: str = ""
    revised: str = ""
    reasons: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


# ── API 请求/响应 ──

class TurnRecord(BaseModel):
    """单轮对话记录"""
    query: str
    intent: str = ""
    slots: dict[str, Any] = Field(default_factory=dict)
    answer: str = ""
    provenance: list[ProvenanceItem] = Field(default_factory=list)
    timestamp: float = 0.0


class AskRequest(BaseModel):
    query: str
    images: list[str] = Field(default_factory=list)      # base64 图片列表，可选
    files: list[str] = Field(default_factory=list)       # base64 文件列表（PDF 等），可选
    intent_hint: str | None = None                       # 可选意图提示（跳过 Router）
    user_id: str | None = None                           # 用于限流/个性化
    session_id: str | None = None                        # 多轮对话会话 ID（不传则自动生成）
    stream: bool = True                                  # 是否 SSE 流式
    max_tokens: int = 2048
    temperature: float = 0.3


class AskResponse(BaseModel):
    answer: str
    provenance: list[ProvenanceItem] = Field(default_factory=list)
    intent: str = ""
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    disclaimer: str | None = None


class StreamingEvent(BaseModel):
    """SSE 事件"""
    event: str                               # token / provenance / done / error
    data: dict[str, Any] = Field(default_factory=dict)
