"""
LangGraph 状态图 — 多轮对话 + Planner + Critic 重试循环 + Reflection 质检

   [start] → load_memory → router → {planner? planner → worker : worker}
                                                          ↓ (并行: worker + fallback)
           → critic → {passed: reflection → save_memory → formatter → [end]}
                      {failed & retry<max: revision → worker (retry with feedback)}
                      {failed & retry>=max: reflection → save_memory → formatter → [end]}

模式说明:
  - Planner: 对"规划一周菜单"等复杂请求，先分解为子步骤再逐一执行
  - Reflection: LLM 驱动的深度质检（比规则 Critic 更深层语义审查），critic 通过后执行
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.agents import (
    CriticAgent,
    FormatterAgent,
    ImageSearchAgent,
    NutritionSQLAgent,
    PDFParseAgent,
    PlannerAgent,
    ReflectionAgent,
    RouterAgent,
    SubstitutionAgent,
    TextRAGAgent,
)
from src.api.schemas import (
    Chunk,
    CriticResult,
    FormatterResult,
    IntentType,
    PDFParseResult,
    PlannerResult,
    ProvenanceItem,
    ReflectionResult,
    RouterResult,
    TextRAGResult,
)
from src.core.memory import ConversationMemory
from src.core.long_term_memory import get_long_term_memory

# ── 需要走 Planner 的关键词（意图匹配后二次检测） ──
_PLANNER_KEYWORDS = (
    "规划", "计划", "安排", "设计",
    "一周", "每天", "每日", "三餐",
    "菜单", "套餐", "搭配", "组合",
    "plan", "weekly", "daily", "menu", "meal prep",
)


class RecipePipelineState(TypedDict, total=False):
    """LangGraph 状态定义"""
    # 输入
    request_id: str
    query: str
    session_id: str
    user_id: str               # 用户标识（用于长效记忆）
    images: list[str]
    files: list[str]            # 文件列表（base64 编码的 PDF 等）
    intent_hint: str | None

    # 路由
    intent: str
    router_confidence: float
    slots: dict[str, Any]

    # 对话记忆
    conversation_context: str       # 历史摘要，传给 Router
    last_slots: dict[str, Any]     # 上一轮槽位
    user_preference_context: str    # 用户偏好上下文（长效记忆）

    # 执行结果
    draft: str
    provenance: list[dict[str, Any]]
    need_disclaimer: bool

    # 兜底缓存（预计算，Worker 失败时瞬间切换）
    fallback_draft: str
    fallback_chunks: list[dict[str, Any]]

    # Planner 规划
    use_planner: bool                # 是否启动 Planner
    plan_goal: str                   # 规划目标
    plan_steps: list[str]            # 规划步骤列表
    plan_summary: str                # Planner 执行汇总
    plan_executed: bool              # Planner 是否已执行

    # 质检
    critic_passed: bool
    critic_reasons: list[str]
    critic_suggestions: list[str]

    # Reflection 反思（LLM 深层质检）
    reflection_passed: bool
    reflection_critique: str
    reflection_revised: str

    # 重试循环控制
    retry_count: int
    max_retries: int

    # 最终输出
    final_response: str

    # 追踪
    agent_results: dict[str, Any]
    total_latency_ms: float
    _start_time: float


# ══════════════════════════════════════════
# 节点函数
# ══════════════════════════════════════════

async def init_node(state: RecipePipelineState) -> dict:
    """初始化：生成 request_id，记录开始时间"""
    return {
        "request_id": str(uuid.uuid4()),
        "_start_time": time.perf_counter(),
        "agent_results": {},
        "provenance": [],
        "retry_count": 0,
        "max_retries": 2,
        "need_disclaimer": False,
        "fallback_draft": "",
        "fallback_chunks": [],
        "draft": "",
        "final_response": "",
    }


async def load_memory_node(state: RecipePipelineState) -> dict:
    """加载对话记忆 + 长效记忆（用户偏好），构造上下文传给 Router"""
    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "")

    memory = await ConversationMemory.get_or_create(session_id)
    conversation_context = await memory.get_context(max_tokens=2048)

    # 加载长效记忆（用户偏好），注入 Router 上下文
    user_preference_context = ""
    if user_id:
        try:
            ltm = get_long_term_memory()
            user_preference_context = await ltm.get_preference_context(user_id)
        except Exception as e:
            import structlog
            structlog.get_logger().warning(
                "long_term_memory.load_failed", user_id=user_id, error=str(e)
            )

    # 将用户偏好拼接到对话上下文中
    if user_preference_context:
        conversation_context = (
            f"{conversation_context}\n[{user_preference_context}]"
            if conversation_context
            else f"[{user_preference_context}]"
        )

    return {
        "conversation_context": conversation_context,
        "last_slots": await memory.get_last_slots(),
        "user_preference_context": user_preference_context,
    }


async def router_node(state: RecipePipelineState) -> dict:
    """意图识别 — 带多轮上下文"""
    agent = RouterAgent()
    result: RouterResult = await agent.run(
        query=state["query"],
        conversation_context=state.get("conversation_context", ""),
        last_slots=state.get("last_slots", {}),
        files=state.get("files", []),
    )
    return {
        "intent": result.intent.value,
        "router_confidence": result.confidence,
        "slots": result.slots,
    }


# ── 兜底检索（无 LLM） ──

async def _prepare_text_fallback(query: str, top_k: int = 3) -> tuple[str, list[dict]]:
    """纯检索兜底，与 Worker 并行执行。"""
    try:
        from src.core.retrievers.hybrid import HybridRetriever
        # 全局共享实例，避免重复加载模型
        if not hasattr(_prepare_text_fallback, "_shared_retriever"):
            _prepare_text_fallback._shared_retriever = HybridRetriever()
        retriever = _prepare_text_fallback._shared_retriever
        chunks: list[Chunk] = await retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return "抱歉，没有找到相关食谱信息。", []

        seen: set[str] = set()
        lines: list[str] = [f"🔍 找到 {len(chunks)} 个相关食谱片段："]
        for c in chunks:
            rid = c.recipe_id
            if rid not in seen:
                seen.add(rid)
                lines.append(f"\n📖 {rid}:")
            lines.append(f"   • {c.content[:150]}...")

        fallback_chunks = [
            {"chunk_id": c.chunk_id, "recipe_id": c.recipe_id,
             "score": c.score, "source": "recipes.json", "snippet": c.content[:100]}
            for c in chunks
        ]
        return "\n".join(lines), fallback_chunks
    except Exception as e:
        import structlog
        structlog.get_logger().warning("fallback.prepare_failed", error=str(e))
        return "抱歉，暂时无法获取食谱信息，请稍后再试。", []


# ── Worker 节点（统一派发，支持重试） ──

async def worker_node(state: RecipePipelineState) -> dict:
    """
    统一 Worker 节点 — 根据 state["intent"] 派发到对应 Agent。
    如果是重试（retry_count > 0），将 Critic 反馈传入 Agent。
    注意：对于 NutritionSQL/Substitution（模板快速命中场景），
    先跑 Worker，成功则直接返回，不等待 Fallback 检索。
    对于 TextRAG（LLM 耗时较长），Worker 与 Fallback 并行执行。
    """
    intent = state["intent"]
    query = state["query"]
    is_retry = state.get("retry_count", 0) > 0

    # ── 构建 extra_kwargs（重试时传入 critic 反馈） ──
    extra_kwargs: dict[str, Any] = {}
    if is_retry:
        suggestions = state.get("critic_suggestions", [])
        extra_kwargs["feedback"] = suggestions
        extra_kwargs["retry_count"] = state["retry_count"]

    # ── 默认返回值 ──
    result_data: dict[str, Any] = {}
    worker_result_model: Any = None

    # ── 根据 intent 派发 ──
    if intent in (IntentType.INGREDIENT_RECOMMEND.value, IntentType.STEP_QA.value):
        agent = TextRAGAgent()
        fallback_task = asyncio.create_task(_prepare_text_fallback(query))
        worker_task = asyncio.create_task(
            agent.run(query=query, slots=state.get("slots", {}), intent=intent, **extra_kwargs)
        )
        worker_result_model, (fb_draft, fb_chunks) = await asyncio.gather(worker_task, fallback_task)

        result_data["fallback_draft"] = fb_draft
        result_data["fallback_chunks"] = fb_chunks

        if worker_result_model.success:
            result_data["draft"] = worker_result_model.answer
            result_data["provenance"] = [
                {"chunk_id": c.chunk_id, "recipe_id": c.recipe_id,
                 "score": c.score, "source": "recipes.json", "snippet": c.content[:100]}
                for c in worker_result_model.chunks
            ]
        else:
            result_data["draft"] = fb_draft
            result_data["provenance"] = fb_chunks

    elif intent == IntentType.NUTRITION_FILTER.value:
        agent = NutritionSQLAgent()
        worker_result_model = await agent.run(query=query, slots=state.get("slots", {}), **extra_kwargs)
        result_data["need_disclaimer"] = True

        if worker_result_model.success:
            result_data["draft"] = worker_result_model.answer
        else:
            # Worker 失败时才跑兜底检索
            fb_draft, fb_chunks = await _prepare_text_fallback(query)
            result_data["draft"] = fb_draft
            result_data["provenance"] = fb_chunks
            result_data["fallback_draft"] = fb_draft
            result_data["fallback_chunks"] = fb_chunks

    elif intent == IntentType.SUBSTITUTION.value:
        agent = SubstitutionAgent()
        worker_result_model = await agent.run(query=query, **extra_kwargs)
        result_data["need_disclaimer"] = True

        if worker_result_model.success:
            result_data["draft"] = worker_result_model.explanation
        else:
            # Worker 失败时才跑兜底检索
            fb_draft, fb_chunks = await _prepare_text_fallback(query)
            result_data["draft"] = fb_draft
            result_data["provenance"] = fb_chunks
            result_data["fallback_draft"] = fb_draft
            result_data["fallback_chunks"] = fb_chunks

    elif intent == IntentType.IMAGE_SEARCH.value:
        agent = ImageSearchAgent()
        has_images = bool(state.get("images", []))
        worker_result_model = await agent.run(query=query, images=state.get("images", []))

        recipes = worker_result_model.recipes
        if recipes:
            lines = []
            for i, r in enumerate(recipes[:5], 1):
                lines.append(f"{i}. **{r.name}**")
            top_names = "、".join(r.name for r in recipes[:3])
            if has_images:
                result_data["draft"] = (
                    f"📷 上传的图片看起来像是「{top_names}」之一。\n\n"
                    f"匹配结果:\n" + "\n".join(lines) + "\n\n"
                    "想了解哪道菜的做法？告诉我菜名，我教你具体步骤！"
                )
            else:
                result_data["draft"] = (
                    f"🔍 根据「{query}」找到以下菜谱:\n" + "\n".join(lines)
                )
            result_data["provenance"] = [
                {"chunk_id": f"clip_{r.recipe_id}", "recipe_id": r.recipe_id,
                 "score": 1.0 / (i + 1), "source": "clip_retrieval",
                 "snippet": r.name}
                for i, r in enumerate(recipes)
            ]
        else:
            result_data["draft"] = "未找到匹配的菜谱，请试试其他关键词或上传更清晰的食物照片。"
        result_data["need_disclaimer"] = False

    elif intent == IntentType.PDF_PARSE.value:
        agent = PDFParseAgent()
        pdf_base64 = ""
        # 优先从 state["files"] 取，其次从 slots 取
        files = state.get("files", [])
        if files:
            pdf_base64 = files[0]
        if not pdf_base64:
            pdf_base64 = state.get("slots", {}).get("pdf", "")

        worker_result_model: PDFParseResult = await agent.run(
            pdf=pdf_base64,
            filename=state.get("slots", {}).get("filename", "document.pdf"),
        )

        if worker_result_model.success:
            text = worker_result_model.text
            preview = worker_result_model.data.get("preview", "")
            pages = worker_result_model.pages
            result_data["draft"] = (
                f"📄 **PDF 解析完成** ({pages} 页, 方法: {worker_result_model.method})\n\n"
                f"{preview}\n\n"
                f"... (共 {len(text)} 字符，完整内容可在会话中继续查询)"
                if len(text) > 500 else text
            )
            result_data["provenance"] = [{
                "chunk_id": f"pdf_{worker_result_model.data.get('filename', 'doc')}",
                "score": 1.0, "source": "pdf_parse",
                "snippet": preview[:200],
            }]
        else:
            result_data["draft"] = f"❌ PDF 解析失败: {worker_result_model.error}"
        result_data["need_disclaimer"] = False

    else:
        result_data["draft"] = "你好！我是食谱助手，可以帮你推荐菜谱、查询营养信息、替换食材等。请问有什么可以帮你的？"
    return result_data


async def critic_node(state: RecipePipelineState) -> dict:
    """Critic 质检"""
    agent = CriticAgent()
    result: CriticResult = await agent.run(
        draft=state.get("draft", ""),
        provenance=state.get("provenance", []),
        intent=state.get("intent", ""),
        need_disclaimer=state.get("need_disclaimer", False),
    )
    return {
        "critic_passed": result.passed,
        "critic_reasons": [r.value for r in result.reasons],
        "critic_suggestions": result.suggestions,
    }


async def revision_node(state: RecipePipelineState) -> dict:
    """重试准备：自增 retry_count，Critic 反馈已在 state 中"""
    return {"retry_count": state.get("retry_count", 0) + 1}


async def save_memory_node(state: RecipePipelineState) -> dict:
    """保存本轮对话到记忆 + 长效记忆（用户偏好），后台启动 LLM 压缩"""
    session_id = state.get("session_id", "default")
    memory = await ConversationMemory.get_or_create(session_id)
    await memory.add_turn(
        query=state["query"],
        intent=state.get("intent", ""),
        slots=state.get("slots", {}),
        answer=state.get("draft", ""),
        provenance=[],
    )
    # 后台触发 LLM 压缩（不阻塞响应），下次请求时摘要已就绪
    asyncio.create_task(memory.compress_if_needed())

    # 保存到长效记忆（后台，不阻塞）
    user_id = state.get("user_id", "")
    if user_id:
        asyncio.create_task(_save_long_term_memory(state, user_id))

    return {}


async def _save_long_term_memory(state: RecipePipelineState, user_id: str):
    """后台保存长效记忆（最近查询 + 偏好提取）"""
    try:
        ltm = get_long_term_memory()
        # 记录最近查询
        await ltm.add_recent_query(user_id, state["query"])
        # 自动提取偏好
        await ltm.extract_and_save_preferences(
            user_id=user_id,
            query=state["query"],
            intent=state.get("intent", ""),
            answer=state.get("draft", ""),
        )
    except Exception as e:
        import structlog
        structlog.get_logger().warning(
            "long_term_memory.save_failed", user_id=user_id, error=str(e)
        )


async def formatter_node(state: RecipePipelineState) -> dict:
    """格式化输出"""
    agent = FormatterAgent()
    result: FormatterResult = await agent.run(
        draft=state.get("draft", ""),
        provenance=state.get("provenance", []),
        need_disclaimer=state.get("need_disclaimer", False),
    )
    return {
        "final_response": result.final_response,
        "total_latency_ms": (time.perf_counter() - state["_start_time"]) * 1000,
    }


# ── Planner 规划节点 ──

async def planner_node(state: RecipePipelineState) -> dict:
    """将复杂请求分解为子步骤并依次执行，结果写入 draft 供后续 Worker 使用。"""
    query = state["query"]
    user_preferences = state.get("user_preference_context", "")

    agent = PlannerAgent()
    result: PlannerResult = await agent.run(
        query=query,
        context=state.get("conversation_context", ""),
        user_preferences=user_preferences,
    )

    return {
        "use_planner": True,
        "plan_goal": result.goal,
        "plan_steps": result.steps,
        "plan_summary": result.summary,
        "plan_executed": True,
        # 将 Planner 汇总设为 draft（后续 Worker 可在此基础上细化）
        "draft": result.summary if result.success else "",
        "agent_results": {
            "planner": {
                "goal": result.goal,
                "steps": result.steps,
                "step_count": len(result.steps),
                "success": result.success,
            }
        },
    }


# ── Reflection 反思质检节点（LLM 驱动的深层审查） ──

async def reflection_node(state: RecipePipelineState) -> dict:
    """对 draft 进行 LLM 驱动的深度质检，必要时修订文本。"""
    draft = state.get("draft", "")
    if not draft:
        return {"reflection_passed": True}

    agent = ReflectionAgent()
    result: ReflectionResult = await agent.run(
        draft=draft,
        query=state.get("query", ""),
        context=state.get("conversation_context", ""),
        provenance=state.get("provenance", []),
    )

    # 如果修订版有改进，使用修订版
    revised = result.revised
    if revised and revised != draft and len(revised) > len(draft) * 0.5:
        draft = revised

    return {
        "reflection_passed": result.passed,
        "reflection_critique": result.critique,
        "reflection_revised": revised,
        "draft": draft,
    }


# ══════════════════════════════════════════
# 条件边
# ══════════════════════════════════════════

def _needs_planning(query: str, intent: str) -> bool:
    """检测是否需启动 Planner（关键词匹配 + 特定意图）。"""
    if intent == IntentType.CHITCHAT.value:
        return False
    q_lower = query.lower()
    return any(kw in q_lower for kw in _PLANNER_KEYWORDS)


def route_by_intent(
    state: RecipePipelineState,
) -> Literal["worker", "planner", "chitchat_direct"]:
    """Router → Worker/Planner/闲聊直通 路由"""
    intent = state.get("intent", "chitchat")
    if intent == IntentType.CHITCHAT.value:
        return "chitchat_direct"
    # 检测是否需要规划
    if _needs_planning(state.get("query", ""), intent):
        return "planner"
    return "worker"


def route_after_planner(
    state: RecipePipelineState,
) -> Literal["worker", "critic"]:
    """Planner → 下一步：有步骤则走 Worker 细化，否则直接进 Critic"""
    if state.get("plan_executed") and state.get("plan_steps"):
        return "worker"
    return "critic"


def after_critic(state: RecipePipelineState) -> Literal["reflection", "revision"]:
    """Critic → 下一步：通过则进 Reflection 深度质检，不通过则重试"""
    import structlog
    logger = structlog.get_logger()
    critic_passed = state.get("critic_passed", True)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    logger.info(
        "routing.after_critic",
        critic_passed=critic_passed,
        retry_count=retry_count,
        max_retries=max_retries,
        reasons=state.get("critic_reasons", []),
    )
    if critic_passed:
        return "reflection"
    if retry_count < max_retries:
        return "revision"
    return "reflection"  # 已达最大重试次数，走 Reflection 尝试最后修订


async def chitchat_direct_node(state: RecipePipelineState) -> dict:
    """闲聊直接回复，跳过 Critic 和 Formatter"""
    draft = "你好！我是食谱助手，可以帮你推荐菜谱、查询营养信息、替换食材等。请问有什么可以帮你的？"
    return {
        "draft": draft,
        "final_response": draft,
        "total_latency_ms": (time.perf_counter() - state["_start_time"]) * 1000,
    }


# ══════════════════════════════════════════
# 构建图
# ══════════════════════════════════════════

def build_recipe_graph() -> StateGraph:
    workflow = StateGraph(RecipePipelineState)

    # 注册节点
    workflow.add_node("init", init_node)
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("chitchat_direct", chitchat_direct_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("save_memory", save_memory_node)
    workflow.add_node("formatter", formatter_node)

    # ── 主流程 ──
    workflow.set_entry_point("init")
    workflow.add_edge("init", "load_memory")
    workflow.add_edge("load_memory", "router")

    # Router → Worker / Planner / 闲聊
    workflow.add_conditional_edges("router", route_by_intent, {
        "worker": "worker",
        "planner": "planner",
        "chitchat_direct": "chitchat_direct",
    })

    # Planner → Worker（细化执行）或 Critic（已直接生成）
    workflow.add_conditional_edges("planner", route_after_planner, {
        "worker": "worker",
        "critic": "critic",
    })

    # Worker → Critic（闲聊不走 Critic）
    workflow.add_edge("worker", "critic")

    # Critic 质检结果分支
    workflow.add_conditional_edges("critic", after_critic, {
        "reflection": "reflection",
        "revision": "revision",
    })

    # 重试循环：revision → worker（走回统一 Worker 节点）
    workflow.add_edge("revision", "worker")

    # Reflection → save_memory（LLM 深度质检后保存）
    workflow.add_edge("reflection", "save_memory")

    # 通过质检后：save_memory → formatter → END
    workflow.add_edge("save_memory", "formatter")
    workflow.add_edge("formatter", END)

    # 闲聊直通 END
    workflow.add_edge("chitchat_direct", END)

    return workflow.compile()
