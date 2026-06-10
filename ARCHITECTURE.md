# 🏗️ 多模态智能食谱助手 — 架构设计文档

> **版本**: 0.1.0 | **更新**: 2026.06

---

## 目录

1. [系统概述](#1-系统概述)
2. [整体架构图](#2-整体架构图)
3. [核心组件详解](#3-核心组件详解)
4. [数据流](#4-数据流)
5. [多轮对话机制](#5-多轮对话机制)
6. [容错与降级](#6-容错与降级)
7. [性能优化](#7-性能优化)
8. [部署架构](#8-部署架构)
9. [技术选型](#9-技术选型)

---

## 1. 系统概述

### 1.1 项目定位

轻量级 **Multi-Agent RAG** 食谱问答系统，面向 5000 条真实食谱提供智能问答服务。个人项目，旨在实践 Multi-Agent 编排、混合检索、Function Calling、LLM 容错等工程能力。

### 1.2 核心能力

| 能力 | 说明 |
|------|------|
| **意图识别** | 7 类意图 + 槽位提取（规则快速通道 + Function Calling 快慢车道 4 态合并） |
| **食谱问答** | BM25 + FAISS + BGE-Reranker 三级检索 + LLM 生成 |
| **营养查询** | 关键词 SQL 模板 → LLM 翻译 SQL → 混合检索三路兜底 |
| **图文检索** | CLIP 零样本以图搜菜名 |
| **食材替换** | LLM 链式推理 + JSON 结构化输出 |
| **PDF 解析** | 双层引擎（PyMuPDF + magic-pdf），自动降级 |
| **多轮对话** | Redis 持久化 + Token 感知滑动窗口 + LLM 压缩 |
| **质量质检** | Critic 重试循环 + Reflection Agent LLM 深度质检 |
| **熔断保护** | 异步三态 Circuit Breaker（CLOSED/OPEN/HALF_OPEN）|
| **长效记忆** | LongTermMemory JSON 文件持久化（跨会话用户偏好）|
| **工具注册** | ToolRegistry MCP 兼容目录 + 自动发现 |
| **规划执行** | Planner Agent 复杂请求分解 → 逐步调工具 → LLM 汇总 |

### 1.3 设计原则

1. **API 优先 + 本地回退** — 绝不因 LLM 不可用而中断服务
2. **Worker + Fallback 并行** — 兜底检索与 LLM 调用同时发起，零额外延迟
3. **容错降级** — 每个环节都有 fallback，失败不崩溃
4. **全局单例** — 模型加载一次，全局共享
5. **免责声明单一来源** — Formatter 统一添加，避免 Critic 误判重试

---

## 2. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        客户端 (Client)                               │
│  curl / Postman / Frontend SPA / Python SDK                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ POST /ask  (JSON + base64 files/images)
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI 层 (src/api/)                                              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────────┐ │
│  │ main.py  │  │ streaming.py │  │ schemas.py (Pydantic 模型)    │ │
│  │ 路由入口  │  │ SSE 流式输出  │  │ AskRequest / AskResponse     │ │
│  └────┬─────┘  └──────────────┘  └───────────────────────────────┘ │
└───────┼─────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  编排层 (src/orchestrator/) — LangGraph StateGraph (12 节点)        │
│                                                                      │
│  ┌──────────┐                                                        │
│  │  init    │ ← 初始化 state: request_id, start_time                │
│  └────┬─────┘                                                        │
│       ▼                                                              │
│  ┌─────────────┐                                                     │
│  │ load_memory │ ← 加载对话历史 + LongTermMemory.load(user_id)      │
│  └────┬────────┘                      ┌───────────────────────┐     │
│       ▼                                │  ConversationMemory   │     │
│  ┌──────────┐                          │  ┌─────────────────┐  │     │
│  │  Router  │── Intent + 槽位 ──────→  │  │ Redis (主存储)   │  │     │
│  └────┬─────┘  快慢车道 4 态合并       │  ├─────────────────┤  │     │
│       │                                │  │ 内存兜底 (备用)  │  │     │
│       ├──→ Planner? 检测规划关键词      │  └─────────────────┘  │     │
│       │    │                           └───────────────────────┘     │
│       │    ▼                                                         │
│       │  ┌──────────┐   ToolRegistry                                 │
│       │  │ Planner  │──→ search_recipes / search_nutrition / etc.    │
│       │  │ Agent    │──→ 分解 → 逐步执行 → LLM 汇总                  │
│       │  └────┬─────┘                                                │
│       └───────┤                                                      │
│               ▼ (按 intent 条件派发)                                  │
│  ┌──────────┐                                                        │
│  │  worker  │ ← 统一派发: TextRAG/NutritionSQL/Substitution/...     │
│  └────┬─────┘    asyncio.gather(Worker, Fallback)                    │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────┐                                                        │
│  │  critic  │ ← 质检: 完整性+合规+事实                              │
│  └────┬─────┘                                                        │
│       │                                                              │
│   ┌────┴────┐                                                       │
│   │         │                                                        │
│  通过      失败 (retry < 2)                                          │
│   │         │                                                        │
│   │    ┌──────────┐                                                  │
│   │    │ revision │ ← retry_count+1, 注入 critic_suggestions        │
│   │    └────┬─────┘  └─→ worker (回边)                              │
│   │         │                                                        │
│   │    retry ≥ 2 → 走预计算兜底                                     │
│   │                                                                  │
│   ▼                                                                  │
│  ┌────────────┐                                                      │
│  │ Reflection │ ← LLM 深度质检 (事实/完整/安全/可操作)              │
│  │  Agent     │ → 需要修订则自动修正, 无问题通过                     │
│  └────┬───────┘                                                      │
│       ▼                                                              │
│  ┌─────────────┐                                                     │
│  │ save_memory │ ← 保存对话 + LongTermMemory.save() + 后台压缩      │
│  └────┬────────┘                                                     │
│       ▼                                                              │
│  ┌───────────┐                                                       │
│  │ formatter │ ← 添加引用标注 + 免责声明                             │
│  └────┬──────┘                                                       │
│       ▼                                                              │
│     [END]  → final_response                                          │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Agent 层 (src/agents/) — 10 个 Agent 各司其职                      │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐      │
│  │ Router   │  │ Planner  │  │ Reflection  │  │ TextRAG      │      │
│  │ 意图识别  │  │ 规划执行  │  │ 深度质检    │  │ 食谱问答     │      │
│  └──────────┘  └──────────┘  └────────────┘  └──────────────┘      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Nutrition │  │ImageSearch│  │Critic    │  │Formatter │            │
│  │SQL       │  │ 图文检索  │  │ 质量质检  │  │ 输出格式化│            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│  ┌──────────┐  ┌──────────┐                                         │
│  │Substitut.│  │PDFParse  │                                         │
│  │ 食材替换  │  │PDF解析   │                                         │
│  └──────────┘  └──────────┘                                         │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  核心层 (src/core/) — 共享基础设施                                   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  检索系统 (retrievers/)                                      │    │
│  │  ┌────────────┐  ┌────────────────┐  ┌──────────────────┐  │    │
│  │  │ Hybrid     │  │ CLIPRetriever  │  │ BGE-Reranker    │  │    │
│  │  │ BM25+FAISS │  │ 以图搜菜名      │  │ CrossEncoder    │  │    │
│  │  └────────────┘  └────────────────┘  └──────────────────┘  │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  LLM 系统 (utils/llm.py) — API 优先 + Circuit Breaker + 本地回退 │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐    │    │
│  │  │ ModelScope API   │  │ Qwen2.5-0.5B (本地, 全局单例) │    │    │
│  │  │ Qwen3.5-35B      │  │ ~1.5GB 显存, lazy load       │    │    │
│  │  │ ← Circuit Breaker│  │ ← API OPEN 时自动降级         │    │    │
│  │  └──────────────────┘  └──────────────────────────────┘    │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  安全沙箱 (sql/) — sqlglot AST 校验 + SQLite 只读执行       │    │
│  │  ┌──────────────┐  ┌──────────────────────────────────┐    │    │
│  │  │ validator.py │  │ sandbox.py (超时熔断 3s)         │    │    │
│  │  └──────────────┘  └──────────────────────────────────┘    │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  工具注册中心 (tools/) — MCP 兼容目录                        │    │
│  │  ┌──────────────┐  ┌──────────────────────────────────┐    │    │
│  │  │ registry.py  │  │ retrieval_tools.py               │    │    │
│  │  │ ToolRegistry │  │ search_recipes / nutrition / sub │    │    │
│  │  └──────────────┘  └──────────────────────────────────┘    │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  PDF 解析 (pdf_parser.py) — PyMuPDF + magic-pdf 双层      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  对话记忆 (memory.py) — Redis + Token 滑动窗口 + LLM 压缩  │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  长效记忆 (long_term_memory.py) — JSON 文件持久化, 零依赖  │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  监控 (utils/metrics.py) — Agent + Circuit Breaker 指标    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件详解

### 3.1 Router — 三阶段意图识别 + 快慢车道 4 态合并

```
query → 规则快速通道 (0ms, 正则匹配)
           │
   置信度 ≥ 0.9 ──→ fast_lane_only (直接返回, 跳过 FC)
           │
        < 0.9
           ↓
    Function Calling (Qwen3.5-35B, ~2s)  慢通道
           │
           ↓
    4 态合并策略:
    ┌─────────────────────────────────────────────────────┐
    │ rule_intent == fc_intent  → agree (boost +0.1)     │
    │ rule_intent ≠ fc_intent                            │
    │   ├ rule_confidence 更高 → conflict_prefer_rule    │
    │   └ fc_confidence 更高  → conflict_prefer_fc       │
    │ FC 完全失败            → fc_failed_fallback_to_rule│
    └─────────────────────────────────────────────────────┘
```

**7 类意图映射**：

| Intent | Function Name | 触发场景 |
|--------|---------------|----------|
| `ingredient_recommend` | 推荐菜谱/怎么做某道菜 | "推荐几道菜"、"红烧肉怎么做" |
| `step_qa` | 烹饪步骤/时间/火候 | "蒸鱼要多久"、"烤箱温度" |
| `nutrition_filter` | 营养查询 | "热量多少"、"高蛋白食谱" |
| `substitution` | 食材替换 | "没有鸡蛋怎么办" |
| `image_search` | 图片检索 | "红烧肉的照片" |
| `pdf_parse` | PDF 文档解析 | 上传 PDF 文件 |
| `chitchat` | 闲聊/无关话题 | "你好"、"天气" |

**关键代码**：`src/agents/router.py` — `_INTENT_TOOLS` 定义 7 个 OpenAPI Function, `_merge_results()` 实现 4 态合并

### 3.2 TextRAG — 混合检索 + LLM 生成

```
用户 query (中文)
      │
      ▼
  translate_query() → 英文检索 query
      │
      ▼
  HybridRetriever.retrieve(query, top_k=5/10)
      │
      ├── BM25 (jieba 中文分词或 whitespace)
      ├── FAISS (all-MiniLM-L6-v2, 384-dim, IndexFlatIP)
      └── RRF 融合 (K=60) → BGE-Reranker 精排
      │
      ▼
  LLM generate(query, context, system_prompt)
      │
      ▼
  最终回答
```

**推荐场景优化**：
- 从 `slots.cuisine_type` 提取菜系关键词构造检索 query
- `top_k` 翻倍 (5 → 10)
- 使用专用 `_RECOMMEND_SYSTEM_PROMPT`

### 3.3 NutritionSQL — 三路策略

```
query → 关键词模板匹配 (0 LLM 延迟)
         ├── 匹配 → 菜系感知 SQL 注入 → 直接返回 (12ms)
         └── 不匹配 → LLM 翻译 SQL → SQLSandbox 执行
                       ├── 成功 → LLM 总结回答
                       └── 失败 → 混合检索兜底 + LLM 生成
```

**5 个预编译 SQL 模板**：低热量、高蛋白、低脂、低碳水、通用

**安全机制**：sqlglot AST 校验 → 仅允许 SELECT → 表名白名单 → 3s 超时熔断

### 3.4 ImageSearch — CLIP 以图搜菜名

```
用户上传图片 / 输入菜名
      │
      ▼
  CLIP 图片编码 / 文本编码 (openai/clip-vit-base-patch32)
      │
      ▼
  FAISS IndexFlatIP 搜索 (5000 × 512)
      │
      ▼
  Top-K 菜名 → 返回匹配菜谱
```

**Prompt 工程**：文本编码加前缀 `"A dish of {name}"`，对齐 CLIP 对比学习训练分布。
**GPU 自动选择**：`nvidia-smi` 解析空闲显存，≥3GB 用 CUDA，否则 CPU。

### 3.5 PDFParse — 双层引擎

| 模式 | 引擎 | 适用场景 | 额外依赖 |
|------|------|---------|---------|
| **快速 (basic)** | PyMuPDF (fitz) | 纯文本 PDF | 无 |
| **增强 (enhanced)** | magic-pdf (MinerU) | 表格/图片/公式 PDF | ultralytics + doclayout_yolo |
| **自动 (auto)** | magic-pdf → PyMuPDF | 自动尝试增强，失败降级 | - |

### 3.6 Critic — 质量检查

**检测维度**：
1. **完整性** — 回答是否为空
2. **合规** — 营养/替换建议是否带免责声明（当 `need_disclaimer=True` 时跳过）
3. **事实性** — TODO: 答案引用是否在 provenance 中

**重试循环**：
```
critic_passed=False & retry_count < 2 → revision → worker (重新生成)
critic_passed=True 或 retry_count ≥ 2 → save_memory → formatter → END
```

### 3.7 Formatter — 结构化输出

- 添加引用标注（provenance 转可读格式）
- **免责声明唯一添加点**（避免 Critic 误判导致重试循环）

### 3.8 Planner Agent — 复杂请求分解执行

> 📥 从 travel-agent-guide 移植。

**适用场景**：用户提出多步骤需求（如"先推荐低卡食谱，再查意大利面做法，然后做个对比"）。

```
PlannerAgent.run(query)
  → LLM 生成 JSON 计划: {"goal": "...", "steps": [{"tool": "search_recipes", "args": {...}}, ...]}
  → _parse_plan() → 验证 JSON 结构
  → _execute_step(step) → get_tool_registry().invoke(tool_name, **kwargs)
  → 所有步骤完成 → _summarize() → LLM 汇总各步骤结果
```

**集成点**：`graph.py` 的 `route_by_intent` 检测 `_PLANNER_KEYWORDS`（"规划/安排/先…再…/plan/schedule"），
路由到 `planner_node`。`route_after_planner` 判断是否有步骤 → 有则进入 worker，无则直接输出。

### 3.9 Reflection Agent — LLM 深度质量检测

> 📥 从 travel-agent-guide 移植。

在 Critic 规则质检之后，做更深层的 LLM 驱动质量检测：

| 维度 | 检查项 |
|------|--------|
| **事实一致性** | draft 是否有虚构内容、与检索结果是否矛盾 |
| **完整性** | 是否全面覆盖用户问题，有无遗漏关键点 |
| **安全性** | 是否包含不安全建议（过敏源、极端饮食等） |
| **可操作性** | 步骤是否清晰，用户能否按描述执行 |

**自动修订**：若 `passed=False` 且 `revised` 优于 draft，自动替换最终输出。
**流程**：`critic → reflection_node → passed → save_memory` 或 `revised → 替换 draft → save_memory`

### 3.10 Circuit Breaker — 异步三态熔断器

> 📥 从 travel-agent-guide 移植，为 LLM API 调用提供快速失败保护。

```python
class AsyncCircuitBreaker:
    state: CLOSED | OPEN | HALF_OPEN
    failure_count: int          # 连续失败计数
    last_failure_time: float
    total_failure_count: int    # 总失败次数

    async def check(self) -> bool           # OPEN 时返回 False, 不走 API
    async def record_success(self)          # HALF_OPEN + 成功 → CLOSED
    async def record_failure(self)          # 计数满 → OPEN
```

**状态机**：`CLOSED → (3 次失败) → OPEN → (30s) → HALF_OPEN → (1 探测成功) → CLOSED`
`OPEN` 状态下 LLM API 调用立即返回失败，走本地回退，不浪费等待时间。

**监控**：`GET /metrics/circuit_breaker` 暴露当前状态和计数。

### 3.11 LongTermMemory — 长效记忆

> 📥 从 travel-agent-guide 移植，零外部依赖，JSON 文件持久化。

按 `user_id` 索引存储跨会话用户画像：

| 字段 | 说明 |
|------|------|
| `dietary_preferences` | 饮食偏好（素食、低卡等）|
| `favorite_recipes` | 收藏食谱列表 |
| `allergies` | 食物过敏信息 |
| `cuisine_preferences` | 菜系偏好（American、Chinese 等）|
| `recent_queries` | 最近 20 条查询 |
| `interaction_history` | 交互历史摘要 |

**集成**：`graph.py` 中 `load_memory_node` 之后加载 → `save_memory_node` 之后保存，使长效记忆在每轮对话中持续累积。

### 3.12 ToolRegistry — 工具注册中心

> 📥 从 travel-agent-guide 移植，MCP 兼容的工具注册与发现机制。

```python
class ToolRegistry:
    def register(self, tool: RegisteredTool)    # 注册工具
    def has(self, name: str) -> bool            # 检查工具是否存在
    def get(self, name: str) -> RegisteredTool   # 获取工具
    def list_tools(self, category=None)          # 按分类列出工具
    def mcp_tool_catalog(self) -> dict           # MCP 兼容目录输出
    def discover_package(self, path: str)        # 自动发现模块内工具
    async def invoke(self, name: str, **kwargs)  # 调用工具
```

**注册的工具**：

| 工具名 | 功能 | 用途集成 |
|--------|------|---------|
| `search_recipes` | 混合检索食谱 | TextRAG, Planner |
| `search_nutrition` | 营养信息查询 | NutritionSQL, Planner |
| `search_substitution` | 食材替代推荐 | Substitution, Planner |

**自动发现**：`discover_package("src.core.tools")` 扫描 `register_*` 函数自动注册。

---

## 4. 数据流

### 4.1 单轮问答流程（含新组件）

```
Client → POST /ask {"query": "先推荐低卡食谱，再查意大利面做法"}
  → FastAPI → RecipeOrchestrator.ask()
    → LangGraph graph.ainvoke(initial_state)
      → init_node: 生成 request_id
      → load_memory_node: 加载对话历史 (Redis) + LongTermMemory.load(user_id)
      → router_node: 快慢车道意图识别 + 4 态合并
        → 检测到规划关键词 → planner_node
      → planner_node: PlannerAgent.run(query)
        → 分解步骤 → ToolRegistry.invoke("search_recipes", ...) → ToolRegistry.invoke("search_recipes", ...)
        → LLM 汇总 → plan_executed=True
      → route_after_planner: 有步骤 → worker_node (常规 intent) | 无步骤 → critic_node
      → worker_node: 按 intent 派发
        → asyncio.gather(Worker, Fallback)
      → critic_node: 规则质检
      → reflection_node: LLM 深度质检 (自动修订)
      → save_memory_node: 保存对话 + LongTermMemory.save() + 后台 LLM 压缩
      → formatter_node: 格式化输出 + 引用标注
    → 返回 AskResponse
```

### 4.2 多轮对话数据流

```
第1轮: session_id="kitchen-1", query="红烧肉怎么做？"
  → load_memory → 无历史 → Router → TextRAG → 回答
  → save_memory: 存入 Redis {query, intent, slots, answer}

第2轮: session_id="kitchen-1", query="那换成猪肉呢？"
  → load_memory → 加载上一轮 {slots: {recipe_name: "红烧肉"}}
  → Router 收到对话上下文 → 理解"那"指代替换
  → intent: substitution, slots: {ingredient: "猪肉", recipe_name: "红烧肉"}

第3轮: session_id="kitchen-1", query="热量多少？"
  → load_memory → 上一轮答的菜→继承 food
  → intent: nutrition_filter, slots: {food: "红烧肉"}
```

---

## 5. 多轮对话机制

### 5.1 存储架构

```
Redis Key 结构:
  recipe:session:{session_id}         → Hash(session_id, created_at, last_active)
  recipe:session:{session_id}:turns   → List(TurnRecord JSON)
  recipe:session:{session_id}:summary → String (LLM 压缩摘要)
  recipe:session:{session_id}:meta    → Hash(compressed_count)

Redis 不可用时的内存兜底:
  ConversationMemory._fallback_store[session_id] = {turns, summary, ...}
```

### 5.2 Token 感知滑动窗口

`get_context(max_tokens=2048)` 从最新轮次往回计算，累积到 `max_tokens` 为止。

```
[摘要]已压缩的旧轮次摘要 (42 tokens)
  ↓ 预算内保留
[最新轮次 1] ...
[最新轮次 2] ...
  ↓ 超出预算截断
[旧轮次 N] ... → 将被 LLM 压缩为摘要
```

### 5.3 LLM 对话压缩（方案 A）

```
save_memory_node 保存对话
      │
      ▼ (后台异步，不阻塞响应)
asyncio.create_task(memory.compress_if_needed())
      │
      ▼
compress_if_needed()
  ├── 检查未压缩轮次总 token 是否 > max_tokens
  ├── 超过预算 → _compress_with_llm()
  │     ├── 提取旧轮次 → LLM 压缩为 1-3 句摘要
  │     └── 合并到已有摘要 + 更新 compressed_count
  └── 未超预算 → 跳过
```

**实测**：9 轮对话, 压缩前 431 tokens → 压缩后 42 tokens, 压缩率 **9.7%**

---

## 6. 容错与降级

### 6.1 LLM 不可用

```
API 调用 → Circuit Breaker.check() → OPEN (3 次连续失败)
  → 不发起 API 请求, 立即走本地回退
  → 30s 后 HALF_OPEN → 放行 1 个探测请求
    → 成功 → CLOSED (恢复正常)
    → 失败 → OPEN (重置等待, 翻倍至 60s max)

API 调用 → Circuit Breaker.check() → CLOSED
  → 正常发起 → 超时/失败
    → record_failure() → 失败计数 +1
    → 回退本地 Qwen2.5-0.5B (全局单例, lazy load)
      → 本地模型也失败
        → 纯检索结果拼接 (无 LLM)
```

### 6.2 Redis 不可用

```
Redis 连接失败
  → ConversationMemory._redis_available = False
  → 内存兜底 (dict 存储, 进程级)
  → 功能完整, 重启后历史丢失
```

### 6.3 Worker 失败

```
Worker LLM 调用失败
  → 瞬间切换预计算兜底 (_prepare_text_fallback)
  → 纯检索片段拼接, 无 LLM 生成开销
  → 返回速度更快但回答质量略低
```

### 6.4 PDF 解析降级

```
magic-pdf 不可用/解析失败
  → 静默降级 PyMuPDF
  → 仅提取纯文本, 丢失表格/图片/公式
```

### 6.5 LongTermMemory 持久化

```
JSON 文件写入失败 (磁盘满/权限)
  → 静默跳过保存, 不影响主流程
  → 下次 `load()` 返回空记忆, 功能完整
```

## 7. 性能优化

### 7.1 已实施的优化

| 优化项 | 效果 | 涉及模块 |
|--------|------|---------|
| 全局单例 LLM | 首加载~43s → 后续~1.5s | llm.py |
| 全局单例 HybridRetriever | 首加载~30s → 后续~50ms | hybrid.py |
| 全局单例 CLIP | 避免重复加载模型 | clip_retriever.py |
| 关键词 SQL 模板 | 113s → **12ms** (营养查询) | nutrition_sql.py |
| Worker + Fallback 并行 | 零额外延迟兜底 | graph.py |
| NutritionSQL 先跑 Worker | 避免 fallback 阻塞 | graph.py |
| 免责声明统一管理 | 消除重试循环 | critic.py, formatter.py |
| Redis 持久化 | 会话跨请求保持 | memory.py |
| LLM 对话压缩 | Token 节省 **90.3%** | memory.py |
| **Circuit Breaker 熔断** | 快速失败, 避免 API 等待超时 | circuit_breaker.py |
| **Router 规则快速通道** | 置信度 ≥ 0.9 直接命中, 0ms 延迟 | router.py |
| **LongTermMemory JSON 持久化** | 零外部依赖, 无网络开销 | long_term_memory.py |

### 7.2 Benchmark 指标

| 指标 | 值 | 说明 |
|------|-----|------|
| P95 延迟 | ~3-5s | 含 LLM 调用 |
| 营养查询 P95 | ~12ms | 关键词模板命中时 |
| 意图识别延迟 | ~2s (FC) / ~0ms (规则) | Function Calling 或规则 |
| 检索延迟 | ~50ms | BM25 + FAISS + Rerank |
| 本地 LLM 首加载 | ~43s | Qwen2.5-0.5B |
| 本地 LLM 后续调用 | ~1.5s | 复用 pipeline |
| CUDA 显存占用 | ~1.5 GB (本地 LLM) | RTX 5070 |
| 支持并发会话 | 1000 | 进程内 + Redis |

---

## 8. 部署架构

### 8.1 单机部署（当前）

```
┌──────────────────────────────┐
│  单机 (Ubuntu + RTX 5070)    │
│                              │
│  ┌──────┐  ┌──────────────┐ │
│  │ Nginx│→│ FastAPI       │ │
│  │ 反向 │  │ uvicorn:8000 │ │
│  │ 代理 │  │ (1 worker)   │ │
│  └──────┘  └──────┬───────┘ │
│                   │         │
│  ┌────────────────▼────────┐│
│  │  Python 进程             ││
│  │  ├── Qwen2.5-0.5B (GPU) ││
│  │  ├── all-MiniLM-L6-v2   ││
│  │  ├── BGE-Reranker       ││
│  │  ├── CLIP-ViT-B/32      ││
│  │  └── FAISS 索引 (内存)  ││
│  └─────────────────────────┘│
│                              │
│  ┌──────┐  ┌──────────────┐ │
│  │Redis │  │ SQLite       │ │
│  │ 会话  │  │ recipe.db   │ │
│  └──────┘  └──────────────┘ │
└──────────────────────────────┘
```

### 8.2 Docker 部署（可选）

```bash
docker compose up -d
# 包含: FastAPI + Redis + 数据卷
```

---

## 9. 技术选型

| 类别 | 选型 | 理由 |
|------|------|------|
| **编排框架** | LangGraph StateGraph | 有向图编排, 天然支持条件边 + 回边 (重试循环) |
| **API 框架** | FastAPI | 异步原生, SSE 流式, Pydantic 集成 |
| **主 LLM** | Qwen3.5-35B (API) | 35B 参数高质量, ModelScope 国内可访问 |
| **回退 LLM** | Qwen2.5-0.5B (本地) | 0.5B 极小体量, ~1.5GB 显存, API 不可用时兜底 |
| **嵌入模型** | all-MiniLM-L6-v2 | 384 维轻量, 语义检索够用 |
| **重排序器** | BGE-reranker-v2-m3 | 多语言支持, 精排提升明显 |
| **图文检索** | CLIP-ViT-B/32 | 零样本以图搜文, 无需标注数据 |
| **关键词检索** | BM25 (自定义实现) | 轻量无依赖, jieba 中文分词 |
| **向量检索** | FAISS (IndexFlatIP) | 精确内积搜索, 5000×384 实时 |
| **SQL 校验** | sqlglot | AST 级别 SQL 解析, 白名单校验 |
| **持久化** | Redis + SQLite | Redis 会话管理, SQLite 结构化数据 |
| **PDF 解析** | PyMuPDF + magic-pdf | 双层引擎, 复杂 PDF 增强解析 |
| **监控** | 自定义 MetricsCollector | 轻量内存指标, 可对接 Prometheus |
| **熔断器** | AsyncCircuitBreaker (自实现) | 异步三态, LLM API 快速失败保护 |
| **长效记忆** | LongTermMemory (自实现) | JSON 文件持久化, 零外部依赖 |
| **工具注册** | ToolRegistry (自实现) | MCP 兼容目录, 自动发现包内工具 |
| **配置** | pydantic-settings | 类型安全, .env 文件加载 |

---

## 附录 A：项目结构映射

```
recipe-assistant/
├── src/
│   ├── agents/          → 8 个 Agent (Router/TextRAG/NutritionSQL/ImageSearch/PDFParse/Substitution/Critic/Formatter)
│   ├── core/
│   │   ├── retrievers/  → 混合检索 (BM25+FAISS) + CLIP 图文检索
│   │   ├── sql/          → SQL 安全沙箱 (validator + sandbox)
│   │   ├── utils/        → LLM 封装 (API 优先+本地回退) + 日志 + 指标
│   │   ├── memory.py     → 对话记忆管理 (Redis + Token 滑动窗口 + LLM 压缩)
│   │   ├── pdf_parser.py → PDF 双层引擎
│   │   └── config.py     → Pydantic 配置
│   ├── orchestrator/    → LangGraph 状态图 (10 节点) + Supervisor 包装
│   ├── api/             → FastAPI 入口 + SSE 流式 + 请求/响应模型
│   ├── data/            → 5K 食谱 + 15K chunks + FAISS 索引 + 图片
│   └── eval/            → 自动化评测 (intent_accuracy / recall@10 / MRR / P95)
├── scripts/             → 数据导入/索引构建
├── tests/               → 单元测试 + 集成测试
├── docs/                → 文档
└── frontend/            → Web 前端 (可选)
```
