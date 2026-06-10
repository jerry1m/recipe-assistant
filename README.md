# 🍳 多模态智能食谱助手 (Multi-Modal Recipe Assistant)

**轻量级 Multi-Agent RAG 系统** | 个人项目 | 2026.06

> 💡 从 travel-agent-guide 项目中借鉴了 Circuit Breaker、ToolRegistry、Planner+Reflection、快慢车道意图识别合并策略等设计，显著提升了系统鲁棒性和编排灵活性。

---

## 🎯 项目动机与背景

### 为什么做这个项目？

> **核心动机**：用**真实可用的产品级工程**来系统实践 Multi-Agent 编排、LLM 工程化和混合检索这三项前沿技术，而不只是一个 POC 或 Demo。

**Multi-Agent 落地场景**：

| 场景 | 对应能力 | 为什么适合 |
|------|---------|-----------|
| "红烧肉怎么做？" | 意图识别 + 检索增强生成 | 需要精准识别用户要菜谱而非营养，然后从菜谱库检索 |
| "热量多少？" | 结构化查询 + 安全校验 | 营养数据是结构化的，走 SQL 比检索更准确、更快速 |
| "没有鸡蛋怎么办？" | 食材替换推理 | 需要 LLM 理解替代逻辑（鸡蛋→豆腐/香蕉），不能只靠检索 |
| "这道菜的照片？" | 多模态检索 | 需要以图搜文/以文搜图，跨模态匹配 |
| "帮我看看这份 PDF 食谱" | 文档解析 + 智能问答 | PDF 中提取信息后再回答问题，端到端处理 |

这 5 个场景天然对应 **不同的技术方案**——不是用一个 LLM 解决所有问题，而是让**专业的 Agent 做专业的事**，再由编排层统一调度。

### 项目收获

1. **Multi-Agent 不只是概念**：8 个 Agent 协同工作 + LangGraph 编排 + Critic 重试循环，在实际场景中可落地且有明确收益
2. **LLM 工程化不是简单调 API**：API+本地双活、全局单例、超时熔断、自动回退、对话压缩、Function Calling 容错——一套完整的工程体系
3. **性能优化需要端到端思维**：113s→12ms 的优化不是改一行代码，而是从路由→SQL→Critic→Formatter 整条链路分析
4. **容错设计决定可用性**：不依赖任何一个单点——LLM 有回退、Redis 有兜底、PDF 有降级、Worker 有 Fallback

---

## 📋 项目简介

基于 **Function Calling** 意图识别（快慢车道 + 显式合并策略）+ **LangGraph 多轮编排**的多智能体食谱问答系统，搭载 **Qwen3.5-35B (ModelScope API)** 大模型
（API 不可用时自动回退本地 **Qwen2.5-0.5B**），
通过混合检索（BM25 + FAISS + BGE-Reranker）与 SQL 安全查询，为 5000 条真实食谱提供智能问答服务。
支持**多轮对话**（ConversationMemory 按 session 管理）、**LLM 对话压缩**（后台异步压缩旧轮次，压缩率 >90%）、**Critic 重试循环**（LangGraph 回边自动重试）、**Circuit Breaker 熔断保护**（API 快速失败）、**LongTermMemory 长效记忆**（跨会话用户偏好）、**Planner/Reflection Agent 模式**（复杂请求分解 + LLM 深度质检）以及 **ToolRegistry 工具注册中心**（自动发现与 MCP 兼容目录）。

---

## 🏗️ 系统架构

```
用户输入 (文本/图片)
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  LongTermMemory.load(user_id) — 加载跨会话用户偏好        │
└──────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────┐    Function Calling 意图识别 (快慢车道 + 合并策略)
│  Router   │ ──────────────────────────────────────────→ Intent
└──────────┘    ① 规则快速通道 (0ms, confidence ≥ 0.9 直接命中)
                 ② Function Calling 慢通道 → 4 态合并: agree / prefer_rule / prefer_fc / fc_failed
      │
      ├──→ Planner?  (检测"规划/安排/先…再…"关键词)
      │     │
      │     ▼
      │  ┌──────────┐    ToolRegistry 自动发现工具
      │  │  Planner  │ ──→ search_recipes / search_nutrition / …
      │  │  Agent    │ ──→ 分解步骤 → 逐步执行 → LLM 汇总
      │  └──────────┘
      │     │
      └─────┤  (Worker + Fallback 并行)
╔══════════════════════════════════════════════════════════════════╗
║  asyncio.gather(WORKER, FALLBACK) —  src/orchestrator/graph.py  ║
╚══════════════════════════════════════════════════════════════════╝
      │
      ├──→ TextRAGAgent        ───┐
      ├──→ NutritionSQLAgent   ───┤  asyncio.gather
      ├──→ SubstitutionAgent   ───┤       +
      ├──→ ImageSearchAgent    ───┤  (CLIP 以图搜菜名)
      ├──→ PDFParseAgent       ───┤  (PDF 文档解析)
      └──→ _prepare_text_fallback ──┘  纯检索兜底 (no LLM)
      │
      ▼
   Worker 成功？
   ╱            ╲
 ✅              ❌
Critic          直接切换
╱      ╲        预计算兜底
通过    失败    (检索片段)
 │       │
 │       ▼
 │   ┌────────────┐
 │   │ Reflection │ ← LLM 深度质检: 事实一致性/完整性/安全性/可操作性
 │   │  Agent     │ → 需要修订则自动修正, 无问题则通过
 │   └────────────┘
 │       │
 └───────┘
      │
      ▼
  LongTermMemory.save(user_id, interaction) — 保存本次交互
      │
      ▼
  Formatter ──→ 最终回答
```

### Agent 职责

| Agent | 职责 | 核心能力 |
|-------|------|----------|
| **Router** | 意图识别 + 槽位提取 | 规则快速通道 + Function Calling, **快慢车道 4 态合并策略** (`fast_lane_only`/`agree`/`conflict_prefer_rule`/`conflict_prefer_fc`) |
| **Planner** | 复杂请求分解执行 | LLM 生成多步骤计划 → **ToolRegistry** 逐步调用工具 → LLM 汇总结果 |
| **Reflection** | 深度质量检测 | LLM 驱动: 事实一致性/完整性/安全性/可操作性; 自动修订并替换 draft |
| **TextRAG** | 食谱问答 | HybridRetriever (BM25+FAISS+Rerank) + LLM 生成 |
| **NutritionSQL** | 营养查询 | **三路策略**: ①关键词 SQL 模板(0 LLM 延迟) ②LLM 翻译 SQL → SQLSandbox 执行 ③混合检索兜底; 带菜系感知过滤 |
| **ImageSearch** | 图片/文本检索 | 混合检索 + CLIP 图文检索 ✅ (openai/clip-vit-base-patch32) |
| **Substitution** | 食材替换推理 | LLM 链式推理, JSON 结构化输出 |
| **PDFParse** | PDF 文档解析 | 双层引擎: PyMuPDF 快速模式 + magic-pdf (MinerU) 增强模式, 自动降级 |
| **Critic** | 事实核查 + 合规 | 规则检查 + LLM 辅助, 中英文免责检测; 输出交给 Reflection 做深度质检 |
| **Formatter** | 结构化输出 | 最终润色 + **免责声明唯一添加点**（避免 Critic 误判导致重试循环）|

---

## 📦 数据

| 数据项 | 数量 | 说明 |
|--------|------|------|
| 食谱总数 | **5,000** | Eitanli/cuisine_type数据集（营养成分由LLM补齐）, 多菜系英文食谱 |
| 文本 Chunks | **15,000** | 按 metadata/ingredients/steps 三类, 步骤全量合并 |
| 向量维度 | 384 | all-MiniLM-L6-v2 嵌入 |
| 检索方式 | BM25 + FAISS (IP) + BGE-Reranker | 混合检索 + 精排 |
| 嵌入模型 | all-MiniLM-L6-v2 (384dim, 本地加载) | 优先从本地缓存加载, 兼容 HF 自动下载 |
| 关系库 | SQLite (recipe.db) | recipes/nutrition/ingredients/steps 四表 |

### 数据文件

```
src/data/
├── recipes_real.json        # 5,000 条原始食谱
├── chunks.json              # 15,000 个检索 Chunk
├── vector_store/            # FAISS 索引 + BM25 模型
│   ├── recipes.index        # 向量索引 (23MB, 15000×384)
│   ├── bm25.pkl             # BM25 序列化 (17MB)
│   └── chunk_ids.npy        # Chunk ID 映射
├── vector_store/            # CLIP 图文检索索引
│   ├── clip_recipe_names.index # CLIP 文本索引 (10MB, 5000×512, IndexFlatIP)
│   ├── clip_recipe_names.pkl   # 菜名映射
│   └── clip_recipe_ids.npy     # recipe_id 映射
├── images/                  # 食谱图片 (CLIP 检索用)
```

---

## 🤖 LLM 配置

| 配置项 | 值 |
|--------|-----|
| 主模型 (API) | **Qwen/Qwen3.5-35B-A3B** (ModelScope API) |
| 回退模型 (本地) | **Qwen/Qwen2.5-0.5B-Instruct** |
| 回退加载方式 | 全局单例 (lazy load, 首次 ~43s, 后续 ~1.5s) |
| 回退显存占用 | ~1.5 GB (RTX 5070) |
| 默认温度 | temperature=0.3 |
| 嵌入模型 | all-MiniLM-L6-v2 (384-dim) |
| 重排序器 | BAAI/bge-reranker-v2-m3 |

**调用策略**：API 优先，所有 Agent 的 LLM 调用先发往 ModelScope API（35B 参数），
请求失败时自动回退到本地 Qwen2.5-0.5B，保证服务不中断。

支持通过 `.env` 文件切换 LLM 提供商（兼容 OpenAI 格式）：
```
RECIPE_LLM_API_KEY=your_key
RECIPE_LLM_BASE_URL=https://api-inference.modelscope.cn/v1
RECIPE_LLM_MODEL=Qwen/Qwen3.5-35B-A3B
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 显卡 (推荐 8GB+ 显存, 用于本地 LLM)
- 依赖见 `requirements.txt`

### 安装

```bash
# 1. 克隆仓库
cd recipe-assistant

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量 (可选)
cp .env.example .env

# 4. 下载模型权重 (首次运行自动下载)
# Qwen2.5-0.5B-Instruct + all-MiniLM-L6-v2 + BGE-reranker-v2-m3
```

### 运行

```bash
# 启动 API 服务 (FastAPI + SSE)
make run
# 或: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# 运行测试
make test

# 运行评测
make benchmark
```

### API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/ask` | 单轮问答 |
| POST | `/ask/stream` | SSE 流式问答 |
| GET  | `/health` | 健康检查 |
| GET  | `/metrics/agents` | Agent 调用统计 |
| GET  | `/metrics/business` | 业务指标 |
| GET  | `/metrics/compression` | 对话压缩统计信息 |
| GET  | `/metrics/circuit_breaker` | Circuit Breaker 状态 (state/failure_count/last_failure_time/total_failure_count) |
| GET  | `/metrics/long_term_memory` | LongTermMemory 统计 (总用户数/总条目数) |

#### 请求示例

```bash
# 单轮问答
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "How to make roasted carrots?", "intent_hint": "text_rag"}'

# 多轮对话（传 session_id 维持上下文）
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "红烧肉怎么做？", "session_id": "my-session-123"}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "那换成猪肉呢？", "session_id": "my-session-123"}'

# 上传 PDF 文档解析
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "帮我解析这个食谱PDF", "files": ["<base64_encoded_pdf>"], "session_id": "my-session-456"}'
```

---

## 🧪 测试

```bash
# 全部测试
pytest tests/ -v

# 带覆盖率
pytest tests/ --cov=src --cov-report=term-missing
```

---

## � 性能优化与问题排查记录

### 案例 1：菜品推荐回答质量差（印度菜推荐）

**现象**：用户问"推荐印度菜"，回答只有寥寥数语，列不出印度菜。

**根因分析**：
1. **检索 query 翻译不当** — 把整句翻译成英文去检索，而非提取关键 cuisine 词
2. **top_k 太小** — 检索结果太少，LLM 无素材可总结
3. **system prompt 通用** — 没有专门针对推荐的 prompt

**修复方案**：
- `text_rag.py`：从 `slots.cuisine_type` 提取菜系关键词构造检索 query
- 当 `is_recommend=True` 时 top_k 翻倍（5→10）
- 使用专用 `_RECOMMEND_SYSTEM_PROMPT`，强调列出多样菜品

**效果**：回答从 ~300 字/2 道菜 → ~1087 字/5 道正宗印度菜 🚀

### 案例 2：NutritionSQL 耗时 113s + 双免责声明 + 菜系不符

**现象**：查询"美国美食但低热量"，路由命中 `nutrition_filter`，耗时 113s，返回了非美式菜品，且两条免责声明。

**根因分析**：

| 问题 | 根因 | 修复 |
|------|------|------|
| **延迟 113s** | 每次查询走 LLM 翻译 SQL，API 超时后回退本地模型（~30s/次），且 Critic 连续拒绝 2 次触发 3 轮重试 | ① 关键词 SQL 模板（0 LLM 延迟）<br>② 常用模式直接匹配预编译 SQL |
| **双免责声明** | NutritionSQL 内嵌了免责声明，Formatter 又追加一条 | 移除所有 Agent 内置免责声明，由 Formatter 统一添加 |
| **Critic 重试循环** | Critic 检查免责声明发现缺失→拒绝→重试，浪费 ~40s | `need_disclaimer=True` 时 Critic 跳过免责检查 |
| **Fallback 并行阻塞** | `asyncio.gather(worker, fallback)` 等待慢速兜底检索完成（~20s） | NutritionSQL/Substitution 先跑 Worker，成功则直接返回 |
| **菜系不符** | Router 按关键词"热量"→ `nutrition_filter`，未提取菜系信息，且 Function Calling 的 `nutrition_filter` 参数缺少 `cuisine_type` 字段 | ① Router 规则通道增加 `_extract_cuisine()` 提取菜系注入 slots<br>② Function Calling 的 nutrition_filter 参数增加 `cuisine_type` 字段<br>③ NutritionSQL 优先从 `slots` 取菜系，兜底从 query 正则匹配<br>④ graph.py 将 `slots` 传给 NutritionSQLAgent |
| **无模板命中时延迟仍高** | 即使无模板匹配，仍可能走 LLM→SQL 的慢路径 | 添加兜底超时控制 + 提速 fallback 响应 |

**修复架构 - NutritionSQL 三路策略**：
```
query → 关键词模板匹配 (0 LLM 延迟)
         ├── 匹配 → 菜系感知 SQL 注入 → 直接返回
         └── 不匹配 → LLM 翻译 SQL → SQLSandbox 执行
                       ├── 成功 → 返回结果
                       └── 失败 → 混合检索兜底 + LLM 生成
```

**免责声明管理规范**：
```
Agent (无免责) → Critic (need_disclaimer=True 时跳过) → Formatter (统一添加)
```

**延迟对比**（同一查询"美国美食低热量"）：

| 阶段 | 延迟 | 重试次数 | 免责声明 |
|------|------|---------|---------|
| 修复前 | **113s** | 2 | 2条 ❌ |
| 第一次修复（关键词模板） | 21.9s | 2 | 2条 ❌ |
| Critic 修复后 | 20.7s | **0** ✅ | 1条 ✅ |
| **移除并行 fallback** | **12ms** 🚀 | 0 | 1条 ✅ |

### 关键经验总结

1. **SQL 模板先行**：对常见营养查询（低热量/高蛋白/低脂/低碳水）使用正则匹配+预编译 SQL，避免每次走 LLM
2. **免责声明单一来源**：Formatter 是唯一添加点，所有 Agent 不内置。Critic 感知 `need_disclaimer` 标志
3. **Worker 成功不等 Fallback**：NutritionSQL/Substitution 这类快速命中的场景，先跑 Worker，成功即返回
4. **菜系感知双通道**：Router 规则快速通道 + Function Calling 都提取 `cuisine_type`，通过 `slots` 传递给下游 Agent
5. **Router 规则通道完善**：常见关键词（热量、卡路里、低卡、低脂、高蛋白等）直接走规则快速通道（0ms），避免 Function Calling 耗时

---

## �📁 项目结构

```
recipe-assistant/
├── src/
│   ├── agents/                    # 🤖 多 Agent 层
│   │   ├── base.py                # 基类: 重试/超时/日志/Fallback
│   │   ├── router.py              # 意图识别 (Function Calling + 规则混合 + 快慢车道 4 态合并)
│   │   ├── planner.py             # 🆕 复杂请求分解 → 逐步执行 → 汇总
│   │   ├── reflection.py          # 🆕 LLM 深度质检 (事实/完整/安全/可操作)
│   │   ├── text_rag.py            # 文本问答 (检索 + LLM)
│   │   ├── image_search.py        # 图文检索
│   │   ├── nutrition_sql.py       # 营养 SQL 查询
│   │   ├── pdf_parse.py           # PDF 文档解析 (双层引擎)
│   │   ├── substitution.py        # 食材替换推理
│   │   ├── critic.py              # 事实核查 + 合规
│   │   └── formatter.py           # 结构化输出
│   ├── core/                      # 🔧 核心组件
│   │   ├── config.py              # Pydantic 配置
│   │   ├── pdf_parser.py          # PDF 解析引擎 (PyMuPDF + magic-pdf)
│   │   ├── long_term_memory.py    # 🆕 长效记忆 (JSON 文件持久化, 跨会话用户偏好)
│   │   ├── retrievers/
│   │   │   ├── hybrid.py          # BM25 + FAISS + Rerank
│   │   │   └── clip_retriever.py  # CLIP 图文检索 (openai/clip-vit-base-patch32)
│   │   ├── sql/
│   │   │   ├── validator.py       # sqlglot AST 校验
│   │   │   └── sandbox.py         # 只读 + 超时熔断
│   │   ├── memory.py              # 多轮对话记忆管理 (session_id 基)
│   │   ├── tools/                 # 🆕 工具注册中心
│   │   │   ├── __init__.py        # 导出 Registry + get_tool_registry() 单例
│   │   │   ├── registry.py        # ToolRegistry 类 (注册/发现/MCP 兼容目录)
│   │   │   └── retrieval_tools.py # 食谱检索工具 (注册到 ToolRegistry)
│   │   ├── utils/
│   │   │   ├── llm.py             # API 优先 + 本地回退 LLM 封装 (含 Function Calling + Circuit Breaker)
│   │   │   ├── circuit_breaker.py # 🆕 异步三态熔断器 (CLOSED/OPEN/HALF_OPEN)
│   │   │   ├── logger.py          # 结构化日志
│   │   │   └── metrics.py         # Prometheus 指标
│   │   └── adapters/              # 领域适配 (预留)
│   ├── orchestrator/
│   │   ├── graph.py               # LangGraph 状态图 (12 节点: +planner +reflection)
│   │   └── supervisor.py          # 编排器: LangGraph 包装层 + ask() 入口
│   ├── api/
│   │   ├── main.py                # FastAPI 入口
│   │   ├── streaming.py           # SSE 流式输出
│   │   └── schemas.py             # Pydantic 模型
│   ├── data/                      # 📦 数据
│   │   ├── recipes_real.json      # 5,000 条原始食谱
│   │   ├── chunks.json            # 15,000 检索 Chunks
│   │   └── vector_store/          # FAISS + BM25 索引
│   └── eval/                      # 📊 评测
│       ├── benchmark.py           # 自动化评测
│       └── test_cases.json        # 测试用例
├── scripts/
│   ├── get_data.py                # Recipe1M 下载
│   └── seed_db.py                 # SQLite 建库 + 导入
├── tests/                         # 单元/集成测试
├── examples/                      # 使用示例
├── docs/                          # 文档
├── docker-compose.yml             # Docker 部署
├── Dockerfile
├── Makefile
├── requirements.txt
└── .env.example
```

---
###  LangGraph 编排 + 多轮对话 + Critic 重试循环
本项目已从自定义 Supervisor **迁移到 LangGraph StateGraph**，支持多轮对话状态维护和 Critic 重试闭环。

```
用户输入 (文本/图片)
      │
      ▼
  ┌──────────┐
  │   init    │ ← 初始化 state
  └────┬─────┘
       │
       ▼
  ┌─────────────┐
  │ load_memory  │ ← 加载 session 历史 + 上一轮槽位
  └────┬────────┘
       │
       ▼
  ┌──────────┐    Function Calling 意图识别
  │  Router   │ ─────────────────────→ Intent + 槽位
  └────┬─────┘    （含多轮上下文注入）
       │
       ▼  (按 intent 派发)
  ┌──────────┐
  │  worker   │ ← 统一派发节点，按 intent 调用对应 Worker
  └────┬─────┘     Worker + Fallback 并行 (asyncio.gather)
       │
       ▼
  ┌──────────┐
  │  critic   │ ← 事实核查 + 合规质检
  └────┬─────┘
       │
   ┌────┴────┐
   │         │
   通过     失败 (retry < max)
   │         │
   │     ┌──────────┐
   │     │ revision  │ ← 自增 retry_count，注入 critic_suggestions 作为 feedback
   │     └────┬─────┘
   │          │ (回边到 worker)
   │         ...
   │         失败 (retry ≥ max) → 走预计算兜底
   │
   ▼
  ┌─────────────┐
  │ save_memory  │ ← 保存对话轮次 + 槽位到 ConversationMemory
  └────┬────────┘
       │
       ▼
  ┌───────────┐
  │ formatter  │ → 最终回答
  └───────────┘
```

**多轮对话详解**：

#### session_id 会话机制

前端/客户端在每次请求中传入同一 `session_id` 即属于同一对话，不传则每次自动生成新 UUID。
支持 **1000 个并发会话**，**3600 秒无活动自动清理**。

```python
# src/orchestrator/supervisor.py
session_id = request.session_id or str(uuid.uuid4())
```

#### ConversationMemory 存储结构

每个 `session_id` 对应一个 `ConversationMemory` 实例，按轮次 `TurnRecord` 存储：

```python
# src/api/schemas.py
class TurnRecord(BaseModel):
    query: str                    # 用户问题
    intent: str                   # 意图
    slots: dict[str, Any]        # 提取的槽位（菜名、食材、营养元素等）
    answer: str                   # 助手回答
    provenance: list[ProvenanceItem]
    timestamp: float

# src/core/memory.py
class ConversationMemory:
    turns: list[TurnRecord]       # 最多 20 轮
    last_active: float            # 最近活跃时间（用于超时清理）
```

#### 三路上下文注入

`load_memory_node` 在每轮请求开始时注入三类上下文到 Router：

| 上下文 | 来源 | 作用 |
|--------|------|------|
| **对话文本摘要** | `memory.get_context(last_n=3)` | 最近 3 轮完整对话，让 LLM 理解指代 |
| **上一轮槽位** | `memory.get_last_slots()` | 上一轮提取的菜名/食材/营养元素 |
| **实体摘要** | `memory.get_entity_summary()` | 全局跨轮实体追踪 |

```python
# src/orchestrator/graph.py — load_memory_node
async def load_memory_node(state: RecipePipelineState) -> dict:
    memory = ConversationMemory.get_or_create(session_id)
    return {
        "conversation_context": memory.get_context(last_n=3),
        "last_slots": memory.get_last_slots(),
    }
```

Router 的 Function Calling 系统 prompt 中注入这些信息：

```python
# src/agents/router.py — _fc_classify()
ctx_parts = [
    "You are a recipe assistant intent classifier...",
    f"\n--- 对话上下文 ---\n{conversation_context}",   # 最近3轮
    f"\n--- 上一轮提取的槽位 ---\n{last_slots}",       # 菜名/食材等
]
```

#### 多轮场景示例

| 轮次 | 用户输入 | session_id | 多轮机制 | 识别结果 |
|------|---------|-----------|----------|---------|
| 1 | 红烧肉怎么做？ | `session-A` | 无历史 | `step_qa` + recipe=红烧肉 |
| 2 | 那换成猪肉呢？ | `session-A` | 上一轮槽位 recipe=红烧肉 → 知道在替换食材 | `substitution` + ingredient=猪肉 |
| 3 | 热量多少？ | `session-A` | 上一轮答的菜是红烧肉 → 继承 food | `nutrition_filter` + food=红烧肉 |
| 4 | 推荐一个甜点 | `session-B` | 新会话，无历史 | `ingredient_recommend` |
| 5 | 这个需要烤箱吗？ | `session-B` | 上轮推荐了提拉米苏 → 指代消解 | `step_qa` + recipe=提拉米苏 |

#### 会话生命周期管理

```
创建: session 首次请求 → ConversationMemory.get_or_create(sid)
活跃: 每轮 add_turn() 更新 last_active
清理: 下次请求时检查 → 3600s 无活动自动删除
上限: 超 1000 会话时淘汰最旧会话
轮次: 每会话最多保留 20 轮（FIFO）
```

```python
# src/core/memory.py — 自动清理
@classmethod
def get_or_create(cls, session_id: str) -> "ConversationMemory":
    now = time.time()
    # 清理过期会话
    expired = [sid for sid, mem in cls._instances.items()
               if now - mem.last_active > cls._TURN_TIMEOUT]   # 3600s
    for sid in expired:
        del cls._instances[sid]
    # 超过上限淘汰最旧
    if len(cls._instances) >= cls._MAX_SESSIONS:               # 1000
        oldest = min(cls._instances.items(), key=lambda x: x[1].last_active)
        del cls._instances[oldest[0]]
    # 创建或复用
    return cls._instances.setdefault(session_id, cls(session_id))
```

#### 调用方式

```bash
# 第1轮：创建会话（不传 session_id 会自动生成）
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "红烧肉怎么做？", "session_id": "my-kitchen"}'

# 第2轮：传相同 session_id 维持上下文
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "那换成猪肉呢？", "session_id": "my-kitchen"}'

# 第3轮：继承上轮槽位自动查营养
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "热量多少？", "session_id": "my-kitchen"}'
```

**Critic 重试循环**：
- 最大重试 2 次，`revision` 节点自增计数器
- 重试时 Worker 收到 `critic_suggestions` 作为 `feedback` 参数
- 仍失败则走预计算检索兜底

**Worker 统一派发**：

```python
# 单一 worker_node 按 intent 派发
intent_map = {
    "text_rag": text_rag.run,
    "nutrition_sql": nutrition_sql.run,
    "substitution": substitution.run,
    "image_search": image_search.run,
    "pdf_parse": pdf_parse.run,
}
# 内部 asyncio.gather(worker, fallback)
```

### LLM 对话压缩（方案 A）— 后台异步压缩旧轮次

超长对话上下文会导致 Token 预算超支和 LLM 注意力分散。方案 A 通过 **LLM 摘要压缩** 将旧轮次对话压缩为简短摘要，大幅节省 Token。

#### 存储扩展

每个 session 在 Redis 中新增两个 key：
- `...:summary` — 压缩摘要（累积拼接）
- `...:meta` — 已压缩轮次数

#### 触发时机

```
save_memory_node 保存对话
      │
      ▼ (后台异步，不阻塞响应)
asyncio.create_task(memory.compress_if_needed())
      │
      ▼
compress_if_needed()
  ├── 检查未压缩轮次总 token 是否 > max_tokens (默认 2000)
  ├── 超过预算 → 调用 _compress_with_llm()
  │     ├── 提取旧轮次（保留最新 N 轮完整对话）
  │     ├── 调用 LLM 压缩为 1-3 句摘要
  │     └── 合并到已有摘要 + 更新 meta
  └── 未超预算 → 跳过
```

#### 上下文组装

```
get_context()
  ├── 优先加载压缩摘要（仅占极少 token）
  ├── 附上最新几轮完整对话
  └── 返回组装后的上下文文本
```

#### 实测指标

| 指标 | 值 |
|------|-----|
| 测试条件 | 9 轮对话, max_tokens=800 强制触发 |
| 压缩前 | 431 tokens（3 轮旧对话）| |
| 压缩后 | 42 tokens（LLM 摘要）|
| **压缩率** | **9.7%**（节省 **90.3%** Token）|

> 摘要内容预览："用户想了解如何烹饪一种经典的中国红猪蹄..."

#### 代码结构

```python
# src/core/memory.py
class ConversationMemory:
    async def compress_if_needed(self):          # 后台检查 + 触发
    async def _compress_with_llm(self, turns):    # LLM 压缩核心逻辑
    def get_compression_stats(self):              # 压缩统计

# src/orchestrator/graph.py — save_memory_node
asyncio.create_task(memory.compress_if_needed())  # 不阻塞响应

# src/api/main.py
GET /metrics/compression  # 压缩指标端点
```

**设计要点**：
- **不阻塞主响应**：`asyncio.create_task` 后台运行
- **仅压缩旧轮次**：保留最近 N 轮完整对话供指代消解
- **摘要累积**：新摘要追加到已有摘要后，不丢失历史
- **Token 预算触发**：仅当未压缩轮次超预算时才执行，避免频繁调用 LLM

###  Circuit Breaker（异步三态熔断器）

> 📥 从 travel-agent-guide 移植，为 LLM API 调用提供快速失败保护。

**三态机**：

```
CLOSED (正常)
  → 连续 3 次失败 → OPEN（立即拒绝请求，不等待超时）
    → 30s 后 → HALF_OPEN（放行 1 个探测请求）
      → 成功 → CLOSED
      → 失败 → OPEN（重置等待期，翻倍至 60s max）
```

**集成位置**：`src/core/utils/llm.py` — 所有 API 调用前先 `circuit_breaker.check()`，若 OPEN 则直接走本地回退，不浪费等待时间。

```python
# src/core/utils/circuit_breaker.py
class CircuitBreakerState(enum.Enum):
    CLOSED = "closed"        # 正常
    OPEN = "open"            # 熔断
    HALF_OPEN = "half_open"  # 半开探测

class AsyncCircuitBreaker:
    async def check(self) -> bool:            # 检查是否允许请求
    async def record_success(self):           # 记录成功
    async def record_failure(self):           # 记录失败（计数满 → OPEN）
    async def get_status(self) -> dict:       # 获取状态（用于 /metrics/circuit_breaker）
```

**监控**：`GET /metrics/circuit_breaker` 返回当前 state / failure_count / last_failure_time / total_failure_count。

---

## � 性能仪表盘

### 端到端延迟

| 场景 | P50 | P95 | 说明 |
|------|-----|-----|------|
| 营养查询（模板命中） | ~8ms | ~12ms | 关键词 SQL 模板，0 LLM 延迟 |
| 食谱问答（TextRAG） | ~2.5s | ~4s | 检索 + LLM 生成 |
| 食材替换（Substitution） | ~2s | ~3.5s | LLM 链式推理 |
| 图文检索（CLIP） | ~1s | ~1.5s | 图片编码 + FAISS 搜索 |
| 闲聊（Chitchat） | ~50ms | ~100ms | 不走 LLM |
| PDF 解析（增强模式） | ~5s | ~15s | magic-pdf 解析 |
| PDF 解析（快速模式） | ~500ms | ~1s | PyMuPDF 提取纯文本 |

### 检索性能

| 指标 | 值 |
|------|-----|
| BM25 检索延迟 | ~10ms |
| FAISS 检索延迟 | ~5ms (15000x384 IndexFlatIP) |
| BGE-Reranker 精排延迟 | ~30ms (top 20) |
| CLIP 图片编码延迟 | ~500ms |
| CLIP 文本检索延迟 | ~5ms (5000x512 IndexFlatIP) |

### 模型资源占用

| 模型 | 显存 | 首次加载 | 后续调用 |
|------|------|---------|---------|
| Qwen2.5-0.5B（本地）| ~1.5 GB | ~43s | ~1.5s |
| all-MiniLM-L6-v2 | ~0.3 GB | ~10s | ~30ms |
| BGE-Reranker-v2-m3 | ~0.5 GB | ~10s | ~30ms |
| CLIP-ViT-B/32 | ~0.6 GB | ~8s | ~500ms |

### 对话压缩指标

| 指标 | 值 |
|------|-----|
| 压缩前（9 轮对话）| 431 tokens |
| 压缩后（LLM 摘要）| 42 tokens |
| **压缩率** | **9.7% 节省 90.3% Token** |
| 后台压缩耗时 | ~1-2s（不阻塞主响应）|

---

## 🧰 技术栈

| 类别 | 技术 |
|------|------|
| **编排** | LangGraph StateGraph (12 节点) + 自定义 Worker 派发 |
| **LLM** | Qwen3.5-35B-A3B (API) / Qwen2.5-0.5B (本地回退) |
| **嵌入** | all-MiniLM-L6-v2 (384dim) |
| **检索** | BM25 + FAISS (IndexFlatIP) + BGE-Reranker + CLIP 图文检索 |
| **熔断** | 异步三态 Circuit Breaker (CLOSED / OPEN / HALF_OPEN) |
| **长效记忆** | LongTermMemory (JSON 文件持久化, 零外部依赖) |
| **工具注册** | ToolRegistry (MCP 兼容目录, 自动发现) |
| **PDF 解析** | PyMuPDF (快速模式) + magic-pdf / MinerU (增强模式) — 自动降级 |
| **SQL** | sqlite3 + sqlglot (AST 校验) |
| **API** | FastAPI + SSE 流式 |
| **日志** | structlog |
| **配置** | pydantic-settings |
| **部署** | Docker / docker-compose |

---

## ⚠️ 免责声明

本项目为个人学习项目，食谱数据来源于 Eitanli/cuisine_type 公开数据集的测试子集，
仅用于非商业研究和演示目的。营养建议由 LLM 生成，仅供参考。
