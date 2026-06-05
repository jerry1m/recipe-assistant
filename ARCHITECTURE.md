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
| **意图识别** | 7 类意图 + 槽位提取（Function Calling + 规则混合） |
| **食谱问答** | BM25 + FAISS + BGE-Reranker 三级检索 + LLM 生成 |
| **营养查询** | 关键词 SQL 模板 → LLM 翻译 SQL → 混合检索三路兜底 |
| **图文检索** | CLIP 零样本以图搜菜名 |
| **食材替换** | LLM 链式推理 + JSON 结构化输出 |
| **PDF 解析** | 双层引擎（PyMuPDF + magic-pdf），自动降级 |
| **多轮对话** | Redis 持久化 + Token 感知滑动窗口 + LLM 压缩 |
| **质量质检** | Critic 重试循环（最大 2 次） |

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
│  编排层 (src/orchestrator/) — LangGraph StateGraph (10 节点)        │
│                                                                      │
│  ┌──────────┐                                                        │
│  │  init    │ ← 初始化 state: request_id, start_time                │
│  └────┬─────┘                                                        │
│       ▼                                                              │
│  ┌─────────────┐                                                     │
│  │ load_memory │ ← 加载 Redis/内存中的对话历史                       │
│  └────┬────────┘                      ┌───────────────────────┐     │
│       ▼                                │  ConversationMemory   │     │
│  ┌──────────┐                          │  ┌─────────────────┐  │     │
│  │  Router  │── Intent + 槽位 ──────→  │  │ Redis (主存储)   │  │     │
│  └────┬─────┘                          │  ├─────────────────┤  │     │
│       │                                │  │ 内存兜底 (备用)  │  │     │
│       ▼ (按 intent 条件派发)            │  └─────────────────┘  │     │
│  ┌──────────┐                          └───────────────────────┘     │
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
│  ┌─────────────┐                                                     │
│  │ save_memory │ ← 保存对话 + 后台 asyncio.create_task(压缩)        │
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
│  Agent 层 (src/agents/) — 8 个 Agent 各司其职                       │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐      │
│  │ Router   │  │ TextRAG  │  │NutritionSQL│  │ ImageSearch  │      │
│  │ 意图识别  │  │ 食谱问答  │  │ 营养查询    │  │ 图文检索     │      │
│  └──────────┘  └──────────┘  └────────────┘  └──────────────┘      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │Critic    │  │Formatter │  │Substitut.│  │PDFParse  │            │
│  │ 质量质检  │  │ 输出格式化│  │ 食材替换  │  │PDF解析    │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
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
│  │  LLM 系统 (utils/llm.py) — API 优先 + 本地回退单例          │    │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐    │    │
│  │  │ ModelScope API   │  │ Qwen2.5-0.5B (本地, 全局单例) │    │    │
│  │  │ Qwen3.5-35B      │  │ ~1.5GB 显存, lazy load       │    │    │
│  │  └──────────────────┘  └──────────────────────────────┘    │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  安全沙箱 (sql/) — sqlglot AST 校验 + SQLite 只读执行       │    │
│  │  ┌──────────────┐  ┌──────────────────────────────────┐    │    │
│  │  │ validator.py │  │ sandbox.py (超时熔断 3s)         │    │    │
│  │  └──────────────┘  └──────────────────────────────────┘    │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  PDF 解析 (pdf_parser.py) — PyMuPDF + magic-pdf 双层      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  对话记忆 (memory.py) — Redis + Token 滑动窗口 + LLM 压缩  │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  监控 (utils/metrics.py) — Agent 调用统计 + 业务指标       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件详解

### 3.1 Router — 三阶段意图识别

```
query → 规则快速通道 (0ms, 正则匹配)
           │
   置信度 ≥ 0.92 ──→ 直接返回
           │
        < 0.92
           ↓
    Function Calling (Qwen3.5-35B, ~2s)
           │
      成功 ──→ intent + 槽位 (结构化 JSON)
           │
      失败
           ↓
       规则兜底 (置信度 0.7)
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

**关键代码**：`src/agents/router.py` — `_INTENT_TOOLS` 定义 7 个 OpenAPI Function

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

---

## 4. 数据流

### 4.1 单轮问答流程

```
Client → POST /ask {"query": "红烧肉怎么做？"}
  → FastAPI → RecipeOrchestrator.ask()
    → LangGraph graph.ainvoke(initial_state)
      → init_node: 生成 request_id
      → load_memory_node: 加载对话历史 (Redis)
      → router_node: Function Calling 识别 intent + 槽位
        → intent: ingredient_recommend, slots: {recipe_name: "红烧肉"}
      → worker_node: 按 intent 派发到 TextRAGAgent
        → asyncio.gather(TextRAG.run, _prepare_text_fallback)
          → HybridRetriever.retrieve("braised pork belly")
          → LLM generate (Qwen3.5-35B)
        → Worker 成功 → 走 Critic
      → critic_node: 质量检查
      → save_memory_node: 保存对话到 Redis + 后台触发 LLM 压缩
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
API 调用 → 超时/失败
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

---

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
