# ❓ 问题记录 & 解决方案

本项目开发过程中遇到并解决的问题汇总，按模块分类。

---

## 目录

1. [LLM / 模型加载](#1-llm--模型加载)
2. [检索与向量索引](#2-检索与向量索引)
3. [数据与分块策略](#3-数据与分块策略)
4. [意图识别 (Router)](#4-意图识别-router)
5. [Agent 执行](#5-agent-执行)
6. [SQL 查询](#6-sql-查询)
7. [质量检测 (Critic)](#7-质量检测-critic)
8. [编排器 (Orchestrator)](#8-编排器-orchestrator)
9. [多轮对话与 Critic 重试循环](#9-多轮对话与-critic-重试循环)

---

## 1. LLM / 模型加载

### Q1.1 每个 Agent 都加载一次模型，显存爆炸

**问题描述：** 每个 Agent（Router、TextRAG、NutritionSQL 等）各自从 HuggingFace 加载 Qwen2.5-0.5B-Instruct，导致显存重复占用，首 Agent 加载耗时 ~43s，后续 Agent 又各自加载数倍显存。

**根因：** 无共享模型实例，每个 Agent 独立初始化 pipeline。

**解决方案：** 实现线程安全的全局单例

```python
# src/core/utils/llm.py
_model_lock = asyncio.Lock()
_model_instance = None  # 全局单例, lazy load
```

- 首调加载 ~43s / ~1.5GB 显存
- 后续调用 ~1.5-2s（复用 pipeline）
- 所有 Agent 通过 `LLMTool.generate()` 共享同一实例

**涉及文件：** `src/core/utils/llm.py`

---

### Q1.2 模型生成阻塞事件循环

**问题描述：** LLM 的 `generate()` 是同步阻塞调用，在 FastAPI 异步接口中会阻塞事件循环。

**根因：** Transformers 的 `model.generate()` 是 CPU/GPU 密集型同步操作。

**解决方案：** 在 `_generate_sync()` 中运行同步代码，通过 `asyncio.to_thread()` 抛到线程池执行。

```python
async def generate(...):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, self._generate_sync, ...)
```

**涉及文件：** `src/core/utils/llm.py`

---

### Q1.3 all-MiniLM-L6-v2 下载失败

**问题描述：** 从 HuggingFace 下载 all-MiniLM-L6-v2 反复超时或失败。

**根因：** 网络环境需走 HF 镜像。

**解决方案：** 优先使用本地缓存路径 `/home/luguanghui/PRNet/REMOTE-main/ROMOTE_code/models/all-MiniLM-L6-v2`，`SentenceTransformer` 自动兼容本地目录和 HF hub。

**涉及文件：** `src/core/retrievers/hybrid.py`、`scripts/ingest_real_data.py`

---

### Q1.4 本地 0.5B 模型回答质量差 / 翻译不稳定

**问题描述：** Qwen2.5-0.5B 能力有限，营养类查询的翻译会"回答"而非"翻译"
（如"鸡肉的热量是多少"输出一整段营养数据），
SQL 翻译准确率低，食材替换推理不够精准。

**根因：** 0.5B 模型的语义理解、指令遵循和世界知识都有限，不足以支撑实际问答场景。

**解决方案：** 改为 **API 优先 + 本地回退** 架构：
1. 新增 `_api_generate_sync()` 函数，使用 OpenAI 兼容客户端调用 ModelScope API（`Qwen/Qwen3.5-35B-A3B`）
2. 所有 LLM 调用（`generate()` / `translate_query()` / `generate_structured()`）先走 API
3. API 成功 → 直接返回；API 失败（网络 / 认证 / 超时） → 自动回退到本地 0.5B
4. 配置通过 `.env` 文件读取，兼容任意 OpenAI 格式 API：

```bash
RECIPE_LLM_API_KEY=your_key
RECIPE_LLM_BASE_URL=https://api-inference.modelscope.cn/v1
RECIPE_LLM_MODEL=Qwen/Qwen3.5-35B-A3B
```

**调用流程：**
```
translate_query / generate
  ├─ API 调用 (35B) → 成功 → 返回
  ├─ API 失败 (异常) → 日志警告 → 回退本地 (0.5B)
  └─ API 未配置 (无 Key) → 直接走本地
```

**效果：**
- 翻译："鸡肉的热量是多少" ✅ → "How many calories are in chicken?"
- 生成：35B 模型的回答质量、指令遵循显著优于 0.5B
- 容错：API 宕机时自动降级，服务不中断

**涉及文件：** `src/core/utils/llm.py`、`.env`

---

## 2. 检索与向量索引

### Q2.1 BM25 加载失败: `'dict' object has no attribute 'get_scores'`

**问题描述：** `HybridRetriever._load_bm25()` 从 pickle 直接读取，但新脚本保存的是 `{'bm25': bm25_object, 'chunk_ids': ..., 'texts': ...}` 字典格式，加载后赋值给 `self._bm25` 导致调用 `get_scores()` 时报错。

**根因：** 写入脚本（`ingest_real_data.py` 和索引重建脚本）保存为 dict 格式，但读取代码仍然按旧的纯 BM25 对象处理。

**解决方案：** `_load_bm25()` 兼容两种格式：

```python
def _load_bm25(self):
    ...
    data = pickle.load(f)
    if isinstance(data, dict):
        self._bm25 = data["bm25"]
    else:
        self._bm25 = data
```

**涉及文件：** `src/core/retrievers/hybrid.py`

---

### Q2.2 BM25 ↔ Chunk 映射错位

**问题描述：** `_bm25_search()` 中使用 `list(self._chunks_map.values())` 按 BM25 索引取 chunk，但 dict 的 `values()` 顺序可能不一致导致映射错误。

**根因：** BM25 索引基于文档列表的固定顺序，但 chunks_map 为 dict 结构，依赖插入顺序做对齐不可靠。

**解决方案：** 在 bm25.pkl 中同时保存与 BM25 对齐的 `chunk_ids` 列表，检索时据此映射：

```python
# ingest 保存时：
pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

# 检索时：
self._bm25_chunk_ids = data.get("chunk_ids")  # 优先使用
id_list = self._bm25_chunk_ids or list(self._chunks_map.keys())  # 降级
```

**已修复时间：** 2026-06-04

**涉及文件：** `src/core/retrievers/hybrid.py`、`scripts/ingest_real_data.py`

---

---

### Q2.3 BM25 / FAISS / Rerank 三分值量纲不同

**问题描述：** 三路检索加权融合时，BM25 分数（通常 0~N）、FAISS cosine 相似度（-1~1）、BGE-Reranker 分数（0~1）尺度不同，直接加权导致某一路主导。

**根因：** 分数未归一化。

**解决方案：** 改用 **RRF（Reciprocal Rank Fusion）** 代替加权融合，基于排名而非原始分数：

```python
K = 60
for rank, c in enumerate(results):
    score = 1.0 / (K + rank + 1)  # 排名越靠前得分越高
```

RRF 天然不受分数量纲影响，BM25、FAISS、Rerank 只需各自返回排序结果即可。

**已修复时间：** 2026-06-04

**涉及文件：** `src/core/retrievers/hybrid.py`

---

### Q2.4 BM25 对中文无效

**问题描述：** BM25 使用 `query.lower().split()` 做 tokenization，对中文文本无法切分（中文无空格），导致含中文的 query 检索结果为 0。

**根因：** 简单的 whitespace tokenizer 不支持 CJK 语言。

**解决方案：** 加入语言感知分词，根据文本是否含中文字符自动切换：

```python
def _tokenize(text: str) -> list[str]:
    if _CJK_RE.search(text):  # 含中文字符
        return list(jieba.cut(text))  # jieba 分词
    return text.lower().split()  # 英文 whitespace 切分
```

同时适配 BM25 构建（`ingest_real_data.py`）和检索（`hybrid.py`）。

**已修复时间：** 2026-06-04

**涉及文件：** `scripts/ingest_real_data.py`、`src/core/retrievers/hybrid.py`



## 3. 数据与分块策略

### Q3.1 步骤按条切分导致碎片化

**问题描述：** 原始策略将某菜谱的每一步骤拆成独立 chunk（`Step 1`, `Step 2`, ..., `Step N`），检索时可能只命中 "Step 1: Preheat the oven..."，LLM 拿不到完整步骤上下文，无法准确回答"第二步要做什么"。

**根因：** 分块粒度过细，步骤间上下文割裂。

**解决方案：** 某菜谱的全部步骤合并为单个 chunk，保留编号：

```python
numbered = "\n".join(
    f"Step {i+1}. {s.strip()}" for i, s in enumerate(recipe["steps"]) if s.strip()
)
```

- 生成 chunk_id: `real_00000_steps`
- chunk 数从 15,314 降至 **15,000**（steps section 从 5,314 个碎片→5,000 个完整块）
- LLM 在完整步骤上下文中自行定位具体某一步

**涉及文件：** `scripts/ingest_real_data.py`

---

### Q3.2 Chunk 内容含中文前缀

**问题描述：** Step chunk 以 `步骤 N:` 开头，Ingredients 以 `食材:` 开头，Metadata 以 `菜系:` / `简介:` 开头。英文 query 检索到中文关键词时匹配度低。

**根因：** 最初设计面向中文用户，使用了中文标签前缀。

**解决方案：** 统一改为英文前缀：

| 旧（中文） | 新（英文） |
|-----------|-----------|
| `步骤 N:` | `Step N.` |
| `食材:` | `Ingredients:` |
| `菜系:` | `Cuisine:` |
| `简介:` | `Description:` |

**涉及文件：** `scripts/ingest_real_data.py`

---

### Q3.3 数据库未填充（NutritionSQL 不可用）

**问题描述：** NutritionSQL Agent 尝试执行 LLM 翻译的 SQL 时，SQLite 数据库 `recipe.db` 不存在或为空表，导致 SQL 执行始终失败。

**根因：** 数据脚本只生成 JSON 和向量索引，未写入 SQLite。

**解决方案：** 编写 `scripts/seed_db.py` 将 `recipes_real.json` 导入 SQLite，建四表：

```sql
recipes (id, title, cuisine, description, cook_time, ...)
ingredients (recipe_id, name, amount, unit)
nutrition (recipe_id, calories, protein, fat, ...)
steps (recipe_id, step_number, instruction)
```

**涉及文件：** `scripts/seed_db.py`

---

### Q3.4 向量索引与数据不同步

**问题描述：** 修改 chunks.json 后，FAISS index 和 BM25 模型未同步重建，检索返回旧数据或 chunk_id 失效。

**根因：** 索引构建与数据生成是分离步骤，缺乏自动化流水线。

**解决方案：** 在 Makefile 加入一键重建 target，一条命令完成全流程：

```bash
make rebuild-index
# 等价于:
# SKIP_DOWNLOAD=1 MODEL_PATH=<local_path> python3 scripts/ingest_real_data.py
```

- 从本地 `recipes_real.json` 重新分块
- 自动重建 FAISS + BM25 索引
- 使用本地缓存的 all-MiniLM-L6-v2（避免 HF 下载）

**已修复时间：** 2026-06-04

**涉及文件：** `scripts/ingest_real_data.py`、`Makefile`

---

## 4. 意图识别 (Router)

### Q4.1 Router 词表以英文为主

**问题描述：** 原 Router 关键词以中文为主，英文 query 被误分类为 chitchat。

**根因：** 关键词表偏向中文，且英文覆盖率不足。

**解决方案：** 重构为 **英文优先、中文辅助** 的规则引擎：
- 每个意图的英文关键词放在前面（优先匹配），中文放在后面兜底
- 补充了更多英文短语（如 `"how do i"`, `"give me"`, `"what to"`, `"leave out"` 等）
- 调整意图检查顺序：`INGREDIENT_RECOMMEND → STEP_QA → NUTRITION → SUBSTITUTION → IMAGE → CHITCHAT`

**已修复时间：** 2026-06-04

**涉及文件：** `src/agents/router.py`

---

## 5. Agent 执行

### Q5.1 ImageSearchAgent 为空实现

**问题描述：** Image Search Agent 只返回占位信息"Image search is a stub"，无法处理图片检索请求。

**根因：** CLIP 模型未接入，图文检索链路未实现。

**解决方案：**
- 短方案：回退到 TextRAG 的 HybridRetriever，至少返回文本检索结果
- 长方案：接入 CLIP（接口已预留 `src/core/retrievers/clip_retriever.py`）

**涉及文件：** `src/agents/image_search.py`、`src/core/retrievers/clip_retriever.py`

---

### Q5.2 NutritionSQL 无兜底策略

**问题描述：** SQL 翻译或执行失败时，Agent 直接报错，没有 fallback。

**根因：** 单链路无备份方案。

**解决方案：** 改为双路策略：
1. **主路（SQL）：** LLM 翻译→SQLValidator 校验→SQLSandbox 执行→LLM 总结
2. **备路（Retrieval）：** 主路失败时自动回退到 HybridRetriever + LLM 生成回答

**涉及文件：** `src/agents/nutrition_sql.py`

---

### Q5.3 LLM 输出含代码块导致解析失败

**问题描述：** 某些 Agent 期望 JSON 结构化输出（如 Substitution, NutritionSQL），但 LLM 返回了 markdown 代码块包裹的内容。

**根因：** LLM 遵循对话格式习惯，在 JSON 外包裹了 ` ```json ... ``` `。

**解决方案：** 在处理 LLM 输出前自动剥离 markdown 代码块标记：

```python
import re
def _strip_code_block(text: str) -> str:
    return re.sub(r'```(?:json|sql)?\n?(.*?)```', r'\1', text, flags=re.DOTALL).strip()
```

**涉及文件：** `src/agents/nutrition_sql.py`、`src/agents/substitution.py`

---

## 6. SQL 查询

### Q6.1 自然语言→SQL 翻译质量不稳定

**问题描述：** LLM 翻译的 SQL 有时语法错误、表名列名不存在或语义偏离用户意图。

**根因：** 0.5B 模型 SQL 能力有限，SQL schema 上下文不足。

**解决方案：**
- 在 `SQL_TRANSLATION_PROMPT` 中提供完整表和列定义
- 使用 `sqlglot` 做 AST 校验，只允许 SELECT 语句
- 表白名单（仅允许 `recipes`, `ingredients`, `nutrition`, `steps`）
- 超时熔断 3s
- 失败自动回退检索兜底

**涉及文件：** `src/agents/nutrition_sql.py`、`src/core/sql/validator.py`、`src/core/sql/sandbox.py`

---

## 7. 质量检测 (Critic)

### Q7.1 Critic 漏检英文营养免责声明

**问题描述：** Critic Agent 检查回答中是否包含营养免责声明，但只检测中文关键词（如"仅供参考"、"建议咨询"），英文回答中无对应检测。

**根因：** 关键词规则仅覆盖中文。

**解决方案：** 添加英文免责关键词：

| 语言 | 关键词 |
|------|--------|
| 中文 | 仅供参考、建议咨询、不构成医疗建议 |
| 英文 | not medical advice, for reference only, consult a professional, informational purposes |

**涉及文件：** `src/agents/critic.py`

---

### Q7.2 Formatter 不自动附加免责声明

**问题描述：** 即使 Critic 检测到营养类回答缺免责声明，Formatter 也未自动补充。

**根因：** Critic 返回 `disclaimer_missing` 标志，但 Formatter 未利用。

**解决方案：** Formatter 检查 Critic 的 `disclaimer_missing` 标志，若为 True 则在回答末尾附加免责声明段落。

**涉及文件：** `src/agents/formatter.py`

---

## 8. 编排器 (Orchestrator)

### Q8.1 LangGraph 状态图：从"备用"到"激活"

**问题描述：** 最初使用 LangGraph StateGraph 作为编排引擎（11 节点），但第一阶段场景是单轮问答，LangGraph 偏重。

**历程：**
1. **第一阶段（单轮）：** 改用自定义 Supervisor（`RecipeOrchestrator.ask()`），顺序执行 Router → Worker+Fallback 并行 → Critic → Formatter。LangGraph 代码保留在 `src/orchestrator/graph.py` 备用。
2. **第二阶段（多轮）：** 用户确认需要多轮对话 + Agent 重试循环后，**重新激活 LangGraph**。将自定义 Supervisor 的逻辑迁移到 StateGraph。

**迁移内容：**
- **新增节点：** `init`（初始化 state）、`load_memory`（加载 session 历史）、`save_memory`（保存对话轮次）、`chitchat_direct`（闲聊直出）、`revision`（重试计数器自增）
- **统一 worker：** 原独立的 TextRAG/NutritionSQL/Substitution 节点合并为单一 `worker_node`，按 `state["intent"]` 派发
- **条件边：** `route_by_intent`（router → worker / chitchat_direct）、`after_critic`（critic → save_memory / revision）
- **Critic 重试回边：** `revision` → `worker`（注入 critic_suggestions 作为 feedback），最大 2 次
- **Worker 内部并行：** node 内 `asyncio.gather(worker, fallback)` 保持原有并行策略

**效果：** 10 节点 StateGraph，支持多轮记忆 + Critic 自动重试，单轮往返延迟与原 Supervisor 一致（Worker+Fallback 并行无额外开销）。

### Q8.2 Worker 失败时 Critic 访问不到 result.answer

**问题描述：** TextRAG / NutritionSQL 失败时返回 `AgentResult`（基类），不含 `.answer` / `.explanation` / `.chunks` 属性，导致 AttributeError。

**根因：** 失败时走入 `_fallback()` 返回 `AgentResult(success=False)`，但后续代码硬编码访问 `rag_result.answer`。

**解决方案：** 检查 `rag_result.success`：
- True → 正常读取 `.answer`、`.chunks`，然后走 Critic
- False → 直接切换到预计算兜底结果（`fallback.draft` + `fallback.chunks`），跳过 Critic

```python
if rag_result.success:
    draft = rag_result.answer
    # ...
    draft, provenance = await self._critic_with_fallback(...)
else:
    draft = fallback.draft
    provenance = fallback_provenance
    # 跳过 Critic
```

**涉及文件：** `src/orchestrator/supervisor.py`

---

## 9. 多轮对话与 Critic 重试循环

### Q9.1 ConversationMemory 会话管理设计

**问题描述：** 需要支持多轮对话时，对话历史存储在内存中，需要考虑自动清理和容量控制。

**解决方案：** 设计 `ConversationMemory` 类，按 `session_id` 管理：

```python
# src/core/memory.py
class ConversationMemory:
    # 类变量：全局会话存储
    _sessions: dict[str, "SessionRecord"] = {}
    _last_access: dict[str, float] = {}

    @classmethod
    def get_or_create(cls, session_id: str) -> "ConversationMemory":
        # 自动清理 3600s 无访问的过期会话
        # 全局最多 1000 个会话

    def add_turn(self, query: str, answer: str, intent: str, slots: dict, provenance: str):
        # 最多保留 20 轮

    def get_context(self, last_n: int = 3) -> str:
        # 返回最近 N 轮对话文本摘要

    def get_last_slots(self) -> dict:
        # 返回上一轮提取的槽位（菜名、食材等）

    def get_entity_summary(self) -> str:
        # 从所有历史轮次提取实体摘要
```

**关键设计：**
- **无持久化：** 会话仅存内存，服务重启后丢失（适合演示/开发阶段）
- **惰性清理：** `get_or_create` 时检查过期会话，不启动后台线程
- **上下文注入：** Router 的 `_fc_classify` 接收 `conversation_context` + `last_slots`，system prompt 注入历史

**涉及文件：** `src/core/memory.py`、`src/agents/router.py`

---

### Q9.2 Critic 重试循环设计

**问题描述：** Critic 发现回答质量不合格时（事实错误、免责缺失等），如何自动修复而不中断流程？

**解决方案：** 使用 LangGraph 回边（edge from `revision` to `worker`）实现重试循环：

```
critic → after_critic
   ├─ 通过 ──────────────→ save_memory → formatter → END
   └─ 不通过
       ├─ retry < max(2) → revision → worker (重试)
       └─ retry >= max(2) → save_memory → formatter → END (兜底)
```

**重试机制：**
- `revision` 节点：自增 `state["retry_count"]`，注入 `state["critic_suggestions"]` 到 `state["feedback"]`
- `worker_node` 接重试：检测到 `state.get("feedback")`，传给 Worker 的 `feedback` 参数
- Worker 重试时使用原文 + Critic 建议重新生成，不使用检索（避免随机性）
- 最大重试 2 次（`REVISION_MAX = 2`），仍失败则走预计算检索兜底

**为什么不是 while 循环？**
- LangGraph 的回边机制让状态图可视化，便于调试
- 条件边 + 计数器的模式与 StateGraph 的声明式风格一致
- 可扩展：未来可以给 revision 节点加不同的重试策略（如退火温度）

**涉及文件：** `src/orchestrator/graph.py`

---

### Q9.3 LangGraph 重新激活的决策过程

**问题描述：** 最初从 LangGraph 迁到自定义 Supervisor（Q8.1），现在又迁回来，原因是什么？

**决策树：**
```
需要多轮对话 + Agent 重试循环吗？
  ├─ 否 → 自定义 Supervisor 更简单（Q8.1 阶段）
  └─ 是 → LangGraph StateGraph 更适合
       ├─ 多轮：load_memory / save_memory 节点自然融入
       ├─ 重试：revision 回边比手动 while 循环更清晰
       └─ 并行：node 内 asyncio.gather 保留，不依赖 LangGraph 并行节点
```

**经验教训：**
- 技术选型应匹配**当前**需求，而非预留未来可能性
- LangGraph 的 StateGraph 在单轮场景偏重，但在多轮+重试场景恰到好处
- 预留切换路径（`graph.py` 一直保留）让回迁成本极低

**涉及文件：** `src/orchestrator/graph.py`、`src/orchestrator/supervisor.py`

---

## 附录：未解决问题（TODO）

| 问题 | 优先级 | 备注 |
|------|--------|------|
| CLIP 图文检索未实现 | 中 | 接口已预留，模型未接入 |
| MinerU PDF 解析未集成 | 低 | 组件已安装 |
| 幻觉检测交叉验证 | 低 | 需支持引用抽取 |
| Benchmark 基线未跑 | 中 | 框架就绪未执行 |

---

*最后更新: 2026-06-04*
