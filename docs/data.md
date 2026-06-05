# 数据管道与知识库构建

## 概述

Recipe Assistant 的数据管道从 HuggingFace 真实食谱数据集出发，经过清洗、结构化、LLM 字段补充、语义分块、向量化索引等步骤，构建面向 RAG 的知识库。

---

## 一、数据源

| 项目 | 内容 |
|------|------|
| **数据源** | [Eitanli/cuisine_type](https://huggingface.co/datasets/Eitanli/cuisine_type) (HuggingFace) |
| **规模** | ~13.5k 英文食谱 |
| **接入量** | 5000 条（首次接入，支持增量扩展） |
| **原始格式** | 纯文本：每行包含 `recipe`（含标题、食材、步骤）和 `cuisine_type`（菜系标签） |
| **菜系覆盖** | 如 Italian, Chinese, Mexican, French, Indian, Japanese 等 |

### 原始数据示例

```
Garlic Shrimp Scampi
1 lb shrimp
4 cloves garlic
2 tbsp butter
Instructions:
Melt butter. Add garlic. Cook shrimp until pink. Serve over pasta.
```

---

## 二、数据接入管道

接入脚本: `scripts/ingest_real_data.py`

### 2.1 数据下载

```python
ds = load_dataset("Eitanli/cuisine_type", split="train", streaming=True)
```

使用 HuggingFace `datasets` 库流式下载，支持断点续传。通过环境变量 `HF_ENDPOINT=https://hf-mirror.com` 配置镜像源（国内网络）。

### 2.2 文本解析

将纯文本结构化:

| 字段 | 说明 | 来源 |
|------|------|------|
| `recipe_id` | 唯一 ID，格式 `real_{i:05d}` | 自增 |
| `name` | 食谱标题 | 文本首行 |
| `cuisine` | 菜系 | 数据集标签 |
| `ingredients` | 结构化食材列表 (name/amount/unit) | 文本正文，`parse_ingredient_line()` |
| `steps` | 步骤列表 | `Instructions:` 之后的分段 |
| `tags` | 标签 | 由 cuisine 拆分 |

**食材行解析** 使用正则提取:

```
^([\d\s\/\.\-]+)?\s*([a-zA-Z]+\.?)?\s+(.+)$
```

- 匹配 `数量 + 单位 + 名称` 模式: `"1 cup flour"` → `{amount: "1", unit: "cup", name: "flour"}`
- 已知 30+ 厨艺单位 (cup, tsp, tbsp, oz, lb, g, ml 等)
- 无法匹配的行整体作为食材名处理

### 2.3 初始空字段

接入后，以下字段留空/零值，后续由 LLM 补充:

```python
{
    "difficulty": "",           # 待补充: easy/medium/hard
    "prep_time": "",            # 待补充: "X mins"
    "cook_time": "",            # 待补充: "X mins"
    "servings": 0,              # 待补充: int
    "nutrition": {              # 待补充
        "calories": 0.0,
        "protein": 0.0,
        "fat": 0.0,
        "carbs": 0.0,
        "fiber": 0.0,
        "sodium": 0.0
    },
    "image_url": ""             # 可选后续配图
}
```

---

## 三、LLM 字段补充

### 3.1 问题背景

5000 条食谱的 nutrition/difficulty/prep_time/cook_time/servings 字段为空，需要自动填充。

### 3.2 方案选型

| 因素 | 选择 |
|------|------|
| **模型** | Qwen2.5-0.5B-Instruct (本地推理) |
| **硬件** | RTX 5070 12GB (GPU 0 专用) |
| **框架** | transformers 4.57.6, PyTorch |
| **精度** | `torch_dtype="auto"` (自动半精度) |
| **镜像** | HF_ENDPOINT=https://hf-mirror.com |

### 3.3 V1 尝试（失败）

```python
BATCH_SIZE = 40    # ❌ 太大
temperature = 0.3
```

- 0.5B 模型无法输出 40 个元素的 JSON 数组
- 输出截断、JSON 格式崩溃、96% 失败率
- 仅 209/5000 条成功（~4%）

### 3.4 V2 修复（成功）

```python
BATCH_SIZE = 5     # ✅ 小 batch
temperature = 0.3
do_sample = True
top_p = 0.9
```

同时引入 4 层容错 JSON 解析:

1. 提取 ` ```json [...] ``` ` 代码块
2. 直接 `json.loads()`
3. 找 `[` `]` 范围后解析
4. 逐对象正则匹配 `\{[^}]+\}`

结果: **3074/5000 成功 (61.5%)**。失败原因: 0.5B 模型能力边界，约 40% 输出无法解析为有效 JSON。

### 3.5 重试补充

脚本: `scripts/retry_failed_enrichment.py`

针对失败 1926 条使用强化策略:

| 策略 | V2 原版 | 重试版 |
|------|---------|--------|
| BATCH_SIZE | 5 | **2** |
| Temperature | 0.3 | **0.5** |
| Top-p | 0.9 | **0.95** |
| 重试次数 | 1 | **≤3 (不同 seed) + greedy fallback** |
| 数据校验 | 无 | **validate_parsed()** (字段齐全 + 非零营养) |

最终结果: 覆盖率从 61.5% 提升至 **97.7%** (4887/5000)，仅 113 条失败。

### 3.6 补充里程碑

| 轮次 | 策略 | 成功 | 覆盖率 | 耗时 |
|------|------|------|--------|------|
| V1 | batch=40, temp=0.3 | 209 | 4.2% | ~5 min |
| V2 | batch=5, temp=0.3, 4层容错 | 3074 | 61.5% | ~32 min |
| 重试第1轮 | batch=2, temp=0.5, ≤3次重试+seed | 4462 | 89.2% | ~16 min |
| 重试第2轮 | 同上, 对剩余538条 | **4887** | **97.7%** | ~5 min |

### 3.7 最终数据质量

- **4887/5000 (97.7%)** 成功补充
- `prep_time=0`: 仅 2 条异常
- `servings=0`: 0 条
- `difficulty` 异常(非 easy/medium/hard): 仅 9 条
- 剩余 113 条(2.3%) 为 0.5B 模型能力边界，需换更大模型才能覆盖

### 3.8 补充后数据结构

```json
{
    "name": "Roasted Carrots with Pecan Pesto",
    "difficulty": "easy",
    "prep_time": "15 mins",
    "cook_time": "30 mins",
    "servings": 4,
    "nutrition": {
        "calories": 350.0,
        "protein": 20.0,
        "fat": 12.0,
        "carbs": 45.0,
        "fiber": 0.0,
        "sodium": 0.0
    }
}
```

> 注: `fiber` 和 `sodium` 保持 0.0（模型未推断，可在后续 SQL 查询中用作默认值）。

### 3.9 检查点恢复机制

两套检查点互不干扰:

| 文件 | 用途 | 保存频率 |
|------|------|---------|
| `recipes_real.checkpoint.json` | V2 主补充检查点 | 每 20 batch |
| `recipes_real.retry_checkpoint.json` | 重试补充检查点 | 每 50 batch |

脚本启动时自动从对应检查点恢复，避免重复处理。

---

## 四、知识库构建

### 4.1 语义分块

脚本 `ingest_real_data.py` 中 `recipe_to_chunks()` 为每条食谱生成 3 类 Chunk:

| Chunk 类型 | ID 格式 | 内容 |
|-----------|---------|------|
| 元数据 | `{rid}_meta` | 名称 + 菜系 + 简介 |
| 食材 | `{rid}_ingredients` | 所有食材列表 |
| 步骤 | `{rid}_step_{i}` | 单步操作 |

5000 条食谱 → **15,314 Chunks**（平均每食谱 ~3 块）。

### 4.2 向量索引 (FAISS)

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatIP(dim)   # 内积 (余弦相似度)
faiss.normalize_L2(embeddings)
index.add(embeddings)
```

| 参数 | 值 |
|------|-----|
| 嵌入模型 | `all-MiniLM-L6-v2` (384 维) |
| 索引类型 | `IndexFlatIP` (暴力内积) |
| 编码 batch | 256 |
| 存储路径 | `src/data/vector_store/recipes.index` |

### 4.3 BM25 关键词索引

```python
from rank_bm25 import BM25Okapi
tokenized = [c["content"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized)
```

- 基于 Chunk 文本的分词构建
- 存储路径: `src/data/vector_store/bm25.pkl`

### 4.4 知识库文件清单

```
src/data/
├── recipes_real.json          # 5000 条标准化食谱
├── recipes_real.checkpoint.json   # V2 LLM 补充检查点
├── recipes_real.retry_checkpoint.json  # 重试检查点
├── chunks.json                # 可检索的 Chunk 文本
├── images/                    # (可选) 配图
├── pdf_books/                 # (可选) 外挂知识库
└── vector_store/
    ├── recipes.index          # FAISS 向量索引
    ├── chunk_ids.npy          # FAISS → chunk_id 映射
    └── bm25.pkl              # BM25 关键词索引
```

---

## 五、运行时检索

检索由 `src/core/retrievers/hybrid.py` 实现:

```
query
  ├── BM25 关键词检索 ─┐
  ├── FAISS 向量检索 ──┤── 加权融合 ──→ 重排序 (BGE Reranker) ──→ Top-K
  └── (SQL 营养查询)  ─┘
```

| 组件 | 权重 (config.py) | 存储位置 |
|------|------------------|---------|
| BM25 | `bm25_weight` | `vector_store/bm25.pkl` |
| FAISS | `vector_weight` | `vector_store/recipes.index` |
| Reranker | `rerank_weight` | 模型 `BAAI/bge-reranker-v2-m3` |

---

## 六、遇到的主要问题与解决

| 问题 | 原因 | 解决 |
|------|------|------|
| HuggingFace 无法访问 | 国内网络限制 | `HF_ENDPOINT=https://hf-mirror.com` |
| LLM batch=40 全失败 | 0.5B 模型无法生成 40 个 JSON | 降至 batch=5 (V2) / batch=2 (重试) |
| 60% 覆盖率上限 | 0.5B 模型能力边界 | 4 层 JSON 容错 + 重试 (seed/greedy) |
| 食材行 `2-3 tbsp` 解析 | 范围格式 | 正则 `\d+(?:\.\d+)?` 取第一个数字 |
| CUDA 显存不足 | GPU 1 被 TMR 训练占用 | `CUDA_VISIBLE_DEVICES=0` 隔离 GPU 0 |
| 文本无 `Instructions:` 分隔 | 部分食谱格式不规范 | fallback: 全部当食材处理 |
| `ingest_real_data.py` 缺少 `import re` | 编辑遗漏 | 添加 `import re` 修复 |
| `recipe_to_chunks()` 无返回值 | 函数缺少 `return chunks` | 添加 `return chunks` + 拆分出 `download_dataset()` |
| 图片缺失 | 未接入配图 | 独立脚本按需运行 (Bing/SD/Unsplash) |

---

## 七、后续优化方向

1. **LLM 补充** — 若需 95%+ 覆盖率，可换 Qwen2.5-1.5B-Instruct 或 API 模型
2. **图像增强** — 运行 `download_images_bing.py` 或 `generate_images_sd.py` 为食谱配图
3. **知识库外挂** — 将 PDF 书籍 (`src/data/pdf_books/`) 切块注入向量库
4. **增量扩展** — 修改 `ingest_real_data.py` 的 `MAX_RECIPES` 从 5000 → 10000
5. **混合精度索引** — 对万级以上数据可换 `faiss.IndexIVFFlat` 加速检索
