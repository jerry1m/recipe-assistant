"""
Recipe1M 风格真实食谱数据接入管道
数据源: Eitanli/cuisine_type (HuggingFace, ~13.5k 英文食谱)

工作流:
  1. 从 HF 流式下载全量食谱
  2. 解析为项目标准 Recipe 格式
  3. 语义分块 (ingredients / steps)
  4. 生成 sentence-transformers 嵌入
  5. 构建 FAISS 向量索引
  6. 构建 BM25 关键词索引
  7. 导出 JSON + 索引文件到 src/data/
  8. (可选) 图片增强: 支持 Bing 爬虫 / SD 生成 / Unsplash API

图片增强为独立步骤, 可后续按需运行:
  Bing 爬图:  SKIP_DOWNLOAD=1 SKIP_INDEX=1 python scripts/download_images.py
  SD 生成:   python scripts/generate_images_sd.py
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

# ── 配置 ──

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

DATA_DIR = PROJECT_ROOT / "src" / "data"
RECIPES_FILE = DATA_DIR / "recipes_real.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
FAISS_INDEX_FILE = DATA_DIR / "vector_store" / "recipes.index"
FAISS_ID_FILE = DATA_DIR / "vector_store" / "chunk_ids.npy"
BM25_FILE = DATA_DIR / "vector_store" / "bm25.pkl"
MAX_RECIPES = 5000  # 首次接入 5k，可增量扩展

# ── 辅助函数 ──


def parse_recipe_text(text: str) -> tuple[str, list[str], list[str]]:
    """
    将 cuisine_type 的 recipe 文本解析为 (title, ingredients, steps)

    格式示例:
        'Recipe Title
        1 cup flour
        2 tbsp sugar
        Instructions:
        Mix ingredients.
        Bake at 350.'
    """
    title_end = text.find("\n")
    title = text[:title_end].strip() if title_end > 0 else text.strip()[:80]

    # 尝试找 "Instructions:" 或 "Directions:" 或 "Method:"
    instr_markers = ["Instructions:", "Directions:", "Method:", "DIRECTIONS:"]
    instr_pos = None
    for marker in instr_markers:
        pos = text.find(marker)
        if pos != -1:
            instr_pos = pos
            break

    if instr_pos:
        ingredients_text = text[title_end:instr_pos].strip()
        instructions_text = text[instr_pos + len(marker):].strip()
    else:
        # 没有明确的分隔，把全部内容当作 ingredients
        ingredients_text = text[title_end:].strip()
        instructions_text = ""

    # 分行解析 ingredients
    ingredient_lines = []
    for line in ingredients_text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("Instructions", "Directions", "Method")):
            ingredient_lines.append(line)

    # 分步解析 instructions
    steps = []
    if instructions_text:
        for line in instructions_text.split("\n"):
            line = line.strip()
            if line and not line[0].isdigit():
                # 不是步骤编号，合并到上一步
                if steps:
                    steps[-1] += " " + line
                else:
                    steps.append(line)
            elif line:
                steps.append(line)

    return title, ingredient_lines, steps


# ── 食材行解析: "1 cup flour" → amount="1", unit="cup", name="flour" ──

_INGREDIENT_PATTERN = re.compile(
    r"^([\d\s\/\.\-]+)?\s*"       # 可选数量: 1, 1/2, 0.5, 2-3
    r"([a-zA-Z]+\.?)?\s+"          # 可选单位: cup, tsp, oz, lb
    r"(.+)$"                       # 食材名: flour, olive oil
)

# 常见厨艺单位列表
_COOKING_UNITS = {
    "cup", "cups", "tsp", "tsp.", "teaspoon", "teaspoons",
    "tbsp", "tbsp.", "tablespoon", "tablespoons",
    "oz", "oz.", "ounce", "ounces",
    "lb", "lb.", "lbs", "pound", "pounds",
    "g", "gram", "grams", "kg", "kilogram", "kilograms",
    "ml", "milliliter", "milliliters", "l", "liter", "liters",
    "pinch", "dash", "to", "taste",
    "clove", "cloves", "slice", "slices", "piece", "pieces",
    "can", "cans", "package", "packages", "bunch", "bunches",
}


def parse_ingredient_line(line: str) -> dict:
    """解析食材行

    >>> parse_ingredient_line("1 cup flour")
    {'name': 'flour', 'amount': '1', 'unit': 'cup'}
    >>> parse_ingredient_line("2 tbsp sugar")
    {'name': 'sugar', 'amount': '2', 'unit': 'tbsp'}
    >>> parse_ingredient_line("Olive oil, for sauteing")
    {'name': 'Olive oil, for sauteing', 'amount': '', 'unit': ''}
    """
    m = _INGREDIENT_PATTERN.match(line.strip())
    if m:
        amount = (m.group(1) or "").strip()
        unit = (m.group(2) or "").strip().rstrip(".")
        name = (m.group(3) or line).strip()
        # 如果第二段不是已知单位，整体当食材名
        if unit and unit.lower() not in _COOKING_UNITS:
            name = f"{unit} {name}" if unit else name
            unit = ""
        return {"name": name, "amount": amount, "unit": unit}
    return {"name": line.strip(), "amount": "", "unit": ""}


def recipe_to_chunks(recipe: dict) -> list[dict]:
    """将 Recipe dict 转为可检索的 Chunk 列表"""
    chunks = []
    rid = recipe["recipe_id"]

    # 1. 元数据块（名称+菜系+难度）
    meta_text = f"{recipe['name']}"
    if recipe.get("cuisine"):
        meta_text += f" | Cuisine: {recipe['cuisine']}"
    meta_text += f" | Description: {recipe.get('description', '')}"
    chunks.append({
        "chunk_id": f"{rid}_meta",
        "recipe_id": rid,
        "content": meta_text,
        "section": "metadata",
        "score": 1.0,
    })

    # 2. 食材块
    if recipe.get("ingredients"):
        ings = []
        for ing in recipe["ingredients"]:
            parts = [ing["name"]]
            if ing.get("amount"):
                parts.append(f"{ing['amount']}{ing.get('unit', '')}")
            ings.append(" ".join(parts))
        chunks.append({
            "chunk_id": f"{rid}_ingredients",
            "recipe_id": rid,
            "content": "Ingredients: " + "; ".join(ings),
            "section": "ingredients",
            "score": 1.0,
        })

    # 3. 步骤块（合并全部步骤，让 LLM 自行定位具体步骤）
    if recipe.get("steps"):
        numbered = "\n".join(
            f"Step {i+1}. {s.strip()}" for i, s in enumerate(recipe["steps"]) if s.strip()
        )
        if numbered:
            chunks.append({
                "chunk_id": f"{rid}_steps",
                "recipe_id": rid,
                "content": numbered,
                "section": "steps",
                "score": 1.0,
            })

    return chunks


def download_dataset(max_recipes: int = 5000) -> list[dict]:
    """从 HuggingFace 下载食谱数据集"""
    print(f"[1/6] 从 HF 下载食谱数据集 (max={max_recipes}) ...")
    from datasets import load_dataset

    ds = load_dataset("Eitanli/cuisine_type", split="train", streaming=True)

    recipes = []
    start = time.time()
    for i, example in enumerate(tqdm(ds, desc="下载食谱", total=max_recipes)):
        if i >= max_recipes:
            break

        text = example["recipe"]
        cuisine = example["cuisine_type"]

        title, ingredient_lines, steps = parse_recipe_text(text)

        recipe = {
            "recipe_id": f"real_{i:05d}",
            "name": title,
            "cuisine": cuisine,
            "difficulty": "",
            "prep_time": "",
            "cook_time": "",
            "servings": 0,
            "description": f"来自 cuisine_type 数据集, 菜系: {cuisine}",
            "ingredients": [
                parse_ingredient_line(line)
                for line in ingredient_lines
            ],
            "steps": steps if steps else [text],
            "tags": cuisine.replace(", ", ",").split(",") if cuisine else [],
            "image_url": "",
            "source": "Eitanli/cuisine_type",
        }
        recipes.append(recipe)

    elapsed = time.time() - start
    print(f"  ✓ 已下载 {len(recipes)} 个食谱 (耗时 {elapsed:.1f}s)")
    return recipes


def build_chunks(recipes: list[dict]) -> list[dict]:
    """为每个食谱生成检索用的 Chunk"""
    print(f"[2/6] 语义分块 ({len(recipes)} 个食谱) ...")
    all_chunks = []
    for recipe in tqdm(recipes, desc="分块"):
        chunks = recipe_to_chunks(recipe)
        all_chunks.extend(chunks)
    print(f"  ✓ 共生成 {len(all_chunks)} 个 Chunk")
    return all_chunks


def build_embeddings_and_faiss(chunks: list[dict]) -> tuple:
    """用 sentence-transformers 生成嵌入 + 构建 FAISS 索引"""
    print("[3/6] 生成嵌入向量 ...")
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("MODEL_PATH", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    texts = [c["content"] for c in chunks]
    batch_size = 256

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="编码"):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    print(f"  ✓ 嵌入维度: {embeddings.shape}")

    print("[4/6] 构建 FAISS 索引 ...")
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积 (余弦相似度近似)
    faiss.normalize_L2(embeddings.astype(np.float32))
    index.add(embeddings.astype(np.float32))
    print(f"  ✓ FAISS 索引大小: {index.ntotal} 个向量")

    return index, embeddings, [c["chunk_id"] for c in chunks]


def build_bm25(chunks: list[dict]):
    """构建 BM25 关键词索引"""
    print("[5/6] 构建 BM25 索引 ...")
    from rank_bm25 import BM25Okapi

    # 中文感知分词
    _HAS_JIEBA = False
    try:
        import jieba
        _HAS_JIEBA = True
    except ImportError:
        pass
    _CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

    def _tokenize(text: str) -> list[str]:
        t = text.lower().strip()
        if not t:
            return []
        if _CJK_RE.search(t):
            return list(jieba.cut(t)) if _HAS_JIEBA else list(t)
        return t.split()

    tokenized = [_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"  ✓ BM25 索引构建完成 ({len(chunks)} 文档)")
    return bm25


def save_all(recipes, chunks, faiss_index, chunk_ids, bm25):
    """保存所有中间产物"""
    print("[6/6] 保存到磁盘 ...")

    # 确保目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "vector_store").mkdir(parents=True, exist_ok=True)

    # 食谱数据
    with open(RECIPES_FILE, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {RECIPES_FILE} ({len(recipes)} 食谱)")

    # Chunks
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {CHUNKS_FILE} ({len(chunks)} chunks)")

    # FAISS 索引
    import faiss
    faiss.write_index(faiss_index, str(FAISS_INDEX_FILE))
    print(f"  ✓ {FAISS_INDEX_FILE}")

    # Chunk ID 映射
    np.save(str(FAISS_ID_FILE), np.array(chunk_ids, dtype=object))
    print(f"  ✓ {FAISS_ID_FILE}")

    # BM25 (附带 chunk_ids 确保映射对齐)
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
    print(f"  ✓ {BM25_FILE} ({len(chunk_ids)} 文档)")

    print("  🎉 全部保存完成!")


def generate_eval_queries(recipes: list[dict], n: int = 50):
    """从真实食谱数据生成评测查询 (与 benchmark 兼容的 schema)"""
    eval_file = PROJECT_ROOT / "src" / "eval" / "test_cases_real.json"

    queries = []
    for i, recipe in enumerate(recipes):
        if i >= n:
            break
        name = recipe["name"]
        cuisine = recipe.get("cuisine", "")
        ings = recipe.get("ingredients", [])[:3]
        ing_names = [x["name"] for x in ings]

        q = {
            "id": f"tc_{i:03d}",
            "query": f"推荐一道{cuisine}菜谱，包含{ing_names[0] if ing_names else name}",
            "intent": "ingredient_recommend",
            "expected_chunks": [recipe["recipe_id"]],
            "expected_sql": "",
        }
        queries.append(q)

    # 加一些 nutrition 查询
    for i, recipe in enumerate(recipes):
        if len(queries) >= n * 2:
            break
        name = recipe["name"]
        q = {
            "id": f"tc_{len(queries):03d}",
            "query": f"{name}的热量是多少？",
            "intent": "nutrition_filter",
            "expected_chunks": [recipe["recipe_id"]],
            "expected_sql": "",
        }
        queries.append(q)

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 已生成 {len(queries)} 条评测查询 -> {eval_file}")


if __name__ == "__main__":
    print("=" * 60)
    print("   Recipe Assistant — 真实食谱数据接入管道")
    print("=" * 60)

    # ── 跳过耗时步骤 (用于快速补图片) ──
    skip_download = os.environ.get("SKIP_DOWNLOAD", "").lower() in ("1", "true", "yes")
    skip_index = os.environ.get("SKIP_INDEX", "").lower() in ("1", "true", "yes")

    if skip_download and RECIPES_FILE.exists():
        print("ℹ SKIP_DOWNLOAD=1, 从本地加载已有食谱...")
        with open(RECIPES_FILE, "r") as f:
            recipes = json.load(f)
    else:
        # 1. 下载
        recipes = download_dataset()

    if skip_index:
        print("ℹ SKIP_INDEX=1, 跳过索引构建...")
        chunks = []
        faiss_index = None
        chunk_ids = []
        bm25 = None
    else:
        # 2. 分块
        chunks = build_chunks(recipes)

        # 3-4. 嵌入 + FAISS
        faiss_index, embeddings, chunk_ids = build_embeddings_and_faiss(chunks)

        # 5. BM25
        bm25 = build_bm25(chunks)

        # 6. 保存
        save_all(recipes, chunks, faiss_index, chunk_ids, bm25)

    # 7. 生成评测
    if not skip_index:
        generate_eval_queries(recipes, n=20)

    # 8. (图片增强为独立步骤, 后续按需运行)
    print("\n💡 图片增强说明:")
    print("   如需配图, 后续可单独运行独立脚本 (无需重复构建索引):")
    print("     Bing 爬图:  python scripts/download_images_bing.py")
    print("     SD 生成:    python scripts/generate_images_sd.py")
    print("     Unsplash:   UNSPLASH_ACCESS_KEY=xxx python scripts/download_images_bing.py")

    print("\n🎉 纯文本接入管道执行完毕!")
