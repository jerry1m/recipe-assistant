"""
用本地 LLM (Qwen2.5-0.5B-Instruct) 推断补充食谱的缺失字段:
  - difficulty (easy/medium/hard)
  - prep_time / cook_time (分钟)
  - servings (人数)
  - nutrition (calories, protein, fat, carbs)

用法:
  HF_ENDPOINT=https://hf-mirror.com python scripts/enrich_with_llm.py

策略:
  - 使用 Qwen2.5-0.5B-Instruct 本地推理 (RTX 5070, 12GB)
  - 每批 5 条 (小 batch 保证 JSON 输出质量)
  - 从 checkpoint 恢复: 只处理 nutrition 为空的食谱
  - 每 20 批自动保存检查点

流程:
  1. 加载现有 recipes_real.json (或 .checkpoint.json)
  2. 筛选出 nutrition 全空的食谱
  3. 批量送入 LLM 推断
  4. 解析结构化输出
  5. 合并回完整数据集并保存
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

DATA_DIR = PROJECT_ROOT / "src" / "data"
RECIPES_FILE = DATA_DIR / "recipes_real.json"

# ── 配置 ──
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 5     # 小 batch 保证 JSON 输出质量
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.3       # 低温度让输出更稳定
CHECKPOINT_INTERVAL = 20  # 每 N 批保存一次检查点
DEVICE = "cuda"

# ── 系统提示 ──
SYSTEM_PROMPT = """Return JSON array only. Each element: {"difficulty":"easy|medium|hard","prep_time":int,"cook_time":int,"servings":int,"calories":int,"protein_g":int,"fat_g":int,"carbs_g":int}

Example:
[{"difficulty":"easy","prep_time":15,"cook_time":30,"servings":4,"calories":350,"protein_g":20,"fat_g":12,"carbs_g":45}]"""


def format_recipe(recipe: dict) -> str:
    """将单条食谱格式化为 LLM 可读的文本"""
    name = recipe.get("name", "")
    cuisine = recipe.get("cuisine", "")
    ingredients = recipe.get("ingredients", [])
    steps = recipe.get("steps", [])

    ing_text = "; ".join(
        f"{i.get('amount','')} {i.get('unit','')} {i.get('name','')}".strip()
        for i in ingredients
    )
    step_text = " | ".join(s.strip() for s in steps[:5])

    return f"""Name: {name}
Cuisine: {cuisine}
Ingredients: {ing_text[:300]}
Steps: {step_text[:300]}"""


def build_batch_prompt(batch: list[dict]) -> str:
    """为一批食谱构建用户提示"""
    parts = []
    for i, recipe in enumerate(batch):
        parts.append(f"[Recipe {i+1}]\n{format_recipe(recipe)}")

    return "\n\n".join(parts)


def _extract_int(value, default=0) -> int:
    """从可能的字符串中提取整数, 如 '5 minutes' → 5, '4-6' → 4, 6.5 → 6"""
    if isinstance(value, (int, float)):
        return round(value)
    if not isinstance(value, str):
        return default
    # 取第一个数字
    nums = re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))
    return round(float(nums[0])) if nums else default


def _is_empty_recipe(recipe: dict) -> bool:
    """检查食谱的 nutrition 等字段是否为空(需补充)"""
    return recipe.get("nutrition", {}).get("calories", 0) == 0


def parse_llm_output(text: str, n_expected: int) -> list[dict | None]:
    """从 LLM 输出中解析 JSON 结果，容错处理"""
    # 尝试提取 ```json ... ``` 代码块
    json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if json_match:
        text = json_match.group(1)

    # 尝试直接解析 JSON
    text = text.strip()
    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results[:n_expected]
    except json.JSONDecodeError:
        pass

    # 尝试找到第一个 [ 和最后一个 ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            results = json.loads(text[start:end + 1])
            if isinstance(results, list):
                return results[:n_expected]
        except json.JSONDecodeError:
            pass

    # 逐行尝试解析对象
    results = []
    obj_pattern = re.compile(r'\{[^}]+\}')
    for match in obj_pattern.finditer(text):
        try:
            obj = json.loads(match.group())
            results.append(obj)
        except json.JSONDecodeError:
            pass

    if results:
        return results[:n_expected]

    return [None] * n_expected


def enrich_recipes(recipes: list[dict]) -> list[dict]:
    """主干：批量推断并补充缺失字段"""
    print(f"加载模型 {MODEL_NAME} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"  模型已加载: {model.device}")

    enriched = []
    failed = 0
    total_batches = (len(recipes) + BATCH_SIZE - 1) // BATCH_SIZE

    for start_idx in tqdm(range(0, len(recipes), BATCH_SIZE), desc="LLM 推断"):
        batch = recipes[start_idx:start_idx + BATCH_SIZE]
        user_prompt = build_batch_prompt(batch)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_llm_output(response, len(batch))

        for i, recipe in enumerate(batch):
            if i < len(parsed) and parsed[i] is not None:
                data = parsed[i]
                recipe["difficulty"] = str(data.get("difficulty", "")).lower()
                recipe["prep_time"] = f"{_extract_int(data.get('prep_time', 0))} mins"
                recipe["cook_time"] = f"{_extract_int(data.get('cook_time', 0))} mins"
                recipe["servings"] = _extract_int(data.get("servings", 0))
                recipe["nutrition"] = {
                    "calories": float(_extract_int(data.get("calories", 0))),
                    "protein": float(_extract_int(data.get("protein_g", 0))),
                    "fat": float(_extract_int(data.get("fat_g", 0))),
                    "carbs": float(_extract_int(data.get("carbs_g", 0))),
                    "fiber": 0.0,
                    "sodium": 0.0,
                }
            else:
                failed += 1

        enriched.extend(batch)

        # 打印第一条示例
        if start_idx == 0 and parsed[0] is not None:
            print(f"\n[sample] {batch[0].get('name','')[:50]}")
            print(f"  → {parsed[0]}")

        # 检查点保存
        batch_num = start_idx // BATCH_SIZE + 1
        if batch_num % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(enriched, batch_num, total_batches)

    print(f"\n✓ 完成! 成功: {len(recipes) - failed}, 失败: {failed}")
    return enriched


def load_recipes() -> list[dict]:
    """加载食谱数据。优先从 checkpoint 恢复."""
    cp_file = RECIPES_FILE.with_suffix(".checkpoint.json")
    if cp_file.exists():
        print(f"ℹ 发现检查点文件, 从中恢复...")
        recipes = json.load(open(cp_file, "r"))
        print(f"  检查点: {len(recipes)} 条")
        # 合并主文件中可能有而检查点没有的
        main_recipes = json.load(open(RECIPES_FILE, "r"))
        if len(main_recipes) > len(recipes):
            existing_ids = {r["recipe_id"] for r in recipes}
            new_recipes = [r for r in main_recipes if r["recipe_id"] not in existing_ids]
            recipes.extend(new_recipes)
            print(f"  合并主文件补充 {len(new_recipes)} 条, 共计 {len(recipes)} 条")
    else:
        recipes = json.load(open(RECIPES_FILE, "r"))
        print(f"已加载 {len(recipes)} 条食谱")
    return recipes


def save_checkpoint(recipes: list[dict], batch_num: int, total: int):
    """保存检查点"""
    cp_file = RECIPES_FILE.with_suffix(".checkpoint.json")
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    success = sum(1 for r in recipes if not _is_empty_recipe(r))
    print(f"\n  ⏺ 检查点已保存 ({batch_num}/{total}) | 成功: {success}/{len(recipes)}")


if __name__ == "__main__":
    print("=" * 60)
    print("   Recipe Assistant — LLM 字段补充")
    print("=" * 60)

    recipes = load_recipes()
    total = len(recipes)

    # 筛选需要补充的
    to_enrich = [r for r in recipes if _is_empty_recipe(r)]
    already_done = total - len(to_enrich)
    print(f"  已完成: {already_done}, 待补充: {len(to_enrich)}")

    if not to_enrich:
        print("✓ 全部已完成!")
        sys.exit(0)

    enriched = enrich_recipes(to_enrich)

    # 合并回完整数据
    enriched_ids = {r["recipe_id"] for r in enriched}
    for i, r in enumerate(recipes):
        if r["recipe_id"] in enriched_ids:
            # 找到 enriched 中的对应数据
            er = [e for e in enriched if e["recipe_id"] == r["recipe_id"]]
            if er:
                recipes[i] = er[0]

    # 保存
    with open(RECIPES_FILE, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)

    final_success = sum(1 for r in recipes if not _is_empty_recipe(r))
    print(f"\n结果已保存到 {RECIPES_FILE}")
    print(f"总计: {total}, 成功补充: {final_success - already_done}, 累计成功: {final_success}")

    # 清理检查点
    cp_file = RECIPES_FILE.with_suffix(".checkpoint.json")
    if cp_file.exists() and final_success == total:
        cp_file.unlink()
        print("ℹ 检查点文件已清理")
