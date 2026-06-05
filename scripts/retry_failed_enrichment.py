"""
重试失败的 LLM 补充 — 对 nutrition 全空的食谱用小 batch + 多轮采样重试。

策略:
  - BATCH_SIZE=2 (降低输出复杂度)
  - temperature=0.5 (适度随机性)
  - 每批最多重试 3 次 (不同 seed), 取第一个有效结果
  - 只处理 nutrition.calories == 0 的食谱
  - 完成后合并回 recipes_real.json

用法:
  HF_ENDPOINT=https://hf-mirror.com python scripts/retry_failed_enrichment.py
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
BATCH_SIZE = 2                     # 小 batch
MAX_NEW_TOKENS = 512               # 单条输出不需要太长
TEMPERATURE = 0.5                  # 适度随机
TOP_P = 0.95
MAX_RETRIES = 3                    # 每批最多重试次数
CHECKPOINT_INTERVAL = 50           # 每 50 批保存一次
DEVICE = "cuda"

SYSTEM_PROMPT = """Return JSON array only. Each element: {"difficulty":"easy|medium|hard","prep_time":int,"cook_time":int,"servings":int,"calories":int,"protein_g":int,"fat_g":int,"carbs_g":int}

Example:
[{"difficulty":"easy","prep_time":15,"cook_time":30,"servings":4,"calories":350,"protein_g":20,"fat_g":12,"carbs_g":45}]"""


def format_recipe(recipe: dict) -> str:
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
    parts = []
    for i, recipe in enumerate(batch):
        parts.append(f"[Recipe {i+1}]\n{format_recipe(recipe)}")
    return "\n\n".join(parts)


def _extract_int(value, default=0) -> int:
    if isinstance(value, (int, float)):
        return round(value)
    if not isinstance(value, str):
        return default
    nums = re.findall(r"\d+(?:\.\d+)?", value.replace(",", ""))
    return round(float(nums[0])) if nums else default


def _is_empty_recipe(recipe: dict) -> bool:
    return recipe.get("nutrition", {}).get("calories", 0) == 0


def parse_llm_output(text: str, n_expected: int) -> list[dict | None]:
    """解析 LLM 输出 JSON, 4 层容错"""
    # 1. 提取 ```json ... ``` 代码块
    json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if json_match:
        text = json_match.group(1)

    text = text.strip()
    # 2. 直接解析
    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results[:n_expected]
    except json.JSONDecodeError:
        pass

    # 3. 找 [] 范围
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            results = json.loads(text[start:end + 1])
            if isinstance(results, list):
                return results[:n_expected]
        except json.JSONDecodeError:
            pass

    # 4. 逐对象匹配
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


def validate_parsed(data: dict) -> bool:
    """检查解析结果是否包含必要字段"""
    required_keys = ["difficulty", "prep_time", "cook_time", "servings",
                     "calories", "protein_g", "fat_g", "carbs_g"]
    for k in required_keys:
        if k not in data:
            return False
    # 确保 difficulty 是有效值
    if data.get("difficulty", "").lower() not in ("easy", "medium", "hard"):
        return False
    # 确保有实质性的营养值 (不能全零)
    if all(data.get(k, 0) == 0 for k in ["calories", "protein_g", "fat_g", "carbs_g"]):
        return False
    return True


def generate_with_retry(model, tokenizer, batch: list[dict], max_retries: int = MAX_RETRIES) -> list[dict | None]:
    """对一批数据进行多次尝试, 取第一个有效结果"""
    import torch

    user_prompt = build_batch_prompt(batch)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries):
        seed = attempt * 42
        torch.manual_seed(seed)

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_llm_output(response, len(batch))

        # 检查是否每个位置都有有效结果
        valid = True
        for p in parsed:
            if p is None or not validate_parsed(p):
                valid = False
                break

        if valid:
            return parsed

    # 所有尝试失败后, 尝试用 greedy decoding 再跑一次
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # greedy
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    parsed = parse_llm_output(response, len(batch))

    # 对每个位置: 如果有效则保留, 否则 None
    result = []
    for p in parsed:
        if p is not None and validate_parsed(p):
            result.append(p)
        else:
            result.append(None)

    # 填充缺失的位置为 None
    while len(result) < len(batch):
        result.append(None)
    return result


def enrich_one_recipe(recipe: dict, data: dict):
    """用解析结果填充食谱字段"""
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


def retry_failed(recipes: list[dict]) -> tuple[list[dict], int]:
    """对失败条目进行重试补充"""
    # 筛选出需要重试的
    to_retry = [r for r in recipes if _is_empty_recipe(r)]
    already_ok = [r for r in recipes if not _is_empty_recipe(r)]

    print(f"已有: {len(already_ok)} 条, 待重试: {len(to_retry)} 条")

    if not to_retry:
        print("没有需要重试的条目!")
        return recipes, 0

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

    new_success = 0
    total_batches = (len(to_retry) + BATCH_SIZE - 1) // BATCH_SIZE

    for start_idx in tqdm(range(0, len(to_retry), BATCH_SIZE), desc="重试 LLM 推断"):
        batch = to_retry[start_idx:start_idx + BATCH_SIZE]
        parsed = generate_with_retry(model, tokenizer, batch)

        for i, recipe in enumerate(batch):
            if i < len(parsed) and parsed[i] is not None and validate_parsed(parsed[i]):
                enrich_one_recipe(recipe, parsed[i])
                new_success += 1

        # 打印第一条成功示例
        if start_idx == 0 and parsed[0] is not None and validate_parsed(parsed[0]):
            print(f"\n[sample] {batch[0].get('name','')[:50]}")
            print(f"  → {parsed[0]}")

        # 检查点保存
        batch_num = start_idx // BATCH_SIZE + 1
        if batch_num % CHECKPOINT_INTERVAL == 0:
            # 合并已成功 + 已处理(含部分成功) + 未处理
            processed_count = start_idx + BATCH_SIZE
            remaining = to_retry[processed_count:]
            current_all = already_ok + to_retry[:processed_count] + remaining
            save_retry_checkpoint(current_all, batch_num, total_batches, new_success)

    # 合并结果
    final_recipes = already_ok + to_retry
    print(f"\n✓ 重试完成! 本轮新增成功: {new_success}/{len(to_retry)}")
    print(f"  总计成功: {len(already_ok) + new_success}/{len(recipes)}")
    return final_recipes, new_success


def save_retry_checkpoint(recipes: list[dict], batch_num: int, total: int, new_success: int):
    """保存重试检查点"""
    cp_file = RECIPES_FILE.with_suffix(".retry_checkpoint.json")
    with open(cp_file, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    success = sum(1 for r in recipes if not _is_empty_recipe(r))
    print(f"\n  ⏺ 重试检查点已保存 ({batch_num}/{total}) | 本轮新增: {new_success} | 总成功: {success}/{len(recipes)}")


def merge_checkpoint():
    """如果存在重试检查点, 合并回主文件"""
    retry_cp = RECIPES_FILE.with_suffix(".retry_checkpoint.json")
    if not retry_cp.exists():
        return False

    print(f"\nℹ 发现重试检查点, 合并回主文件...")
    retry_data = json.load(open(retry_cp, "r"))
    with open(RECIPES_FILE, "w", encoding="utf-8") as f:
        json.dump(retry_data, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in retry_data if not _is_empty_recipe(r))
    print(f"  合并完成: {success}/{len(retry_data)} 条已补充")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("   Recipe Assistant — LLM 字段补充 (重试失败条目)")
    print("   策略: batch=2, temp=0.5, 最多重试 3 次 + greedy fallback")
    print("=" * 60)

    # 先尝试合并已有的检查点
    merge_checkpoint()

    # 加载当前数据
    recipes = json.load(open(RECIPES_FILE, "r"))
    print(f"已加载 {len(recipes)} 条食谱")

    final_recipes, new_success = retry_failed(recipes)

    # 保存最终结果
    with open(RECIPES_FILE, "w", encoding="utf-8") as f:
        json.dump(final_recipes, f, ensure_ascii=False, indent=2)

    total_success = sum(1 for r in final_recipes if not _is_empty_recipe(r))
    print(f"\n{'='*60}")
    print(f"最终结果: {total_success}/{len(final_recipes)} 条已补充 ({100*total_success/len(final_recipes):.1f}%)")
    print(f"{'='*60}")
