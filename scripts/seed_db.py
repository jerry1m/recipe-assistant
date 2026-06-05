"""
初始化数据库 — 从 recipes_real.json 创建表并导入全部 5000 条菜谱
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "recipe.db"
DATA_PATH = BASE_DIR / "src" / "data" / "recipes_real.json"


def seed_database(db_path: str = str(DB_PATH), data_path: str = str(DATA_PATH)):
    """创建表并导入数据"""
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS recipes (
            recipe_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            cuisine TEXT,
            difficulty TEXT,
            prep_time TEXT,
            cook_time TEXT,
            servings INTEGER,
            tags TEXT,
            image_url TEXT,
            source TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id TEXT NOT NULL,
            name TEXT NOT NULL,
            amount TEXT,
            unit TEXT,
            alternative TEXT,
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
        );

        CREATE TABLE IF NOT EXISTS nutrition (
            recipe_id TEXT PRIMARY KEY,
            calories REAL,
            protein REAL,
            fat REAL,
            carbs REAL,
            fiber REAL,
            sodium REAL,
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
        );

        CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id TEXT NOT NULL,
            step_number INTEGER,
            content TEXT,
            FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id)
        );
    """)

    # 导入数据
    with open(data_path) as f:
        recipes = json.load(f)

    for recipe in recipes:
        rid = recipe["recipe_id"]

        # recipes 表
        cursor.execute(
            """INSERT OR REPLACE INTO recipes
               (recipe_id, name, cuisine, difficulty, prep_time, cook_time, servings, tags, image_url, source, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rid, recipe["name"], recipe.get("cuisine", ""),
                recipe.get("difficulty", ""), recipe.get("prep_time", ""),
                recipe.get("cook_time", ""), recipe.get("servings", 0),
                json.dumps(recipe.get("tags", []), ensure_ascii=False),
                recipe.get("image_url", ""), recipe.get("source", ""),
                recipe.get("description", ""),
            ),
        )

        # nutrition 表
        n = recipe.get("nutrition", {})
        cursor.execute(
            """INSERT OR REPLACE INTO nutrition
               (recipe_id, calories, protein, fat, carbs, fiber, sodium)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (rid, n.get("calories", 0), n.get("protein", 0),
             n.get("fat", 0), n.get("carbs", 0),
             n.get("fiber", 0), n.get("sodium", 0)),
        )

        # ingredients 表
        for ing in recipe.get("ingredients", []):
            cursor.execute(
                """INSERT INTO ingredients (recipe_id, name, amount, unit, alternative)
                   VALUES (?, ?, ?, ?, ?)""",
                (rid, ing["name"], ing.get("amount", ""), ing.get("unit", ""),
                 json.dumps(ing.get("alternative", []), ensure_ascii=False)),
            )

        # steps 表
        for i, step in enumerate(recipe.get("steps", []), start=1):
            cursor.execute(
                "INSERT INTO steps (recipe_id, step_number, content) VALUES (?, ?, ?)",
                (rid, i, step),
            )

    conn.commit()
    conn.close()
    elapsed = time.time() - t0
    print(f"✅ 已导入 {len(recipes)} 个菜谱到 {db_path} (耗时 {elapsed:.1f}s)")


if __name__ == "__main__":
    seed_database()
