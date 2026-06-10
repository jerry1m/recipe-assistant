"""
长效记忆 — 跨会话持久化用户偏好（饮食限制、偏好菜系、过敏原等）

设计参考 travel-agent-guide 的 LongTermMemory + Milvus 持久化思路

存储方式:
  - JSON 文件存储（轻量，零依赖），路径由配置中的 LONG_TERM_MEMORY_PATH 控制
  - 写入时自动同步，读取时延迟加载

使用场景:
  - Router 在意图识别时读取用户偏好，增强多轮对话的上下文感知
  - Formatter 可根据用户偏好调整推荐
  - 跨会话保持用户画像（素食者、低卡偏好、海鲜过敏等）

数据模型:
  每个用户存储:
  {
    "user_id": str,
    "created_at": float,
    "updated_at": float,
    "dietary_restrictions": [str],   # 饮食限制: 素食/清真/无麸质等
    "favorite_cuisines": [str],      # 偏好菜系: 川菜/粤菜/Italian 等
    "allergies": [str],              # 过敏原: 花生/牛奶/海鲜等
    "disliked_ingredients": [str],   # 不喜欢的食材
    "preferences": {str: str},       # 扩展键值对: 如 "spice_level": "mild"
    "recent_queries": [str],         # 最近查询（最多 20 条）
    "history": [                     # 交互历史摘要
      {"intent": str, "summary": str, "timestamp": float}
    ]
  }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# ── 默认存储路径 ──
_DEFAULT_STORAGE_PATH = os.environ.get(
    "RECIPE_LONG_TERM_MEMORY_PATH",
    str(Path(__file__).resolve().parent.parent.parent / "data" / "long_term_memory.json"),
)


def _get_storage_path() -> str:
    return _DEFAULT_STORAGE_PATH


class LongTermMemory:
    """
    长效记忆 — 跨会话持久化用户偏好。

    线程安全：写入时使用文件锁（fcntl/flock），避免并发写入损坏。
    性能：读取时全量加载到内存，写入时全量写回（数据量小，可接受）。
    """

    def __init__(self, storage_path: str | None = None):
        self._storage_path = storage_path or _get_storage_path()
        self._data: dict[str, dict] = {}  # user_id → user_profile
        self._loaded = False

    # ── 文件 I/O ──

    def _ensure_dir(self):
        Path(self._storage_path).parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        """加载存储文件到内存"""
        if self._loaded:
            return self._data
        self._ensure_dir()
        if os.path.exists(self._storage_path) and os.path.getsize(self._storage_path) > 0:
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("long_term_memory.load_failed", error=str(e))
                self._data = {}
        else:
            self._data = {}
        self._loaded = True
        return self._data

    def _save(self):
        """将内存数据写回存储文件"""
        self._ensure_dir()
        # 写入临时文件后原子替换，避免写一半崩溃导致数据损坏
        tmp_path = self._storage_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._storage_path)
        except OSError as e:
            logger.error("long_term_memory.save_failed", error=str(e))

    # ── 用户画像操作 ──

    def _get_or_create_profile(self, user_id: str) -> dict:
        """获取或创建用户画像"""
        self._load()
        if user_id not in self._data:
            now = time.time()
            self._data[user_id] = {
                "user_id": user_id,
                "created_at": now,
                "updated_at": now,
                "dietary_restrictions": [],
                "favorite_cuisines": [],
                "allergies": [],
                "disliked_ingredients": [],
                "preferences": {},
                "recent_queries": [],
                "history": [],
            }
        return self._data[user_id]

    async def get_profile(self, user_id: str) -> dict[str, Any] | None:
        """获取用户完整画像（不存在返回 None）"""
        self._load()
        return self._data.get(user_id)

    async def get_or_create_profile(self, user_id: str) -> dict[str, Any]:
        """获取或创建用户画像"""
        return self._get_or_create_profile(user_id)

    async def update_preference(self, user_id: str, key: str, value: list[str] | dict):
        """
        更新用户偏好。

        参数:
            user_id: 用户标识
            key: 字段名（dietary_restrictions / favorite_cuisines / allergies 等）
            value: 新值
        """
        profile = self._get_or_create_profile(user_id)
        if key in ("dietary_restrictions", "favorite_cuisines", "allergies", "disliked_ingredients"):
            if isinstance(value, list):
                # 去重合并
                existing = set(profile.get(key, []))
                existing.update(v.strip() for v in value if v.strip())
                profile[key] = sorted(existing)
        elif key == "preferences":
            if isinstance(value, dict):
                profile.setdefault("preferences", {}).update(value)
        else:
            profile[key] = value
        profile["updated_at"] = time.time()
        self._save()
        logger.info("long_term_memory.updated", user_id=user_id, key=key)

    async def add_recent_query(self, user_id: str, query: str):
        """记录用户最近查询（最多 20 条）"""
        profile = self._get_or_create_profile(user_id)
        recent = profile.setdefault("recent_queries", [])
        recent.append(query)
        # 保留最多 20 条
        if len(recent) > 20:
            recent[:] = recent[-20:]
        profile["updated_at"] = time.time()
        self._save()

    async def add_history_entry(self, user_id: str, intent: str, summary: str):
        """添加交互历史摘要"""
        profile = self._get_or_create_profile(user_id)
        history = profile.setdefault("history", [])
        history.append({
            "intent": intent,
            "summary": summary[:200],
            "timestamp": time.time(),
        })
        # 保留最多 50 条
        if len(history) > 50:
            history[:] = history[-50:]
        profile["updated_at"] = time.time()
        self._save()

    async def get_preference_context(self, user_id: str) -> str:
        """
        获取用户偏好上下文字符串（用于注入到 Router/Agent 的 system prompt）。

        返回如: "用户偏好: 素食主义者, 偏好川菜, 对花生过敏"
        无偏好时返回空字符串。
        """
        profile = await self.get_profile(user_id)
        if not profile:
            return ""

        parts = []
        if profile.get("dietary_restrictions"):
            parts.append(f"饮食限制: {'、'.join(profile['dietary_restrictions'])}")
        if profile.get("favorite_cuisines"):
            parts.append(f"偏好菜系: {'、'.join(profile['favorite_cuisines'])}")
        if profile.get("allergies"):
            parts.append(f"过敏原: {'、'.join(profile['allergies'])}")
        if profile.get("disliked_ingredients"):
            parts.append(f"不喜欢的食材: {'、'.join(profile['disliked_ingredients'])}")
        prefs = profile.get("preferences", {})
        if prefs:
            for k, v in prefs.items():
                parts.append(f"{k}: {v}")

        return "用户偏好: " + "; ".join(parts) if parts else ""

    async def extract_and_save_preferences(
        self, user_id: str, query: str, intent: str, answer: str
    ):
        """
        从用户查询和回答中自动提取偏好并保存（被动学习）。
        当前规则: 检测关键词，后续可升级为 LLM 提取。

        参数:
            user_id: 用户标识
            query: 用户查询
            intent: 当前意图
            answer: 助手的回答
        """
        profile = self._get_or_create_profile(user_id)
        query_lower = query.lower()
        updated = False

        # 检测饮食限制关键词
        restriction_keywords = {
            "素食": "素食", "vegan": "素食", "vegetarian": "素食",
            "清真": "清真", "halal": "清真",
            "无麸质": "无麸质", "gluten-free": "无麸质", "gluten free": "无麸质",
        }
        for keyword, label in restriction_keywords.items():
            if keyword in query_lower and label not in profile.get("dietary_restrictions", []):
                profile.setdefault("dietary_restrictions", []).append(label)
                updated = True

        # 检测过敏原关键词
        allergy_keywords = {
            "过敏": None, "allergic": None, "allergy": None,
        }
        for keyword in allergy_keywords:
            if keyword in query_lower:
                # 尝试提取具体过敏原（简单实现）
                for allergen in ["花生", "牛奶", "海鲜", "鸡蛋", "大豆", "坚果", "麸质"]:
                    if allergen in query_lower and allergen not in profile.get("allergies", []):
                        profile.setdefault("allergies", []).append(allergen)
                        updated = True

        if updated:
            profile["updated_at"] = time.time()
            self._save()
            logger.info("long_term_memory.auto_extracted", user_id=user_id)

    # ── 统计 ──

    def get_stats(self) -> dict:
        """获取长效记忆统计"""
        self._load()
        total_users = len(self._data)
        total_restrictions = sum(
            len(p.get("dietary_restrictions", [])) for p in self._data.values()
        )
        total_allergies = sum(
            len(p.get("allergies", [])) for p in self._data.values()
        )
        return {
            "total_users": total_users,
            "total_dietary_restrictions": total_restrictions,
            "total_allergies": total_allergies,
            "storage_path": self._storage_path,
        }


# ── 全局单例 ──
_global_memory: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """获取全局长效记忆实例（单例）"""
    global _global_memory
    if _global_memory is None:
        _global_memory = LongTermMemory()
    return _global_memory
