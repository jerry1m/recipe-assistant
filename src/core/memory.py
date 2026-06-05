"""
对话记忆管理 — Redis 持久化 + Token 感知滑动窗口 + LLM 压缩

设计:
  1. Redis 存储: 每个 session 一个 Hash + 一个 List(Turns)
     - recipe:session:{id}        → Hash(session_id, created_at, last_active)
     - recipe:session:{id}:turns  → List(TurnRecord JSON)
     - recipe:session:{id}:summary → 已压缩的旧轮次摘要（方案 A: LLM 压缩）
     - recipe:session:{id}:meta   → 压缩元数据（已压缩轮次数量）
  2. Redis TTL 自动管理过期会话，无需手动清理
  3. Token 感知滑动窗口: get_context(max_tokens) 从最新轮次往回计算，
     累积到 max_tokens 为止，避免撑爆 LLM context window
  4. 方案 A — LLM 压缩: 当旧轮次被滑出窗口时，调用 LLM 将其压缩为摘要，
     替代原始轮次保存在上下文中，保留关键信息的同时大幅节省 token
  5. Redis 不可用时自动兜底到进程内内存（开发/单机部署友好）
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.api.schemas import IntentType, ProvenanceItem, TurnRecord


def _estimate_tokens(text: str) -> int:
    """估算 token 数：中文约 1.5 chars/token，英文/混合约 2.5 chars/token"""
    if not text:
        return 0
    # 粗略统一按 2 chars/token 估算
    return max(1, len(text) // 2)


class ConversationMemory:
    """
    按 session_id 存储对话历史（Redis 持久化）。
    使用 Token 感知的滑动窗口优化 context 长度，节省 token。
    """

    _TURN_TIMEOUT = 86400               # 会话 TTL（秒），Redis 自动过期
    _MAX_TURNS = 50                     # 最多保留轮数
    _DEFAULT_MAX_CONTEXT_TOKENS = 2048  # 滑动窗口默认最大 token 数

    _redis = None                        # 延迟初始化的 Redis 连接
    _redis_available = True              # Redis 是否可用
    _fallback_store: dict[str, dict] = {}  # Redis 不可用时的内存兜底

    # ── LLM 压缩统计（类级别，全局累积）──
    _compression_metrics: dict[str, float] = {
        "total_original_tokens": 0.0,
        "total_compressed_tokens": 0.0,
        "total_turns_compressed": 0,
        "total_compression_calls": 0,
    }

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._key = f"recipe:session:{session_id}"
        self._turns_key = f"recipe:session:{session_id}:turns"
        self._summary_key = f"recipe:session:{session_id}:summary"
        self._meta_key = f"recipe:session:{session_id}:meta"

    # ── Redis 连接（延迟初始化，全局共享）──

    @classmethod
    def _get_redis(cls):
        """获取 Redis 连接（单例），不可用返回 None"""
        if not cls._redis_available:
            return None
        if cls._redis is None:
            try:
                import redis.asyncio as aioredis
                from src.core.config import get_settings
                settings = get_settings()
                cls._redis = aioredis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    password=settings.redis_password or None,
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Redis unavailable, falling back to in-memory storage", exc_info=e
                )
                cls._redis_available = False
                cls._redis = None
        return cls._redis

    # ── 工厂方法 ──

    @classmethod
    async def get_or_create(cls, session_id: str) -> "ConversationMemory":
        """获取或创建会话。Redis TTL 自动管理过期清理，无需手动删除。"""
        inst = cls(session_id)
        r = cls._get_redis()
        if r is not None:
            try:
                exists = await r.exists(inst._key)
                if not exists:
                    await r.hset(inst._key, mapping={
                        "session_id": session_id,
                        "created_at": str(time.time()),
                        "last_active": str(time.time()),
                    })
                await r.expire(inst._key, cls._TURN_TIMEOUT)
                await r.expire(inst._turns_key, cls._TURN_TIMEOUT)
                await r.expire(inst._summary_key, cls._TURN_TIMEOUT)
                await r.expire(inst._meta_key, cls._TURN_TIMEOUT)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Redis command failed, falling back to in-memory", exc_info=e
                )
                cls._redis_available = False
                cls._redis = None
                r = None
        if r is None:
            # Redis 不可用 → 内存兜底
            if session_id not in cls._fallback_store:
                cls._fallback_store[session_id] = {
                    "turns": [],
                    "last_active": time.time(),
                    "created_at": time.time(),
                    "summary": "",
                    "compressed_count": 0,
                }
        return inst

    # ── Redis 辅助 ──

    async def _touch(self):
        """刷新 session TTL（静默忽略 Redis 错误）"""
        r = self._get_redis()
        if r is not None:
            try:
                await r.expire(self._key, self._TURN_TIMEOUT)
                await r.expire(self._turns_key, self._TURN_TIMEOUT)
                await r.expire(self._summary_key, self._TURN_TIMEOUT)
                await r.expire(self._meta_key, self._TURN_TIMEOUT)
            except Exception:
                pass

    async def _load_turns(self) -> list[TurnRecord]:
        """从 Redis（或内存兜底）加载所有轮次"""
        r = self._get_redis()
        if r is not None:
            try:
                raw = await r.lrange(self._turns_key, 0, -1)
                return [TurnRecord.model_validate_json(item) for item in raw]
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Redis lrange failed, falling back to in-memory", exc_info=e
                )
        fb = ConversationMemory._fallback_store.get(self.session_id, {})
        return fb.get("turns", [])

    # ── 压缩摘要辅助 ──

    async def _load_summary(self) -> str:
        """从 Redis（或内存兜底）加载压缩摘要"""
        r = self._get_redis()
        if r is not None:
            try:
                val = await r.get(self._summary_key)
                return val or ""
            except Exception:
                pass
        fb = ConversationMemory._fallback_store.get(self.session_id, {})
        return fb.get("summary", "")

    async def _store_summary(self, summary: str) -> None:
        """存储压缩摘要到 Redis（或内存兜底）"""
        r = self._get_redis()
        if r is not None:
            try:
                await r.set(self._summary_key, summary)
                await r.expire(self._summary_key, self._TURN_TIMEOUT)
                return
            except Exception:
                pass
        fb = ConversationMemory._fallback_store.setdefault(self.session_id, {
            "turns": [], "last_active": time.time(), "created_at": time.time(),
            "summary": "", "compressed_count": 0,
        })
        fb["summary"] = summary

    async def _load_compressed_count(self) -> int:
        """已压缩的轮次数"""
        r = self._get_redis()
        if r is not None:
            try:
                val = await r.hget(self._meta_key, "compressed_count")
                return int(val) if val else 0
            except Exception:
                return 0
        fb = ConversationMemory._fallback_store.get(self.session_id, {})
        return fb.get("compressed_count", 0)

    async def _store_compressed_count(self, count: int) -> None:
        """存储已压缩轮次数"""
        r = self._get_redis()
        if r is not None:
            try:
                await r.hset(self._meta_key, "compressed_count", str(count))
                await r.expire(self._meta_key, self._TURN_TIMEOUT)
                return
            except Exception:
                pass
        fb = ConversationMemory._fallback_store.setdefault(self.session_id, {
            "turns": [], "last_active": time.time(), "created_at": time.time(),
            "summary": "", "compressed_count": 0,
        })
        fb["compressed_count"] = count

    # ── 核心方法 ──

    async def add_turn(
        self,
        query: str,
        intent: str,
        slots: dict[str, Any],
        answer: str,
        provenance: list[ProvenanceItem],
    ) -> None:
        """记录一轮对话（追加到 Redis List，自动裁剪超出的轮次）"""
        turn = TurnRecord(
            query=query, intent=intent, slots=slots,
            answer=answer, provenance=provenance,
            timestamp=time.time(),
        )
        r = self._get_redis()
        if r is not None:
            try:
                await r.rpush(self._turns_key, turn.model_dump_json())
                count = await r.llen(self._turns_key)
                if count > self._MAX_TURNS:
                    await r.lpop(self._turns_key, count - self._MAX_TURNS)
                await r.hset(self._key, "last_active", str(time.time()))
                await self._touch()
                return
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Redis rpush failed, falling back to in-memory", exc_info=e
                )
        # Redis 不可用或失败 → 内存兜底
        fb = ConversationMemory._fallback_store.setdefault(self.session_id, {
            "turns": [], "last_active": time.time(), "created_at": time.time(),
        })
        fb["turns"].append(turn)
        fb["last_active"] = time.time()
        if len(fb["turns"]) > self._MAX_TURNS:
            fb["turns"] = fb["turns"][-self._MAX_TURNS:]

    async def get_context(self, max_tokens: int = 0) -> str:
        """
        Token 感知的滑动窗口上下文（支持 LLM 压缩摘要）。

        流程：
          1. 加载已有压缩摘要 + 所有轮次
          2. 摘要 token 计入预算，从最新轮次往前保留尽可能多的轮次
          3. 超出预算的旧轮次不直接丢弃 —— 它们已被 / 将被 LLM 压缩为摘要

        默认使用 _DEFAULT_MAX_CONTEXT_TOKENS（2048）。
        """
        if max_tokens <= 0:
            max_tokens = self._DEFAULT_MAX_CONTEXT_TOKENS

        turns = await self._load_turns()
        if not turns:
            return ""

        summary = await self._load_summary()

        lines: list[str] = ["===== 对话历史 ====="]
        token_count = _estimate_tokens("===== 对话历史 =====")

        # 先插入压缩摘要（如果存在）
        if summary:
            lines.append(f"[对话摘要] {summary}")
            token_count += _estimate_tokens(summary)

        # 从最新轮次往前遍历，在预算内保留尽可能多的轮次
        kept_count = 0
        for t in reversed(turns):
            turn_text = f"用户: {t.query}\n意图: {t.intent}\n助手: {t.answer}"
            turn_tokens = _estimate_tokens(turn_text)
            # 至少保留一轮对话
            if token_count + turn_tokens > max_tokens and kept_count > 0:
                break
            turn_idx = turns.index(t) + 1
            # 插入到摘要行之后（保持时间正序）
            insert_pos = 2 if summary else 1
            lines.insert(insert_pos, f"[{turn_idx}] {turn_text}")
            token_count += turn_tokens
            kept_count += 1

        return "\n".join(lines)

    # ── LLM 压缩（方案 A）──

    async def _compress_with_llm(self, turns_text: str) -> str:
        """调用 LLM 将多轮对话文本压缩为 1-3 句简洁摘要。"""
        try:
            from src.core.utils.llm import generate

            system_prompt = (
                "你是一个高效的对话摘要助手。"
                "请将以下用户与食谱助手的多轮对话压缩为1-3句话的简洁中文摘要。"
                "保留关键的查询意图、菜名、食材和营养偏好信息。"
                "只输出摘要内容，不要添加任何解释或前缀。"
            )
            result = await generate(
                query=f"请压缩以下对话：\n{turns_text}",
                system_prompt=system_prompt,
                max_new_tokens=256,
                temperature=0.1,
            )
            return result.strip() if result else ""
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "LLM compression failed", exc_info=e
            )
            return ""

    async def compress_if_needed(self, max_tokens: int = 0) -> dict:
        """
        检查并压缩超出预算的旧轮次（方案 A）。

        使用 LLM 将最早的不再适合窗口的轮次压缩为摘要，
        追加到已有摘要中。此方法设计为在保存新轮次后调用，
        通常以 background task 方式运行，不阻塞主流程。

        返回压缩统计: {turns_compressed, original_tokens, compressed_tokens,
                        compression_ratio, summary_preview}
        """
        if max_tokens <= 0:
            max_tokens = self._DEFAULT_MAX_CONTEXT_TOKENS

        turns = await self._load_turns()
        if len(turns) < 3:
            return {"turns_compressed": 0, "reason": "对话轮次不足（< 3）"}

        summary = await self._load_summary()
        compressed_count = await self._load_compressed_count()

        # 已压缩到最新轮次，无需再压缩
        if compressed_count >= len(turns):
            return {"turns_compressed": 0, "reason": "已压缩至最新"}

        # 计算摘要占用的 token
        header_tokens = _estimate_tokens("===== 对话历史 =====")
        summary_tokens = _estimate_tokens(summary) if summary else 0
        used_tokens = header_tokens + summary_tokens

        # 从最新往旧遍历，看有多少轮次能留在预算内
        n_fit = 0
        for t in reversed(turns):
            turn_text = f"用户: {t.query}\n意图: {t.intent}\n助手: {t.answer}"
            t_tokens = _estimate_tokens(turn_text)
            if used_tokens + t_tokens > max_tokens and n_fit > 0:
                break
            used_tokens += t_tokens
            n_fit += 1

        # 需要压缩的轮次 = 未压缩的轮次中，最旧且装不下的部分
        already_fit_uncompressed = max(0, len(turns) - compressed_count - n_fit)
        if already_fit_uncompressed <= 0:
            return {"turns_compressed": 0, "reason": "预算充足无需压缩"}

        # 取最旧的那批轮次进行压缩
        turns_to_compress = turns[compressed_count:compressed_count + already_fit_uncompressed]
        if len(turns_to_compress) < 2:
            return {"turns_compressed": 0, "reason": "单轮无需压缩"}

        # 构建 LLM 输入
        turns_text = "\n---\n".join(
            f"用户: {t.query}\n助手: {t.answer}" for t in turns_to_compress
        )
        original_tokens = _estimate_tokens(turns_text)

        # 调用 LLM 压缩
        new_summary = await self._compress_with_llm(turns_text)
        if not new_summary:
            return {"turns_compressed": 0, "reason": "LLM 压缩返回空"}

        compressed_tokens = _estimate_tokens(new_summary)

        # 合并到已有摘要
        if summary:
            combined = f"{summary} | {new_summary}"
        else:
            combined = new_summary

        # 持久化
        await self._store_summary(combined)
        new_count = compressed_count + len(turns_to_compress)
        await self._store_compressed_count(new_count)

        # 更新全局统计
        self._compression_metrics["total_original_tokens"] += original_tokens
        self._compression_metrics["total_compressed_tokens"] += compressed_tokens
        self._compression_metrics["total_turns_compressed"] += len(turns_to_compress)
        self._compression_metrics["total_compression_calls"] += 1

        ratio = compressed_tokens / max(1, original_tokens)
        return {
            "turns_compressed": len(turns_to_compress),
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": round(ratio, 4),
            "summary_preview": new_summary[:80],
        }

    @classmethod
    def get_compression_stats(cls) -> dict:
        """获取全局压缩统计"""
        m = cls._compression_metrics
        total_orig = m["total_original_tokens"]
        total_comp = m["total_compressed_tokens"]
        return {
            "total_compression_calls": m["total_compression_calls"],
            "total_turns_compressed": m["total_turns_compressed"],
            "total_original_tokens": int(total_orig),
            "total_compressed_tokens": int(total_comp),
            "overall_compression_ratio": round(total_comp / max(1, total_orig), 4)
                if m["total_compression_calls"] > 0 else 0.0,
            "tokens_saved": int(total_orig - total_comp),
        }

    async def get_last_turn(self) -> TurnRecord | None:
        """获取上一轮对话记录"""
        turns = await self._load_turns()
        return turns[-1] if turns else None

    async def get_last_intent(self) -> str | None:
        """获取上一轮意图"""
        last = await self.get_last_turn()
        return last.intent if last else None

    async def get_last_slots(self) -> dict[str, Any]:
        """获取上一轮提取的槽位"""
        last = await self.get_last_turn()
        return last.slots if last else {}

    async def get_entity_summary(self) -> str:
        """从对话历史中提取实体摘要（菜名、食材等）。"""
        turns = await self._load_turns()
        mentioned: dict[str, set[str]] = {"recipes": set(), "ingredients": set()}
        for t in turns:
            slots = t.slots
            if slots.get("recipe_name"):
                mentioned["recipes"].add(slots["recipe_name"])
            if slots.get("ingredient"):
                mentioned["ingredients"].add(slots["ingredient"])
            if slots.get("food"):
                mentioned["recipes"].add(slots["food"])

        parts = []
        if mentioned["recipes"]:
            parts.append(f"提及的菜: {', '.join(mentioned['recipes'])}")
        if mentioned["ingredients"]:
            parts.append(f"提及的食材: {', '.join(mentioned['ingredients'])}")
        return " | ".join(parts) if parts else ""

    async def clear(self) -> None:
        """清空当前会话的 Redis 数据（含压缩摘要）"""
        r = self._get_redis()
        if r is not None:
            await r.delete(self._key, self._turns_key, self._summary_key, self._meta_key)
        else:
            self._fallback_store.pop(self.session_id, None)

    @classmethod
    async def clear_all(cls) -> None:
        """清空所有会话（仅限内存兜底数据）"""
        cls._fallback_store.clear()
