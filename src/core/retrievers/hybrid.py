"""
混合检索 — BM25 + 向量 + Rerank 三级融合
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

# 优先使用 HF 镜像（如果没设 HF_ENDPOINT，默认用镜像站）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re

from src.api.schemas import Chunk
from src.core.config import get_settings
from src.core.retrievers.base import BaseRetriever

settings = get_settings()

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src" / "data"
VECTOR_DIR = DATA_DIR / "vector_store"

# ── 语言感知分词 ──

_HAS_JIEBA = False
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    pass

# 中文字符范围
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


def _tokenize(text: str) -> list[str]:
    """语言感知分词：中文用 jieba，英文用 whitespace split"""
    text = text.lower().strip()
    if not text:
        return []
    if _CJK_RE.search(text):
        if _HAS_JIEBA:
            return list(jieba.cut(text))
        # 无 jieba 时按单字切分作为兜底
        return list(text)
    return text.split()


class HybridRetriever(BaseRetriever):
    """BM25 + 向量语义 + Rerank 精排三级融合"""

    def __init__(self):
        self.bm25_weight = settings.retriever_bm25_weight
        self.vector_weight = settings.retriever_vector_weight
        self.rerank_weight = settings.retriever_rerank_weight

        # 懒加载缓存
        self._bm25 = None
        self._bm25_chunk_ids: list[str] | None = None  # BM25 文档顺序对应的 chunk_id 列表
        self._faiss_index = None
        self._chunk_ids: list[str] | None = None
        self._chunks_map: dict[str, dict] | None = None
        self._sentence_model = None
        self._rerank_model = None

    # ── 索引加载 ──

    def _load_bm25(self):
        if self._bm25 is not None:
            return
        bm25_path = VECTOR_DIR / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            # 支持两种格式：直接 BM25 对象，或 {'bm25': ..., 'chunk_ids': ...} 字典
            if isinstance(data, dict):
                self._bm25 = data["bm25"]
                self._bm25_chunk_ids = data.get("chunk_ids")
            else:
                self._bm25 = data
                self._bm25_chunk_ids = None
        else:
            self._bm25 = None
            self._bm25_chunk_ids = None

    def _load_faiss(self):
        if self._faiss_index is not None:
            return
        index_path = VECTOR_DIR / "recipes.index"
        ids_path = VECTOR_DIR / "chunk_ids.npy"
        if index_path.exists() and ids_path.exists():
            import faiss
            self._faiss_index = faiss.read_index(str(index_path))
            self._chunk_ids = np.load(ids_path, allow_pickle=True).tolist()
        else:
            self._faiss_index = None
            self._chunk_ids = []

    def _load_chunks_map(self):
        if self._chunks_map is not None:
            return
        chunks_path = DATA_DIR / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_list = json.load(f)
            self._chunks_map = {c["chunk_id"]: c for c in chunks_list}
        else:
            self._chunks_map = {}

    def _get_embedder(self):
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        return self._sentence_model

    # ── 核心检索 ──

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[Chunk]:
        """
        三级检索流水线：
        1. BM25 关键词初筛
        2. 向量语义召回
        3. Rerank 精排融合
        """
        # 并行执行 BM25 + 向量检索
        bm25_chunks = await self._bm25_search(query, top_k * 2)
        vector_chunks = await self._vector_search(query, top_k * 2)

        # 融合去重
        merged = self._fusion(bm25_chunks, vector_chunks)

        # Rerank 精排
        reranked = await self._rerank(query, merged)

        return reranked[:top_k]

    async def _bm25_search(self, query: str, top_k: int) -> list[Chunk]:
        """BM25 关键词检索"""
        self._load_bm25()
        if self._bm25 is None:
            return []

        self._load_chunks_map()

        # 中文检测 + 分词
        tokenized = _tokenize(query)
        scores = self._bm25.get_scores(tokenized)

        # 取 top_k
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 使用 BM25 pkl 中保存的 chunk_ids 映射（确保索引对齐）
        if self._bm25_chunk_ids is not None:
            id_list = self._bm25_chunk_ids
        else:
            # 降级：用 chunks_map 的插入序
            id_list = list(self._chunks_map.keys())

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            if idx >= len(id_list):
                continue
            cid = id_list[idx]
            c = self._chunks_map.get(cid)
            if c is None:
                continue
            results.append(Chunk(
                chunk_id=c["chunk_id"],
                recipe_id=c["recipe_id"],
                content=c["content"],
                section=c.get("section", ""),
                score=float(scores[idx]),
            ))
        return results

    async def _vector_search(self, query: str, top_k: int) -> list[Chunk]:
        """向量语义检索 (FAISS + sentence-transformers)"""
        self._load_faiss()
        if self._faiss_index is None:
            return []

        self._load_chunks_map()
        model = self._get_embedder()

        # 编码 query
        q_vec = model.encode([query], show_progress_bar=False)
        q_vec = q_vec.astype(np.float32)
        import faiss
        faiss.normalize_L2(q_vec)

        # 搜索
        distances, indices = self._faiss_index.search(q_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunk_ids):
                continue
            cid = self._chunk_ids[idx]
            c = self._chunks_map.get(cid)
            if c is None:
                continue
            # FAISS 内积归一化后近似余弦相似度 -> [0, 1]
            score = max(0.0, float(dist))
            results.append(Chunk(
                chunk_id=c["chunk_id"],
                recipe_id=c["recipe_id"],
                content=c["content"],
                section=c.get("section", ""),
                score=score,
            ))
        return results

    async def _rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Rerank 精排 — 使用 BGE-Reranker"""
        if not chunks:
            return chunks

        try:
            if self._rerank_model is None:
                from sentence_transformers import CrossEncoder
                self._rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")

            pairs = [(query, c.content) for c in chunks]
            scores = self._rerank_model.predict(pairs, show_progress_bar=False)

            for c, s in zip(chunks, scores):
                c.score = float(s)

            chunks.sort(key=lambda x: x.score, reverse=True)
        except Exception:
            # 降级：如果 rerank 失败，返回原序
            pass

        return chunks

    def _fusion(
        self,
        bm25_chunks: list[Chunk],
        vector_chunks: list[Chunk],
    ) -> list[Chunk]:
        """RRF (Reciprocal Rank Fusion) 融合 + 去重 — 不依赖分数量纲"""
        K = 60  # RRF 常数

        rank_score: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, c in enumerate(bm25_chunks):
            rank_score[c.chunk_id] = rank_score.get(c.chunk_id, 0) + 1.0 / (K + rank + 1)
            chunk_map[c.chunk_id] = c

        for rank, c in enumerate(vector_chunks):
            rank_score[c.chunk_id] = rank_score.get(c.chunk_id, 0) + 1.0 / (K + rank + 1)
            chunk_map[c.chunk_id] = c

        result = []
        for cid, chunk in chunk_map.items():
            chunk.score = rank_score[cid]
            result.append(chunk)

        result.sort(key=lambda x: x.score, reverse=True)
        return result
