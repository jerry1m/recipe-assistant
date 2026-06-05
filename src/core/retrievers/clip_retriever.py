"""
CLIP 图文跨模态检索 — 以图搜菜名

流程：
  用户上传图片 → CLIP 图片编码 → FAISS 检索菜名 → Top-K 匹配菜谱

依赖:
  pip install transformers Pillow faiss-cpu
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

# 优先使用 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from PIL import Image

from src.api.schemas import Chunk
from src.core.config import get_settings
from src.core.retrievers.base import BaseRetriever

settings = get_settings()

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src" / "data"
VECTOR_DIR = DATA_DIR / "vector_store"

CLIP_INDEX_PATH = VECTOR_DIR / "clip_recipe_names.index"
CLIP_NAMES_PATH = VECTOR_DIR / "clip_recipe_names.pkl"
CLIP_IDS_PATH = VECTOR_DIR / "clip_recipe_ids.npy"


class CLIPRetriever(BaseRetriever):
    """使用 CLIP 实现「图片 → 菜名」零样本检索

    将 5000 条菜谱名称用 CLIP 文本编码器预编码为 FAISS 索引。
    用户上传图片 → CLIP 图片编码 → FAISS 搜索 → 返回 Top-K 菜名。
    """

    _shared_model = None           # 全局单例 CLIP 模型
    _shared_processor = None       # 全局单例 processor
    _shared_index = None           # 全局单例 FAISS 索引
    _shared_recipe_names = None    # 全局单例 菜名列表
    _shared_recipe_ids = None      # 全局单例 recipe_id 列表
    _shared_loaded = False         # 是否已加载
    _shared_device = None          # 实际使用的 device

    @staticmethod
    def _pick_device() -> str:
        """自动选择最佳设备：需要 ≥3GB 空闲显存，否则 CPU

        CLIP-ViT-B/32 模型 605MB，推理 peak ~1.5GB，留 2 倍余量。
        """
        import subprocess
        if not torch.cuda.is_available():
            print("[CLIPRetriever] 无 GPU，回退 CPU")
            return "cpu"
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split(", ")
                if len(parts) == 2:
                    idx, free_mib = int(parts[0]), int(parts[1])
                    if free_mib >= 3000:  # 至少 3GB 空闲
                        print(f"[CLIPRetriever] 选用 GPU {idx} (可用 {free_mib} MiB)")
                        return f"cuda:{idx}"
        except Exception:
            pass
        print("[CLIPRetriever] 无足够显存的 GPU，回退 CPU")
        return "cpu"

    def __init__(self):
        self.model_name = settings.clip_model
        self._model = None
        self._processor = None
        self._index = None
        self._recipe_names: list[str] = []
        self._recipe_ids: list[str] = []

    # ── 模型懒加载（全局单例） ──

    def _ensure_loaded(self):
        """保证模型 + 索引已加载（懒加载 + 全局单例）"""
        if CLIPRetriever._shared_loaded:
            self._model = CLIPRetriever._shared_model
            self._processor = CLIPRetriever._shared_processor
            self._index = CLIPRetriever._shared_index
            self._recipe_names = CLIPRetriever._shared_recipe_names
            self._recipe_ids = CLIPRetriever._shared_recipe_ids
            return

        # 0. 自动选择可用 GPU
        if CLIPRetriever._shared_device is None:
            CLIPRetriever._shared_device = self._pick_device()

        device = CLIPRetriever._shared_device

        # 1. 加载 CLIP 模型 + processor
        if CLIPRetriever._shared_model is None:
            from transformers import CLIPModel, CLIPProcessor
            dtype = torch.float16 if "cuda" in device else torch.float32
            CLIPRetriever._shared_model = CLIPModel.from_pretrained(
                self.model_name, dtype=dtype,
            )
            CLIPRetriever._shared_model.eval()
            CLIPRetriever._shared_model = CLIPRetriever._shared_model.to(device)
            CLIPRetriever._shared_processor = CLIPProcessor.from_pretrained(self.model_name)
        elif CLIPRetriever._shared_processor is None:
            # crash recovery: 模型已加载但 processor 丢失
            from transformers import CLIPProcessor
            CLIPRetriever._shared_processor = CLIPProcessor.from_pretrained(self.model_name)

        # 同步到实例（_build_index 需要 processor）
        self._model = CLIPRetriever._shared_model
        self._processor = CLIPRetriever._shared_processor

        # 2. 加载或构建 FAISS 索引
        if CLIPRetriever._shared_index is None:
            import faiss

            if CLIP_INDEX_PATH.exists():
                # 索引已存在 → 直接加载
                self._index = faiss.read_index(str(CLIP_INDEX_PATH))
                with open(CLIP_NAMES_PATH, "rb") as f:
                    names, ids = pickle.load(f)
                CLIPRetriever._shared_recipe_names = names
                CLIPRetriever._shared_recipe_ids = ids
            else:
                # 索引不存在 → 构建
                self._build_index()

            CLIPRetriever._shared_index = self._index
            if CLIPRetriever._shared_recipe_names is None:
                CLIPRetriever._shared_recipe_names = self._recipe_names
                CLIPRetriever._shared_recipe_ids = self._recipe_ids

        # 3. 同步到实例
        self._model = CLIPRetriever._shared_model
        self._processor = CLIPRetriever._shared_processor
        self._index = CLIPRetriever._shared_index
        self._recipe_names = CLIPRetriever._shared_recipe_names
        self._recipe_ids = CLIPRetriever._shared_recipe_ids
        CLIPRetriever._shared_loaded = True

    # ── 构建 FAISS 索引 ──

    def _build_index(self):
        """读取 recipes_real.json，编码所有菜名，构建 FAISS 索引并持久化"""
        import faiss

        recipes_path = DATA_DIR / "recipes_real.json"
        if not recipes_path.exists():
            raise FileNotFoundError(
                f"{recipes_path} 不存在，请先运行 scripts/ingest_real_data.py"
            )

        with open(recipes_path, "r", encoding="utf-8") as f:
            recipes = json.load(f)

        names: list[str] = []
        ids: list[str] = []
        for r in recipes:
            rid = r.get("recipe_id", "")
            name = r.get("name", "").strip()
            if not name:
                name = rid.replace("_", " ").title()
            names.append(name)
            ids.append(rid)

        print(f"[CLIPRetriever] 编码 {len(names)} 条菜名 ...")
        # 加 prompt 前缀对齐 CLIP 训练分布
        texts = [f"A dish of {n}" for n in names]

        # 索引构建使用 CPU（避免和训练任务抢 GPU 显存）
        build_device = "cpu"
        model_cpu = CLIPRetriever._shared_model.to(build_device)
        batch_size = 16
        all_embs: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                emb = model_cpu.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())

        # 将模型移回原设备
        CLIPRetriever._shared_model = model_cpu.to(CLIPRetriever._shared_device)

        embeddings = np.vstack(all_embs).astype(np.float32)
        print(f"[CLIPRetriever] 嵌入矩阵形状: {embeddings.shape}")

        # IndexFlatIP: 内积 = 余弦相似度（已 L2 归一化）
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # 持久化
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(CLIP_INDEX_PATH))
        with open(CLIP_NAMES_PATH, "wb") as f:
            pickle.dump((names, ids), f)
        np.save(str(CLIP_IDS_PATH), ids)

        print(f"[CLIPRetriever] 索引已保存到 {CLIP_INDEX_PATH}")

        self._index = index
        self._recipe_names = names
        self._recipe_ids = ids

    # ── 公开接口 ──

    async def encode_image(self, image_base64: str) -> list[float]:
        """将 base64 图片转为归一化 embedding"""
        self._ensure_loaded()

        raw = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        device = next(self._model.parameters()).device
        inputs = self._processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            emb = self._model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten().tolist()

    async def encode_text(self, text: str) -> list[float]:
        """将文本转为归一化 embedding"""
        self._ensure_loaded()

        device = next(self._model.parameters()).device
        inputs = self._processor(text=text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            emb = self._model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().flatten().tolist()

    async def retrieve(self, query: str = "", top_k: int = 10, **kwargs: Any) -> list[Chunk]:
        """以图搜菜名

        Args:
            query: 文字描述（兜底，仅当无 image 时使用）
            top_k: 返回数量
            kwargs:
                image: base64 图片字符串

        Returns:
            list[Chunk]: 每个 Chunk 对应一个匹配菜谱, score 为余弦相似度
        """
        self._ensure_loaded()

        image_b64 = kwargs.get("image", "")
        if not image_b64:
            vec = await self.encode_text(query)
        else:
            vec = await self.encode_image(image_b64)

        query_vec = np.array([vec], dtype=np.float32)
        scores, indices = self._index.search(query_vec, top_k)

        chunks: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rid = self._recipe_ids[idx]
            name = self._recipe_names[idx]
            chunks.append(Chunk(
                chunk_id=f"clip_{rid}",
                recipe_id=rid,
                content=name,
                section="recipe_name",
                score=float(score),
            ))

        return chunks
