"""
PDF 文档解析 Agent — 上传 PDF → 解析 → 返回 Markdown/Text

流程:
  用户上传 PDF → 保存到临时目录 → pdf_parser.parse_pdf() → 返回解析内容

应用场景:
  - 上传食谱 PDF（菜谱书），提取文本内容
  - 上传食材清单 PDF，提取做菜步骤
  - 上传营养手册 PDF，作为知识库问答
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from src.agents.base import BaseAgent
from src.api.schemas import PDFParseResult

logger = structlog.get_logger()

# 上传目录
UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "data" / "pdf_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class PDFParseAgent(BaseAgent):
    """PDF 文档解析 Agent

    接收 base64 编码的 PDF → 解析为 Markdown/Text → 返回结构化内容。
    支持 method="auto"|"basic"|"enhanced" 三种模式。
    """

    def __init__(self):
        super().__init__(name="pdf_parse", timeout=120.0, max_retries=1)

    async def _execute(self, **kwargs: Any) -> PDFParseResult:
        pdf_base64 = kwargs.get("pdf", "") or kwargs.get("file", "")
        filename = kwargs.get("filename", "document.pdf")
        method = kwargs.get("method", "auto")
        max_pages = kwargs.get("max_pages", 0)

        if not pdf_base64:
            return PDFParseResult(
                success=False,
                error="未提供 PDF 文件内容（base64）",
                data={"note": "请上传 PDF 文件。"},
            )

        # 解码 base64
        import base64
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
        except Exception as exc:
            return PDFParseResult(
                success=False,
                error=f"PDF base64 解码失败: {exc}",
            )

        # 校验文件头
        if not pdf_bytes[:4] in (b"%PDF",):
            return PDFParseResult(
                success=False,
                error="文件格式错误：不是有效的 PDF 文件",
            )

        # 保存到上传目录（供后续使用）
        safe_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = UPLOAD_DIR / safe_name
        with open(save_path, "wb") as f:
            f.write(pdf_bytes)

        # 解析
        from src.core.pdf_parser import parse_pdf

        start = time.perf_counter()
        result = parse_pdf(pdf_bytes, filename=filename, method=method)
        elapsed_ms = (time.perf_counter() - start) * 1000

        text = result["text"]
        pages = result["pages"]
        parse_method = result["method"]
        metadata = result["metadata"]

        # 截取页数限制
        if max_pages > 0 and pages > max_pages:
            # 用 PyMuPDF 截取前 max_pages 页
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            truncated = fitz.open()
            truncated.insert_pdf(doc, from_page=0, to_page=min(max_pages, pages) - 1)
            truncated_bytes = truncated.tobytes()
            truncated.close()
            doc.close()
            result = parse_pdf(truncated_bytes, filename=filename, method=method)
            text = result["text"]
            pages = result["pages"]

        # 提取第一页摘要（用于快速预览）
        first_page_text = ""
        if text:
            lines = text.strip().split("\n")
            preview_lines = [l for l in lines if l.strip()][:10]
            first_page_text = "\n".join(preview_lines)

        return PDFParseResult(
            success=result["success"],
            text=text,
            pages=pages,
            method=parse_method,
            metadata=metadata,
            data={
                "filename": filename,
                "saved_path": str(save_path),
                "preview": first_page_text[:500],
                "elapsed_ms": round(elapsed_ms, 1),
            },
        )
