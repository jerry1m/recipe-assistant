"""
PDF 文档解析引擎 — 双层策略

1. **快速模式** (PyMuPDF): 纯文本 PDF，直接提取文本
2. **增强模式** (magic-pdf/MinerU): 复杂 PDF（表格、图片、公式）→ Markdown

自动降级：magic-pdf 不可用时静默回退 PyMuPDF。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# ── 模型缓存目录 ──
PDF_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "data" / "pdf_output"
PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _check_magic_pdf() -> bool:
    """检查 magic-pdf CLI 是否可用"""
    try:
        result = subprocess.run(
            ["magic-pdf", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _extract_with_pymupdf(pdf_bytes: bytes) -> str:
    """PyMuPDF 基础文本提取"""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str] = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(pages)


def _extract_with_magic_pdf(pdf_bytes: bytes, filename: str = "document") -> str:
    """magic-pdf 增强解析（PDF → Markdown）"""
    # 写入临时文件
    tmp_dir = tempfile.mkdtemp(prefix="magic_pdf_")
    tmp_pdf = os.path.join(tmp_dir, filename)
    with open(tmp_pdf, "wb") as f:
        f.write(pdf_bytes)

    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        start = time.perf_counter()
        result = subprocess.run(
            ["magic-pdf", "-p", tmp_pdf, "-o", output_dir, "-m", "txt"],
            capture_output=True, text=True, timeout=300,
        )
        elapsed = time.perf_counter() - start
        logger.info("magic_pdf.parse", filename=filename, elapsed_ms=round(elapsed * 1000))

        if result.returncode != 0:
            logger.warning("magic_pdf.failed", filename=filename, stderr=result.stderr[:500])
            return ""

        # 查找生成的 markdown 文件
        md_files = list(Path(output_dir).rglob("*.md"))
        if not md_files:
            logger.warning("magic_pdf.no_md_output", filename=filename)
            return ""

        md_content = md_files[0].read_text(encoding="utf-8")
        return md_content

    except subprocess.TimeoutExpired:
        logger.warning("magic_pdf.timeout", filename=filename)
        return ""
    except Exception as exc:
        logger.warning("magic_pdf.error", filename=filename, error=str(exc))
        return ""
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_pdf(pdf_bytes: bytes, filename: str = "document.pdf",
              method: str = "auto") -> dict[str, Any]:
    """解析 PDF 文件，返回结构化内容

    Args:
        pdf_bytes: PDF 文件二进制内容
        filename: 文件名（用于日志和 magic-pdf）
        method: "auto" | "basic" | "enhanced"

    Returns:
        {
            "text": str,          # 纯文本内容
            "method": str,        # 实际使用的方法 (basic/enhanced)
            "pages": int,         # 页数
            "metadata": dict,     # 元数据
            "success": bool,
        }
    """
    import fitz

    # 获取基础信息
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    metadata = doc.metadata or {}
    doc.close()

    # 决定使用哪种方法
    use_enhanced = method == "enhanced" or (method == "auto" and _check_magic_pdf())

    if use_enhanced:
        markdown = _extract_with_magic_pdf(pdf_bytes, filename)
        if markdown:
            return {
                "text": markdown,
                "method": "enhanced",
                "pages": num_pages,
                "metadata": metadata,
                "success": True,
            }
        logger.info("pdf_parser.magic_pdf_fallback", filename=filename)

    # 基本提取
    text = _extract_with_pymupdf(pdf_bytes)
    return {
        "text": text,
        "method": "basic",
        "pages": num_pages,
        "metadata": metadata,
        "success": bool(text.strip()),
    }


def parse_pdf_file(filepath: str, **kwargs) -> dict[str, Any]:
    """从文件路径解析 PDF"""
    with open(filepath, "rb") as f:
        pdf_bytes = f.read()
    return parse_pdf(pdf_bytes, filename=os.path.basename(filepath), **kwargs)
