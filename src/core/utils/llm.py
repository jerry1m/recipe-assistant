"""
共享 LLM 工具 — API 优先（ModelScope/Qwen3.5-35B），失败时回退到本地 Qwen2.5-0.5B
单例模式，本地模型仅加载一次，所有 Agent 共享

Circuit Breaker 集成: API 调用受熔断器保护，连续失败后快速失败并回退到本地模型
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from functools import lru_cache
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from dotenv import load_dotenv
load_dotenv()

from src.core.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

logger = logging.getLogger(__name__)

# ── API 配置（从环境变量读取，.env 自动加载） ──
_API_KEY = os.environ.get("RECIPE_LLM_API_KEY", "")
_API_BASE_URL = os.environ.get("RECIPE_LLM_BASE_URL", "https://api-inference.modelscope.cn/v1")
_API_MODEL = os.environ.get("RECIPE_LLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
_API_TIMEOUT = float(os.environ.get("RECIPE_LLM_TIMEOUT", "30"))

# ── API 熔断器（全局单例） ──
_api_breaker = CircuitBreaker(
    failure_threshold=int(os.environ.get("RECIPE_CIRCUIT_BREAKER_THRESHOLD", "3")),
    recovery_timeout=float(os.environ.get("RECIPE_CIRCUIT_BREAKER_TIMEOUT", "60.0")),
    half_open_max_calls=int(os.environ.get("RECIPE_CIRCUIT_BREAKER_HALF_OPEN", "2")),
    name="llm_api",
)

# ── 本地模型单例 ──
_model = None
_tokenizer = None
_model_lock = asyncio.Lock()


def _load_model():
    """懒加载 Qwen2.5-0.5B-Instruct（同步，在线程中执行）"""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    logger.info(f"Loading local LLM: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    logger.info("Local LLM loaded successfully")
    return _model, _tokenizer


@lru_cache(maxsize=1)
def _get_system_prompt() -> str:
    return (
        "You are a professional recipe assistant. Answer the user's question "
        "based on the provided recipe context. If the context doesn't contain "
        "enough information, say so honestly. Always cite the recipe names you reference. "
        "Use a friendly and helpful tone. Answer in the same language as the user's question."
    )


# ── API 调用（OpenAI 兼容） ──


def _api_generate_sync(
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> str | None:
    """调用远程 API，失败返回 None（受熔断器保护）"""
    if not _API_KEY:
        logger.warning("No RECIPE_LLM_API_KEY configured, skipping API call")
        return None

    # 检查熔断器状态（快速失败路径）
    if not _api_breaker.is_available:
        logger.warning(
            "circuit_breaker.rejecting",
            name=_api_breaker.name,
            state=_api_breaker.state.value,
        )
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=_API_KEY, base_url=_API_BASE_URL, timeout=_API_TIMEOUT)
        resp = client.chat.completions.create(
            model=_API_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else ""
    except Exception as e:
        logger.warning(f"API call failed, falling back to local model: {e}")
        return None


# ── 本地模型调用 ──


def _generate_sync(
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """同步生成（在线程中执行）"""
    model, tokenizer = _load_model()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# ── 中文字符检测 ──
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")


async def _api_call_with_breaker(
    sync_fn, *args, **kwargs
) -> Any:
    """通过熔断器调用 API 的同步函数。熔断打开时返回 None。"""
    try:
        return await _api_breaker.call(
            lambda: asyncio.to_thread(sync_fn, *args, **kwargs)
        )
    except CircuitBreakerOpenError:
        logger.warning("circuit_breaker.open_skip", name=_api_breaker.name)
        return None


async def translate_query(query: str) -> str:
    """
    检测 query 是否含中文，若有则翻译为英文。
    不含中文则原样返回。
    优先使用 API（受熔断器保护），API 不可用时回退到本地模型。
    """
    if not _CJK_RE.search(query):
        return query

    sp = (
        "You are a strict translator. Translate Chinese to English. "
        "Rules:\n"
        "1. Translate ONLY, do NOT answer the question.\n"
        "2. Return ONLY the English translation, nothing else.\n"
        "3. Keep food/cooking terms accurate.\n"
        "4. If the query is already English, return it unchanged.\n"
        "5. Output a single short sentence, no explanations.\n"
        "Example:\n"
        "  鸡肉的热量是多少 → How many calories in chicken?"
    )
    messages = [
        {"role": "system", "content": sp},
        {"role": "user", "content": query},
    ]

    # API 优先（受熔断器保护）
    result = await _api_call_with_breaker(_api_generate_sync, messages, 128, 0.1)
    if result is not None:
        result = result.strip().strip('"\'').strip()
        logger.info("query.translated[api]", original=query, translated=result)
        return result

    # 回退到本地模型
    async with _model_lock:
        translation = await asyncio.to_thread(_generate_sync, messages, 128, 0.1)
    result = translation.strip().strip('"\'').strip()
    logger.info("query.translated[local]", original=query, translated=result)
    return result


async def generate(
    query: str,
    context: str = "",
    system_prompt: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    异步调用 LLM — API 优先（受熔断器保护），失败回退到本地。

    参数:
        query: 用户问题
        context: 检索得到的上下文文本（Chunk 内容）
        system_prompt: 自定义系统提示（None 则用默认）
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
    """
    sp = system_prompt or _get_system_prompt()
    messages = [{"role": "system", "content": sp}]
    if context:
        messages.append({
            "role": "user",
            "content": f"Here is the recipe context:\n{context}\n\nQuestion: {query}"
        })
    else:
        messages.append({"role": "user", "content": query})

    # API 优先（受熔断器保护）
    result = await _api_call_with_breaker(
        _api_generate_sync, messages, max_new_tokens, temperature
    )
    if result is not None:
        return result

    # 回退到本地模型
    async with _model_lock:
        result = await asyncio.to_thread(
            _generate_sync,
            messages,
            max_new_tokens,
            temperature,
        )
    return result


# ── API 调用（支持 Function Calling） ──


def _api_generate_tools_sync(
    messages: list[dict[str, str]],
    tools: list[dict],
    tool_choice: str | dict = "auto",
    max_tokens: int = 256,
) -> tuple[str | None, dict | None]:
    """调用远程 API 并返回 function call 结果。
    返回 (content, tool_call) — content 为文本回复，tool_call 为模型选择的函数调用。
    API 失败时返回 (None, None)。受熔断器保护。
    """
    if not _API_KEY:
        logger.warning("No RECIPE_LLM_API_KEY configured, skipping API call")
        return None, None

    # 检查熔断器状态（快速失败路径）
    if not _api_breaker.is_available:
        logger.warning(
            "circuit_breaker.rejecting_tools",
            name=_api_breaker.name,
            state=_api_breaker.state.value,
        )
        return None, None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=_API_KEY, base_url=_API_BASE_URL, timeout=_API_TIMEOUT)
        resp = client.chat.completions.create(
            model=_API_MODEL,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperature=0.1,
        )

        choice = resp.choices[0].message

        # 检查是否有 function call
        if choice.tool_calls and len(choice.tool_calls) > 0:
            tc = choice.tool_calls[0]
            import json
            args = json.loads(tc.function.arguments)
            return choice.content, {
                "name": tc.function.name,
                "arguments": args,
            }

        # 没有 function call，返回文本
        text = choice.content
        return text.strip() if text else "", None

    except Exception as e:
        logger.warning(f"API function calling failed: {e}")
        return None, None


async def generate_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict],
    tool_choice: str | dict = "auto",
    max_tokens: int = 256,
) -> tuple[str | None, dict | None]:
    """异步调用 LLM with function calling。返回 (text_content, tool_call_dict)。
    tool_call_dict = {"name": "function_name", "arguments": {key: value}}
    API 优先（受熔断器保护），失败时回退到本地模型（仅返回 text，不支持本地 tools）。
    """
    result = await _api_call_with_breaker(
        _api_generate_tools_sync, messages, tools, tool_choice, max_tokens
    )
    if result is not None:
        return result

    # 回退到本地模型（不支持 tools，只返回文本）
    async with _model_lock:
        text = await asyncio.to_thread(_generate_sync, messages, max_tokens, 0.1)
    return text, None


async def generate_structured(
    query: str,
    context: str = "",
    system_prompt: str | None = None,
    max_new_tokens: int = 768,
    temperature: float = 0.3,
) -> str:
    """
    生成结构化输出（用于替换推理等需要格式化结果的场景）。
    在 system_prompt 中说明输出格式即可。
    API 优先，失败回退到本地。
    """
    return await generate(
        query=query,
        context=context,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def get_api_breaker_stats() -> dict:
    """获取 API 熔断器统计（用于 /metrics/circuit_breaker 端点）"""
    return _api_breaker.get_stats()


def is_local_model_loaded() -> bool:
    """检查本地模型是否已加载（用于健康检查）"""
    global _model
    return _model is not None
