"""
意图识别 Agent — 使用 Function Calling 实现高精度意图分类 + 槽位提取
策略: 规则快速通道 (0ms) → Function Calling (高精度) → 规则兜底
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import IntentType, RouterResult
from src.core.utils.llm import generate_with_tools

# ── Function Calling 工具定义 ──
# 每个 intent 对应一个 function，function name = intent name
# description 就是分类依据，parameters 提取槽位
_INTENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ingredient_recommend",
            "description": "用户想要推荐食材、菜谱，询问'有什么菜''怎么做某道菜'，或请求推荐。例如：推荐一道红烧肉、有什么简单的家常菜、我想做鱼香肉丝",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "用户提到的具体菜名，没有则为空字符串",
                    },
                    "cuisine_type": {
                        "type": "string",
                        "description": "偏好的菜系类型（如川菜、粤菜、西餐），没有则为空字符串",
                    },
                    "dietary_preference": {
                        "type": "string",
                        "description": "饮食偏好或限制（如低热量、素食、无辣），没有则为空字符串",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "step_qa",
            "description": "用户询问烹饪步骤、操作细节、时间火候等。例如：蒸鱼要多久、烤箱温度设置多少、红烧肉怎么做（指做法步骤）、需要哪些调料",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "用户提到的菜名，没有则为空字符串",
                    },
                    "step_type": {
                        "type": "string",
                        "enum": ["time", "temperature", "technique", "ingredient_list", "general"],
                        "description": "询问的具体方面：时间、温度、技巧、配料表、或通用步骤",
                    },
                },
                "required": ["step_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nutrition_filter",
            "description": "用户查询营养信息，询问热量、卡路里、蛋白质、脂肪、碳水、维生素等。例如：鸡肉的热量、番茄炒蛋的蛋白质含量、哪些菜低卡路里",
            "parameters": {
                "type": "object",
                "properties": {
                    "food": {
                        "type": "string",
                        "description": "用户查询的食物或菜品名称，没有则为空字符串",
                    },
                    "nutrient": {
                        "type": "string",
                        "description": "具体营养元素：calories/protein/fat/carbs/fiber/sodium，不明确则为空字符串",
                    },
                    "cuisine_type": {
                        "type": "string",
                        "description": "偏好的菜系类型（如American、Chinese、Italian等），没有则为空字符串",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substitution",
            "description": "用户询问食材替换、替代方案，或说'没有某食材可以用什么代替'。例如：没有鸡蛋可以用什么代替、鱼香肉丝没有豆瓣酱怎么办、素食者用什么替代肉",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredient": {
                        "type": "string",
                        "description": "需要被替换的食材名称",
                    },
                    "recipe_name": {
                        "type": "string",
                        "description": "涉及的菜名，没有则为空字符串",
                    },
                    "dietary_restriction": {
                        "type": "string",
                        "description": "饮食限制原因（如过敏、素食、没有存货），没有则为空字符串",
                    },
                },
                "required": ["ingredient"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdf_parse",
            "description": "用户上传了PDF文件，需要解析PDF文档内容（如食谱PDF、食材清单PDF等）。注意：如果用户提问但没上传文件，不要识别为此意图",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "PDF 内容描述（用户提到的内容）",
                    },
                    "pages": {
                        "type": "string",
                        "description": "用户指定的页码范围（如'前10页'），没有则为空",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "用户想看菜品图片、照片，或说'看起来怎么样''show me'等。例如：红烧肉的照片、这道菜看起来怎么样、我想看看鱼香肉丝的样子",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "用户想看的菜名",
                    },
                },
                "required": ["recipe_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chitchat",
            "description": "用户打招呼、问候、闲聊，或与食谱无关的提问。例如：你好、再见、今天天气怎么样、你会做什么",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["greeting", "farewell", "irrelevant", "other"],
                        "description": "闲聊类型：问候、告别、无关话题、其他",
                    },
                },
                "required": ["query_type"],
            },
        },
    },
]

# 反向映射：function name → IntentType
_NAME_TO_INTENT = {
    "ingredient_recommend": IntentType.INGREDIENT_RECOMMEND,
    "step_qa": IntentType.STEP_QA,
    "nutrition_filter": IntentType.NUTRITION_FILTER,
    "substitution": IntentType.SUBSTITUTION,
    "image_search": IntentType.IMAGE_SEARCH,
    "pdf_parse": IntentType.PDF_PARSE,
    "chitchat": IntentType.CHITCHAT,
}


class RouterAgent(BaseAgent):
    """意图识别 + 槽位提取（Function Calling + 规则混合，支持多轮上下文）"""

    def __init__(self):
        super().__init__(name="router", timeout=10.0, max_retries=2)

    async def _execute(self, **kwargs: Any) -> RouterResult:
        query = kwargs.get("query", "")
        conversation_context = kwargs.get("conversation_context", "")
        last_slots = kwargs.get("last_slots", {})
        files = kwargs.get("files", [])

        # ── 规则快速通道 ──

        # PDF 文件预检查：只要有上传的 PDF 文件即路由到 pdf_parse
        if files:
            return RouterResult(
                intent=IntentType.PDF_PARSE,
                slots={
                    "query": query,
                    "description": query or "解析上传的PDF文档",
                    "pages": "",
                },
                confidence=0.95,
            )

        rule_intent, rule_confidence, rule_slots = self._rule_based_classify(query)

        # 高置信度规则命中 → 直接返回（0 LLM 延迟）
        if rule_confidence >= 0.9:
            slots = {"query": query, **rule_slots}
            return RouterResult(
                intent=rule_intent,
                slots=slots,
                confidence=rule_confidence,
            )

        # ── Function Calling 分类（带上下文） ──
        fc_intent, fc_slots, fc_confidence = await self._fc_classify(
            query, conversation_context, last_slots,
        )
        if fc_intent is not None:
            fc_slots["query"] = query
            return RouterResult(
                intent=fc_intent,
                slots=fc_slots,
                confidence=fc_confidence,
            )

        # ── 兜底：规则结果 ──
        return RouterResult(
            intent=rule_intent,
            slots={"query": query, **rule_slots},
            confidence=rule_confidence,
        )

    async def _fc_classify(
        self,
        query: str,
        conversation_context: str = "",
        last_slots: dict[str, Any] | None = None,
    ) -> tuple[IntentType | None, dict[str, Any], float]:
        """使用 Function Calling 进行意图分类。
        可传入对话上下文，用于多轮对话中理解省略/指代。
        返回 (intent, slots, confidence)。
        """
        # 构建上下文感知的系统提示
        ctx_parts = [
            "You are a recipe assistant intent classifier. "
            "Analyze the user's query and select the MOST appropriate function "
            "that represents their intent. Pay attention to context and nuance. "
            "If unsure, choose the closest match rather than defaulting to chitchat.",
        ]
        if conversation_context:
            ctx_parts.append(f"\n--- 对话上下文 ---\n{conversation_context}")
        if last_slots:
            ctx_parts.append(f"\n--- 上一轮提取的槽位 ---\n{last_slots}")

        messages = [
            {"role": "system", "content": "\n".join(ctx_parts)},
            {"role": "user", "content": query},
        ]

        try:
            _, tool_call = await generate_with_tools(
                messages=messages,
                tools=_INTENT_TOOLS,
                tool_choice="auto",
                max_tokens=256,
            )

            if tool_call is None:
                return None, {}, 0.0

            name = tool_call["name"]
            arguments = tool_call.get("arguments", {})

            intent = _NAME_TO_INTENT.get(name)
            if intent is None:
                return None, {}, 0.0

            # 根据是否有槽位信息调整置信度
            has_slots = bool(arguments and any(v for v in arguments.values()))
            confidence = 0.92 if has_slots else 0.85

            return intent, arguments, confidence

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"FC classify failed: {e}")
            return None, {}, 0.0

    _CUISINE_KEYWORDS: dict[str, str] = {
        "美国|美式|american|usa": "American",
        "中国|中式|川菜|粤菜|湘菜|chinese": "Chinese",
        "日本|日式|japanese": "Japanese",
        "印度|印式|indian": "Indian",
        "法国|法式|french": "French",
        "意大利|意式|italian": "Italian",
        "墨西哥|墨式|mexican": "Mexican",
        "泰国|泰式|thai": "Thai",
        "韩国|韩式|korean": "Korean",
    }

    @staticmethod
    def _extract_cuisine(query: str) -> str:
        """从 query 中提取菜系关键词"""
        q_lower = query.lower()
        for pattern, cuisine_en in RouterAgent._CUISINE_KEYWORDS.items():
            import re
            if re.search(pattern, q_lower):
                return cuisine_en
        return ""

    def _rule_based_classify(self, query: str) -> tuple[IntentType, float, dict]:
        """基于关键词的意图分类（快速通道，0 LLM 延迟）
        返回 (intent, confidence, slots)
        """
        q = query.lower()

        # ── 高精度关键词（置信度 0.9+，直接返回） ──
        # 营养查询特有词
        if any(kw in q for kw in [
            "热量", "卡路里", "calorie", "kcal",
            "蛋白质", "protein", "脂肪", "fat", "碳水",
            "低卡", "低脂", "低碳", "高蛋白", "低热量",
            "少油", "清淡", "少糖", "无糖",
        ]):
            cuisine = self._extract_cuisine(query)
            slots = {}
            if cuisine:
                slots["cuisine_type"] = cuisine
            return IntentType.NUTRITION_FILTER, 0.92, slots

        # 替换特有词
        if any(kw in q for kw in [
            "代替", "替代", "替换", "substitute", "swap",
            "没有鸡蛋", "没有...用", "可以用什么代替",
        ]):
            return IntentType.SUBSTITUTION, 0.92, {}

        # ── 中等置信度规则（需要 LLM 确认） ──
        if any(kw in q for kw in [
            "推荐", "有什么菜", "suggest", "recommend",
            "怎么做", "how to", "recipe for",
        ]):
            return IntentType.INGREDIENT_RECOMMEND, 0.7, {}

        if any(kw in q for kw in [
            "图片", "照片", "图", "image", "photo", "show me",
        ]):
            return IntentType.IMAGE_SEARCH, 0.7, {}

        if any(kw in q for kw in [
            "步骤", "多少分钟", "火候", "step", "minute", "temperature", "bake", "fry",
        ]):
            return IntentType.STEP_QA, 0.7, {}

        return IntentType.CHITCHAT, 0.5, {}
