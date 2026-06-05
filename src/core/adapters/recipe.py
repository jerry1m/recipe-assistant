"""
食谱领域适配器 — 集中管理食谱助手的 Prompt/权重/指标/降级话术
"""

from __future__ import annotations

from src.core.adapters.base import DomainAdapter


class RecipeAdapter(DomainAdapter):
    """食谱领域实现"""

    @property
    def name(self) -> str:
        return "recipe"

    def system_prompt(self) -> str:
        return """你是一个专业的食谱助手，擅长：
- 根据食材推荐菜谱
- 回答烹饪步骤相关问题
- 提供食材替换建议
- 查询营养信息

回答要求：
1. 引用来源：每条建议需标注参考的菜谱 ID
2. 营养建议必须附带免责声明
3. 不知道的不要编造，直接说"暂未收录"
4. 使用中文回答，语气亲切"""

    def retrieval_weights(self) -> dict[str, float]:
        return {
            "bm25": 0.3,     # 菜名/食材关键词匹配
            "vector": 0.5,   # 语义相似度
            "rerank": 0.2,   # 精排
        }

    def eval_metrics(self) -> list[str]:
        return [
            "recall@10",
            "faithfulness",
            "sql_accuracy",
            "p95_latency",
            "critic_revision_rate",
            "fallback_rate",
        ]

    def fallback_message(self) -> str:
        return "抱歉，暂时无法找到相关菜谱信息，请尝试换个问法或稍后再试。"

    def disclaimer(self) -> str:
        return "⚠️ 营养建议仅供参考，具体请以专业营养师意见为准。"
