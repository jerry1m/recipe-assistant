"""
规划 Agent (Planner) — 将复杂食谱查询分解为有序步骤并按序执行

参考 travel-agent-guide 的 PlanningAgent 设计，适配 recipe-assistant 场景。
用于"帮我规划一周健康食谱""准备一个三人的晚宴菜单"等复杂请求。
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.agents.base import BaseAgent
from src.api.schemas import PlannerResult
from src.core.tools.registry import get_tool_registry, invoke as tool_invoke

# ── 规划用的 system prompt ──
_PLANNER_SYSTEM_PROMPT = (
    "You are a professional recipe planning assistant. "
    "Your role is to decompose complex recipe/meal requests into executable steps.\n\n"
    "Available tools:\n"
    "{tools_context}\n\n"
    "Rules:\n"
    "1. Analyze the user's request and break it down into 2-5 concrete steps.\n"
    "2. Each step should map to ONE tool invocation if possible.\n"
    "3. Output a JSON plan with the following format:\n"
    "   {\n"
    "     \"goal\": \"用户目标的简要描述\",\n"
    "     \"steps\": [\n"
    "       {\"step\": 1, \"action\": \"search_recipes\", \"params\": {\"query\": \"低热量晚餐\", \"top_k\": 5}},\n"
    "       {\"step\": 2, \"action\": \"search_nutrition\", \"params\": {\"food\": \"鸡胸肉\"}}\n"
    "     ],\n"
    "     \"summary_instructions\": \"如何整合各步骤结果的说明\"\n"
    "   }\n\n"
    "4. If the request doesn't need planning (simple query), return a single-step plan.\n"
    "5. Use ONLY the tools listed above. If no tool fits, set action to \"reason\" and explain in params.\n"
    "6. Respond with ONLY the JSON, no other text."
)


class PlannerAgent(BaseAgent):
    """
    规划 Agent：将复杂食谱任务分解为步骤，依次通过 ToolRegistry 执行。
    """

    _STEP_LINE = re.compile(r"^\s*\d+[\).、]\s*(.+)$")

    def __init__(self):
        super().__init__(name="planner", timeout=30.0, max_retries=2)

    async def _execute(self, **kwargs: Any) -> PlannerResult:
        query = kwargs.get("query", "")
        context = kwargs.get("context", "")
        user_preferences = kwargs.get("user_preferences", "")

        registry = get_tool_registry()
        tools = registry.list_tools()
        tools_context = "\n".join(
            f"  - {t.name}: {t.description}" for t in tools
        )

        full_context = f"用户目标：{query}\n"
        if context:
            full_context += f"对话上下文：{context}\n"
        if user_preferences:
            full_context += f"用户偏好：{user_preferences}\n"

        # ── 1. LLM 生成规划 ──
        from src.core.utils.llm import generate_structured
        planner_prompt = _PLANNER_SYSTEM_PROMPT.format(tools_context=tools_context)
        plan_text = await generate_structured(
            query=full_context,
            context="",
            system_prompt=planner_prompt,
            max_new_tokens=1024,
            temperature=0.2,
        )

        # ── 2. 解析 JSON 规划 ──
        goal, steps, summary_instructions = self._parse_plan(plan_text)

        if not steps:
            return PlannerResult(
                success=False,
                goal=query,
                steps=[],
                step_outputs=[],
                summary=f"规划解析失败，原始输出：{plan_text[:200]}",
            )

        # ── 3. 依次执行每一步 ──
        step_outputs: list[dict[str, Any]] = []
        all_success = True

        for step_def in steps:
            action = step_def.get("action", "reason")
            params = step_def.get("params", {})
            step_desc = step_def.get("step", f"{action}")

            try:
                if action == "reason":
                    # 纯推理步骤（无需调用工具）
                    from src.core.utils.llm import generate
                    reasoning = await generate(
                        query=str(params),
                        system_prompt="你是食谱规划助手，根据已有信息给出专业建议。",
                        max_new_tokens=512,
                    )
                    step_outputs.append({
                        "step": step_desc,
                        "action": "reason",
                        "output": reasoning,
                        "success": True,
                    })
                else:
                    result = await tool_invoke(registry, action, params)
                    step_outputs.append({
                        "step": step_desc,
                        "action": action,
                        "output": str(result)[:500],
                        "success": True,
                    })
            except KeyError:
                # 工具不存在，回退到 LLM 推理
                from src.core.utils.llm import generate
                reasoning = await generate(
                    query=f"用户请求步骤「{step_desc}」，但工具 {action} 不可用。请根据已有知识给出建议。参数：{params}",
                    system_prompt="你是食谱规划助手。",
                    max_new_tokens=512,
                )
                step_outputs.append({
                    "step": step_desc,
                    "action": "reason",
                    "output": reasoning,
                    "success": True,
                })
            except Exception as e:
                all_success = False
                step_outputs.append({
                    "step": step_desc,
                    "action": action,
                    "output": f"执行失败：{e}",
                    "success": False,
                })

        # ── 4. 汇总结果 ──
        summary = await self._generate_summary(
            goal, steps, step_outputs, summary_instructions
        )

        return PlannerResult(
            success=all_success,
            goal=goal,
            steps=[s.get("step_desc", str(s.get("step", ""))) for s in steps],
            step_outputs=step_outputs,
            summary=summary,
        )

    def _parse_plan(self, text: str) -> tuple[str, list[dict], str]:
        """从 LLM 输出中解析 JSON 规划。"""
        # 提取 JSON（可能被 ``` 包裹）
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        try:
            plan = json.loads(text)
            goal = plan.get("goal", "")
            steps = plan.get("steps", [])
            summary_instructions = plan.get("summary_instructions", "")
            return goal, steps, summary_instructions
        except json.JSONDecodeError:
            # 尝试找第一个 { 和最后一个 }
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    plan = json.loads(text[start:end+1])
                    return plan.get("goal", ""), plan.get("steps", []), plan.get("summary_instructions", "")
                except json.JSONDecodeError:
                    pass
            return text[:100], [], ""

    async def _generate_summary(
        self,
        goal: str,
        steps: list[dict],
        step_outputs: list[dict[str, Any]],
        instructions: str,
    ) -> str:
        """用 LLM 将步骤结果汇总为最终回答。"""
        parts = [f"# 规划目标：{goal}\n"]
        for i, (sd, so) in enumerate(zip(steps, step_outputs), 1):
            step_desc = sd.get("step_desc", str(sd.get("step", f"步骤{i}")))
            parts.append(f"## 步骤 {i}: {step_desc}")
            parts.append(so.get("output", "（无输出）"))

        context = "\n".join(parts)

        from src.core.utils.llm import generate
        summary = await generate(
            query=f"请根据以下规划执行结果，生成一份完整的回答给用户。\n汇总说明：{instructions}",
            context=context,
            system_prompt="你是专业的食谱规划助手。用友好、清晰的语言将规划结果呈现给用户。",
            max_new_tokens=1024,
        )
        return summary
