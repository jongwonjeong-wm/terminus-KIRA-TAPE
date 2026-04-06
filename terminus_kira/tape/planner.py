"""TAPEPlanner — generates M candidate high-level plans via LLM tool calling."""

import asyncio
import json
import logging
from pathlib import Path

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
)

from anthropic_caching import add_anthropic_caching
from terminus_kira.tape.types import Plan, Subgoal, ToolType

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "prompt-templates" / "tape-planner.txt"
)

_TOOL_TYPE_MAP = {
    "execute_commands": ToolType.EXECUTE_COMMANDS,
    "task_complete": ToolType.TASK_COMPLETE,
    "image_read": ToolType.IMAGE_READ,
}

# Tool description strings (mirroring terminus_kira.py pattern)
_SUBMIT_PLAN_DESC = "Submit a high-level plan as a sequence of subgoals."

_SUBGOALS_DESC = "Ordered list of subgoals to accomplish the task."

_SUBGOAL_DESC = (
    "What this step accomplishes (specific and actionable). "
    "The agent will analyze the current terminal state "
    "(what has been accomplished, what still needs to be done), "
    "then plan specific commands with expected outcomes. "
    "Write the subgoal so the agent can form a clear analysis and plan from it."
)

_PREDICTED_TOOL_DESC = (
    "Which tool the agent should use for this step. "
    "execute_commands: Execute commands in the terminal. "
    "The agent first analyzes the current state based on the terminal output "
    "(what do you see? what has been accomplished? what still needs to be done?), "
    "then describes a plan for the next steps "
    "(what commands will you run and why? what do you expect each command to accomplish?), "
    "then provides the commands with wait durations "
    "(0.1s for immediate tasks like cd/ls/echo/cat, "
    "1.0s for normal commands like gcc/find/rustc, "
    "longer for slow commands like make/wget but never >60s). "
    "task_complete: Signal that the task is finished. "
    "image_read: Read and analyze an image file visually."
)

# Tool definition for structured plan output
PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": _SUBMIT_PLAN_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "subgoals": {
                        "type": "array",
                        "description": _SUBGOALS_DESC,
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": _SUBGOAL_DESC,
                                },
                                "predicted_tool": {
                                    "type": "string",
                                    "enum": ["execute_commands", "task_complete", "image_read"],
                                    "description": _PREDICTED_TOOL_DESC,
                                },
                            },
                            "required": ["description", "predicted_tool"],
                        },
                    },
                },
                "required": ["subgoals"],
            },
        },
    },
]


class TAPEPlanner:
    """Generates M candidate plans by calling LLM with diverse prompts."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        M: int = 4,
        temperature: float = 0.8,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.M = M
        self.temperature = temperature
        self._prompt_template = _PROMPT_TEMPLATE_PATH.read_text()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type((BadRequestError, LiteLLMAuthenticationError))
        ),
        reraise=True,
    )
    async def _call_llm(self, messages: list[dict]) -> dict | None:
        """Call LLM with tool calling and return parsed tool arguments."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "tools": PLANNER_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "submit_plan"}},
            "timeout": 120,
            "drop_params": True,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = await litellm.acompletion(**kwargs)
        message = response.choices[0].message

        if hasattr(message, "tool_calls") and message.tool_calls:
            tc = message.tool_calls[0]
            args_str = tc.function.arguments
            return json.loads(args_str) if isinstance(args_str, str) else args_str

        return None

    async def _generate_single_plan(
        self,
        plan_id: int,
        task_instruction: str,
        current_terminal_state: str,
        chat_history_summary: str,
    ) -> Plan | None:
        """Generate a single candidate plan."""
        user_parts = [f"# Task\n{task_instruction}"]

        if chat_history_summary:
            user_parts.append(f"# Interaction History\n{chat_history_summary}")

        user_parts.append(f"# Current Terminal State\n{current_terminal_state or '(empty)'}")

        user_message = "\n\n".join(user_parts)

        messages = add_anthropic_caching(
            [
                {"role": "system", "content": self._prompt_template},
                {"role": "user", "content": user_message},
            ],
            self.model_name,
        )

        try:
            result = await self._call_llm(messages)
        except Exception as e:
            logger.warning("[TAPE Planner] Plan %d LLM call failed: %s", plan_id, e)
            return None

        if result is None:
            logger.warning("[TAPE Planner] Plan %d returned no tool call", plan_id)
            return None

        raw_subgoals = result.get("subgoals", [])
        if not raw_subgoals:
            logger.warning("[TAPE Planner] Plan %d produced 0 subgoals", plan_id)
            return None

        subgoals = []
        for i, step in enumerate(raw_subgoals):
            tool_str = step.get("predicted_tool", "execute_commands")
            tool_type = _TOOL_TYPE_MAP.get(tool_str, ToolType.EXECUTE_COMMANDS)
            subgoals.append(
                Subgoal(
                    id=f"subgoal_p{plan_id}_s{i}",
                    description=step.get("description", ""),
                    predicted_tool=tool_type,
                )
            )

        return Plan(plan_id=plan_id, subgoals=subgoals)

    async def generate_plans(
        self,
        task_instruction: str,
        current_terminal_state: str,
        chat_history_summary: str = "",
    ) -> list[Plan]:
        """Generate M candidate plans concurrently.

        Diversity comes from temperature-based sampling (temperature=0.8),
        not from explicit style hints. This follows the TAPE paper approach.
        """
        tasks = [
            self._generate_single_plan(
                plan_id=i,
                task_instruction=task_instruction,
                current_terminal_state=current_terminal_state,
                chat_history_summary=chat_history_summary,
            )
            for i in range(self.M)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        plans = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("[TAPE Planner] Plan %d raised exception: %s", i, result)
            elif result is not None:
                plans.append(result)

        logger.info("[TAPE Planner] Generated %d/%d plans", len(plans), self.M)
        return plans
