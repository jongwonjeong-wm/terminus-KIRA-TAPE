"""MismatchChecker — single-stage step judgment via LLM tool calling.

After each agent turn, judges whether to COMPLETE (subgoal achieved,
move to next), IN_PROGRESS (keep working on same subgoal), or
REPLAN (state diverged, need new plan).
"""

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

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompt-templates" / "tape-step-judge.txt"
)

# Tool definition for structured output
STEP_JUDGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_step_judgment",
            "description": "Submit the judgment for this step.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["complete", "in_progress", "replan"],
                        "description": (
                            "complete: subgoal achieved, move to next. "
                            "in_progress: agent is working toward the subgoal but has not achieved it yet. "
                            "replan: state diverged, need new plan."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of the decision.",
                    },
                },
                "required": ["decision", "reason"],
            },
        },
    },
]


class SubgoalStatus:
    """Result of step judgment."""

    COMPLETE = "complete"            # Subgoal achieved, move to next
    IN_PROGRESS = "in_progress"      # Working toward subgoal, stay on it
    REPLAN = "replan"                # State diverged, need new plan


class MismatchChecker:
    """Single-stage step judge: decides COMPLETE / IN_PROGRESS / REPLAN after each turn."""

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self._prompt_template = _PROMPT_PATH.read_text()

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
            "tools": STEP_JUDGE_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "submit_step_judgment"}},
            "max_tokens": 512,
            "timeout": 60,
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

    async def check(
        self,
        terminal_history: str,
        current_terminal_output: str,
        subgoal_description: str,
        predicted_state: str,
        task_instruction: str,
        agent_analysis: str = "",
        agent_plan: str = "",
        recent_outputs: list[str] | None = None,
    ) -> str:
        """Single-call step judgment.

        Returns one of:
            SubgoalStatus.COMPLETE — subgoal achieved, proceed to next
            SubgoalStatus.IN_PROGRESS — still working, stay on subgoal
            SubgoalStatus.REPLAN — state diverged, need new plan
        """
        if len(current_terminal_output) > 3000:
            current_terminal_output = current_terminal_output[-3000:]

        # Build recent context from previous episodes
        recent_context = ""
        if recent_outputs:
            for i, output in enumerate(recent_outputs):
                truncated = output[-1500:] if len(output) > 1500 else output
                recent_context += f"[Turn -{len(recent_outputs)-i}]\n{truncated}\n\n"

        user_parts = [
            f"# Task\n{task_instruction}",
            f"# Current Subgoal\n{subgoal_description}",
            f"# Predicted State (from simulation)\n{predicted_state or '(none)'}",
            f"# Agent's Intent\n"
            f"Analysis: {agent_analysis or '(none)'}\n"
            f"Plan: {agent_plan or '(none)'}",
        ]
        if recent_context:
            user_parts.append(f"# Previous Turns\n{recent_context}")
        user_parts.append(f"# Current Terminal Output\n{current_terminal_output}")
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
            logger.warning("[TAPE Judge] LLM call failed: %s — defaulting to IN_PROGRESS", e)
            return SubgoalStatus.IN_PROGRESS

        if result is None:
            logger.warning("[TAPE Judge] No tool call returned — defaulting to IN_PROGRESS")
            return SubgoalStatus.IN_PROGRESS

        decision = result.get("decision", "in_progress")
        reason = result.get("reason", "")

        if decision == "complete":
            logger.info("[TAPE Judge] COMPLETE: %s", reason)
            return SubgoalStatus.COMPLETE
        elif decision == "replan":
            logger.info("[TAPE Judge] REPLAN: %s", reason)
            return SubgoalStatus.REPLAN
        else:
            logger.info("[TAPE Judge] IN_PROGRESS: %s", reason)
            return SubgoalStatus.IN_PROGRESS
