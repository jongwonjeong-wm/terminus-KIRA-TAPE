"""MismatchChecker — single-stage step judgment via LLM tool calling.

After each agent turn, judges whether to ADVANCE (next subgoal)
or REPLAN (state diverged). Binary decision — no CONTINUE state.
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
                        "enum": ["advance", "replan"],
                        "description": (
                            "advance: subgoal done, move to next. "
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

    COMPLETED_MATCH = "completed_match"       # ADVANCE — done, proceed
    COMPLETED_MISMATCH = "completed_mismatch"  # REPLAN — state diverged


class MismatchChecker:
    """Single-stage step judge: decides ADVANCE / REPLAN after each turn."""

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
    ) -> str:
        """Single-call step judgment.

        Returns one of:
            SubgoalStatus.COMPLETED_MATCH — ADVANCE, proceed to next subgoal
            SubgoalStatus.COMPLETED_MISMATCH — REPLAN, state diverged
        """
        if len(current_terminal_output) > 3000:
            current_terminal_output = current_terminal_output[-3000:]

        user_parts = [
            f"# Task\n{task_instruction}",
            f"# Current Subgoal\n{subgoal_description}",
            f"# Predicted State (from simulation)\n{predicted_state or '(none)'}",
            f"# Terminal Output (this turn)\n{current_terminal_output}",
        ]
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
            logger.warning("[TAPE Judge] LLM call failed: %s — defaulting to ADVANCE", e)
            return SubgoalStatus.COMPLETED_MATCH

        if result is None:
            logger.warning("[TAPE Judge] No tool call returned — defaulting to ADVANCE")
            return SubgoalStatus.COMPLETED_MATCH

        decision = result.get("decision", "advance")
        reason = result.get("reason", "")

        if decision == "replan":
            logger.info("[TAPE Judge] REPLAN: %s", reason)
            return SubgoalStatus.COMPLETED_MISMATCH
        else:
            logger.info("[TAPE Judge] ADVANCE: %s", reason)
            return SubgoalStatus.COMPLETED_MATCH
