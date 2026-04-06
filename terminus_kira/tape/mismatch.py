"""MismatchChecker — two-stage subgoal completion + state mismatch detection via LLM tool calling.

Stage 1: Check if the current subgoal is complete or still in progress.
Stage 2: If complete, check if the actual state matches the predicted next state.
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

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompt-templates"
_COMPLETION_PROMPT_PATH = _PROMPT_DIR / "tape-completion-check.txt"
_MISMATCH_PROMPT_PATH = _PROMPT_DIR / "tape-mismatch.txt"

# Tool description strings (mirroring terminus_kira.py pattern)
_COMPLETION_STATUS_DESC = "Submit whether the current subgoal is complete or still in progress."

_SUBGOAL_STATUS_DESC = (
    "Is the current subgoal complete based on the terminal output? "
    "'completed' means the subgoal's commands have been executed and results are visible "
    "(includes both successful and failed executions). "
    "'in_progress' means commands are still running or not yet attempted."
)

_MISMATCH_RESULT_DESC = "Submit whether the actual state matches the predicted state."

_IS_MISMATCH_DESC = (
    "True if there is a significant mismatch requiring replanning. "
    "Ignore minor differences (formatting, timestamps, wording). "
    "Flag command errors, missing output, wrong values, failed compilation, "
    "or state that makes subsequent planned steps impossible."
)

_MISMATCH_REASON_DESC = "Explanation of the mismatch. Empty string if no mismatch."

# Tool definitions for structured output
COMPLETION_CHECK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_completion_status",
            "description": _COMPLETION_STATUS_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "subgoal_status": {
                        "type": "string",
                        "enum": ["in_progress", "completed"],
                        "description": _SUBGOAL_STATUS_DESC,
                    },
                },
                "required": ["subgoal_status"],
            },
        },
    },
]

MISMATCH_CHECK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_mismatch_result",
            "description": _MISMATCH_RESULT_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "is_mismatch": {
                        "type": "boolean",
                        "description": _IS_MISMATCH_DESC,
                    },
                    "reason": {
                        "type": "string",
                        "description": _MISMATCH_REASON_DESC,
                    },
                },
                "required": ["is_mismatch", "reason"],
            },
        },
    },
]


class SubgoalStatus:
    """Result of the two-stage check."""

    IN_PROGRESS = "in_progress"
    COMPLETED_MATCH = "completed_match"
    COMPLETED_MISMATCH = "completed_mismatch"


class MismatchChecker:
    """Two-stage checker: subgoal completion + state mismatch.

    Stage 1 (every episode):
        Is the current subgoal complete?
        Input: history + current terminal + subgoal + task
        Output: "in_progress" | "completed"

    Stage 2 (only when completed):
        Does the actual state match the predicted state?
        Input: current terminal + predicted_state + task
        Output: is_mismatch + reason
    """

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        temperature: float = 0.1,
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self._completion_prompt_template = _COMPLETION_PROMPT_PATH.read_text()
        self._mismatch_prompt_template = _MISMATCH_PROMPT_PATH.read_text()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type((BadRequestError, LiteLLMAuthenticationError))
        ),
        reraise=True,
    )
    async def _call_llm(self, messages: list[dict], tools: list[dict], tool_name: str) -> dict | None:
        """Call LLM with tool calling and return parsed tool arguments."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
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

    async def check_subgoal_completion(
        self,
        terminal_history: str,
        current_terminal_output: str,
        subgoal_description: str,
        task_instruction: str,
    ) -> bool:
        """Stage 1: Check if the current subgoal is complete.

        Returns True if completed, False if still in progress.
        """
        if len(terminal_history) > 5000:
            terminal_history = "...(truncated)...\n" + terminal_history[-5000:]
        if len(current_terminal_output) > 3000:
            current_terminal_output = current_terminal_output[-3000:]

        # Build dynamic user message
        user_parts = [
            f"# Task\n{task_instruction}",
            f"# Current Subgoal\n{subgoal_description}",
            f"# Terminal History (previous observations)\n{terminal_history or '(none)'}",
            f"# Current Terminal Output (latest)\n{current_terminal_output}",
        ]
        user_message = "\n\n".join(user_parts)

        messages = add_anthropic_caching(
            [
                {"role": "system", "content": self._completion_prompt_template},
                {"role": "user", "content": user_message},
            ],
            self.model_name,
        )

        try:
            result = await self._call_llm(messages, COMPLETION_CHECK_TOOLS, "submit_completion_status")
        except Exception as e:
            logger.warning("[TAPE Completion] LLM call failed: %s", e)
            return False

        if result is None:
            logger.warning("[TAPE Completion] No tool call returned")
            return False

        status = result.get("subgoal_status", "in_progress")
        is_completed = status == "completed"

        if is_completed:
            logger.info("[TAPE Completion] Subgoal completed: %s", subgoal_description)
        else:
            logger.debug("[TAPE Completion] Subgoal in progress: %s", subgoal_description)

        return is_completed

    async def check_mismatch(
        self,
        predicted_state: str,
        actual_terminal_output: str,
        subgoal_description: str,
        task_instruction: str,
    ) -> tuple[bool, str]:
        """Stage 2: Check if actual state matches predicted state.

        Only called when subgoal is completed (Stage 1 returned True).

        Returns:
            (is_mismatch, reason)
        """
        if len(actual_terminal_output) > 3000:
            actual_terminal_output = actual_terminal_output[-3000:]

        # Build dynamic user message
        user_parts = [
            f"# Task\n{task_instruction}",
            f"# Subgoal Just Executed\n{subgoal_description}",
            f"# Predicted State (from simulation)\n{predicted_state}",
            f"# Actual Terminal Output\n{actual_terminal_output}",
        ]
        user_message = "\n\n".join(user_parts)

        messages = add_anthropic_caching(
            [
                {"role": "system", "content": self._mismatch_prompt_template},
                {"role": "user", "content": user_message},
            ],
            self.model_name,
        )

        try:
            result = await self._call_llm(messages, MISMATCH_CHECK_TOOLS, "submit_mismatch_result")
        except Exception as e:
            logger.warning("[TAPE Mismatch] LLM call failed: %s", e)
            return False, ""

        if result is None:
            logger.warning("[TAPE Mismatch] No tool call returned")
            return False, ""

        is_mismatch = bool(result.get("is_mismatch", False))
        reason = str(result.get("reason", ""))

        if is_mismatch:
            logger.info("[TAPE Mismatch] Detected: %s", reason)
        else:
            logger.debug("[TAPE Mismatch] No mismatch")

        return is_mismatch, reason

    async def check(
        self,
        terminal_history: str,
        current_terminal_output: str,
        subgoal_description: str,
        predicted_state: str,
        task_instruction: str,
    ) -> str:
        """Full two-stage check. Convenience method.

        Returns one of:
            SubgoalStatus.IN_PROGRESS — subgoal not done yet
            SubgoalStatus.COMPLETED_MATCH — done, state matches prediction
            SubgoalStatus.COMPLETED_MISMATCH — done, state diverged
        """
        is_completed = await self.check_subgoal_completion(
            terminal_history=terminal_history,
            current_terminal_output=current_terminal_output,
            subgoal_description=subgoal_description,
            task_instruction=task_instruction,
        )

        if not is_completed:
            return SubgoalStatus.IN_PROGRESS

        is_mismatch, reason = await self.check_mismatch(
            predicted_state=predicted_state,
            actual_terminal_output=current_terminal_output,
            subgoal_description=subgoal_description,
            task_instruction=task_instruction,
        )

        if is_mismatch:
            logger.info("[TAPE] Completed with mismatch: %s", reason)
            return SubgoalStatus.COMPLETED_MISMATCH

        return SubgoalStatus.COMPLETED_MATCH
