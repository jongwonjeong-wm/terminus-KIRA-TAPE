"""TAPESimulator — simulates plans step-by-step via multi-turn LLM conversation.

Each plan is simulated in a single conversation where:
1. System: static simulator instructions
2. User: task + history + initial state + first subgoal
3. Assistant: predicted state (via tool call)
4. User: next subgoal
5. Assistant: predicted state (via tool call)
...and so on. This way the LLM naturally carries forward the accumulated state
through the conversation context.
"""

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
from terminus_kira.tape.types import Plan

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "prompt-templates" / "tape-simulator.txt"
)

# Tool description strings (mirroring terminus_kira.py pattern)
_SUBMIT_STEP_DESC = (
    "Submit simulation result for a single subgoal step. "
    "Predict the semantic outcome of executing this subgoal."
)

_STATE_REASON_DESC = (
    "Explain what will happen when this subgoal is executed from the current state. "
    "Consider the environment, dependencies, and likely outcomes before committing to a predicted state."
)

_PREDICTED_STATE_DESC = (
    "Abstract description of what changes after this step. "
    "Describe WHAT was accomplished (files created, services started, errors hit), "
    "NOT the literal terminal output. "
    "Example: 'Compiled successfully, binary created at ./release' "
    "instead of simulating the actual gcc output."
)

_ESTIMATED_DURATION_DESC = (
    "Estimated wall-clock seconds for the entire subgoal. "
    "Sum up all commands the agent would run for this subgoal."
)

_DURATION_REASON_DESC = (
    "Brief reason why this step takes that long."
)

_REWARD_REASON_DESC = (
    "Explain why this state gets that reward score. "
    "Reference the task goal: does this state directly solve the task (reward=1), "
    "bring the agent meaningfully closer to the solution (0 < reward < 1), "
    "neither help nor hurt (reward=0), "
    "or move toward failure / waste effort (reward < 0)?"
)

_REWARD_DESC = (
    "Rate the resulting STATE (not the action) — how does being in this state "
    "affect the probability of ultimately solving the task? "
    "1 = this IS the goal state (task solved). "
    "Between 0 and 1 = state that brings the agent closer to the correct solution "
    "(e.g., 0.8 for a verified working setup, 0.3 for basic progress). "
    "Between -1 and 0 = state that moves toward failure or wastes effort "
    "(e.g., -0.5 for a broken build, -0.2 for an unnecessary detour). "
    "0 = neutral state that neither helps nor hurts."
)

# Tool definition for a single step simulation
SIMULATOR_STEP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_step_simulation",
            "description": _SUBMIT_STEP_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "state_reason": {
                        "type": "string",
                        "description": _STATE_REASON_DESC,
                    },
                    "predicted_state": {
                        "type": "string",
                        "description": _PREDICTED_STATE_DESC,
                    },
                    "duration_reason": {
                        "type": "string",
                        "description": _DURATION_REASON_DESC,
                    },
                    "estimated_duration": {
                        "type": "number",
                        "description": _ESTIMATED_DURATION_DESC,
                    },
                    "reward_reason": {
                        "type": "string",
                        "description": _REWARD_REASON_DESC,
                    },
                    "reward": {
                        "type": "number",
                        "description": _REWARD_DESC,
                    },
                },
                "required": ["state_reason", "predicted_state", "duration_reason", "estimated_duration", "reward_reason", "reward"],
            },
        },
    },
]


class TAPESimulator:
    """Simulates each plan step-by-step in a multi-turn conversation.

    For each plan, a single conversation is maintained:
    - System message: static simulator instructions
    - First user message: task + history + initial state + first subgoal
    - Assistant responds with predicted state (tool call)
    - Next user message: next subgoal to simulate
    - Assistant responds with next predicted state
    - ...repeat until all subgoals are simulated

    Different plans are simulated concurrently (each in its own conversation).
    """

    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.api_base = api_base
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
    async def _call_llm(self, messages: list[dict]) -> tuple[dict | None, dict | None]:
        """Call LLM and return (parsed_args, raw_assistant_message).

        Returns both so we can append the raw message to conversation history.
        """
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "tools": SIMULATOR_STEP_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "submit_step_simulation"}},
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
            parsed = json.loads(args_str) if isinstance(args_str, str) else args_str

            # Build raw assistant message for conversation history
            assistant_msg = {"role": "assistant", "content": message.content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments),
                    },
                }
            ]
            return parsed, assistant_msg

        return None, None

    @staticmethod
    def _format_subgoal(plan_id: int, step_idx: int, subgoal) -> str:
        """Format a subgoal as a user message for the next simulation turn."""
        step_info = json.dumps({
            "subgoal_id": f"subgoal_p{plan_id}_s{step_idx}",
            "description": subgoal.description,
            "predicted_tool": subgoal.predicted_tool.value,
        }, indent=2)
        return (
            f"Now simulate the next subgoal. Predict what happens when it is executed "
            f"from the current state.\n\n{step_info}"
        )

    async def simulate_plan(
        self,
        plan: Plan,
        task_instruction: str,
        initial_state: str,
        history_context: str = "",
    ) -> Plan | None:
        """Simulate a plan in a single multi-turn conversation.

        Turn 1: system + user(task, history, state, first subgoal) -> assistant(prediction)
        Turn 2: user(next subgoal) -> assistant(prediction)
        ...
        """
        # Build initial user message with full context + first subgoal
        first_subgoal = plan.subgoals[0]
        first_step_info = json.dumps({
            "subgoal_id": f"subgoal_p{plan.plan_id}_s0",
            "description": first_subgoal.description,
            "predicted_tool": first_subgoal.predicted_tool.value,
        }, indent=2)

        user_parts = [f"# Task\n{task_instruction}"]
        if history_context:
            user_parts.append(f"# Interaction History\n{history_context}")
        user_parts.append(f"# Current Terminal State\n{initial_state or '(empty)'}")
        user_parts.append(
            f"# First Subgoal to Simulate\n{first_step_info}\n\n"
            f"Predict what happens when this subgoal is executed."
        )
        first_user_message = "\n\n".join(user_parts)

        # Start conversation
        messages = add_anthropic_caching(
            [
                {"role": "system", "content": self._prompt_template},
                {"role": "user", "content": first_user_message},
            ],
            self.model_name,
        )

        # Simulate each step, building up the conversation
        for i, sg in enumerate(plan.subgoals):
            if i > 0:
                # Add next subgoal as user message
                user_msg = self._format_subgoal(plan.plan_id, i, sg)
                messages.append({"role": "user", "content": user_msg})

            try:
                result, assistant_msg = await self._call_llm(messages)
            except Exception as e:
                logger.warning(
                    "[TAPE Simulator] Plan %d step %d failed: %s",
                    plan.plan_id, i, e,
                )
                result, assistant_msg = None, None

            if result is not None and assistant_msg is not None:
                sg.state_reason = result.get("state_reason", "")
                sg.predicted_state = result.get("predicted_state", "")
                sg.duration_reason = result.get("duration_reason", "")
                sg.estimated_duration = float(result.get("estimated_duration", 1.0))
                sg.reward_reason = result.get("reward_reason", "")
                sg.reward = float(result.get("reward", 0.0))

                # Append assistant response + tool result to conversation
                messages.append(assistant_msg)
                tool_call_id = assistant_msg["tool_calls"][0]["id"]
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Recorded. Predicted state: {sg.predicted_state}",
                })
            else:
                sg.predicted_state = f"(simulation failed for step {i})"
                sg.estimated_duration = 1.0
                # Add a placeholder so conversation can continue
                messages.append({
                    "role": "assistant",
                    "content": f"Step {i} simulation failed, continuing with next step.",
                })

        plan.total_estimated_duration = sum(
            sg.estimated_duration for sg in plan.subgoals
        )

        logger.info(
            "[TAPE Simulator] Plan %d: duration=%.1fs",
            plan.plan_id, plan.total_estimated_duration,
        )
        return plan

    async def simulate_all(
        self,
        plans: list[Plan],
        task_instruction: str,
        initial_state: str,
        history_context: str = "",
    ) -> list[Plan]:
        """Simulate all plans concurrently.

        Each plan runs its own multi-turn conversation in parallel.
        """
        tasks = [
            self.simulate_plan(plan, task_instruction, initial_state, history_context)
            for plan in plans
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        simulated = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("[TAPE Simulator] Plan %d raised exception: %s", i, result)
            elif result is not None:
                simulated.append(result)

        logger.info("[TAPE Simulator] Simulated %d/%d plans", len(simulated), len(plans))
        return simulated
