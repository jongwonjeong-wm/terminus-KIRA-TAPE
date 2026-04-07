"""PlanGraphBuilder — merges M simulated plans into a DAG via LLM-based state abstraction.

Per TAPE (Section 3.1): merging function f_θ : Ŝ → V merges states that represent
the same observation and task progress into a single node.
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
from terminus_kira.tape.types import (
    Plan,
    PlanEdge,
    PlanGraphData,
    PlanNode,
)

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent.parent / "prompt-templates" / "tape-state-merge.txt"
)

# Tool description strings (mirroring terminus_kira.py pattern)
_SUBMIT_GROUPS_DESC = "Submit groups of state IDs that should be merged into single nodes."

_GROUPS_DESC = (
    "List of groups. Each group is a list of state_ids that represent "
    "the same task progress and should be merged into one node. "
    "Every state_id must appear in exactly one group."
)

_GROUP_ITEMS_DESC = "List of state_ids to merge into one node."


def _build_merge_tools(state_ids: list[str]) -> list[dict]:
    """Build tool definition with state_ids constrained via enum."""
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_state_groups",
                "description": _SUBMIT_GROUPS_DESC,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "groups": {
                            "type": "array",
                            "description": _GROUPS_DESC,
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": state_ids,
                                },
                                "description": _GROUP_ITEMS_DESC,
                            },
                        },
                    },
                    "required": ["groups"],
                },
            },
        },
    ]


class PlanGraphBuilder:
    """Merges multiple simulated plans into a single plan DAG.

    Uses an LLM to determine which predicted states across plans represent
    the same task progress (state abstraction), then merges them into
    shared nodes. This implements the merging function f_θ from TAPE Section 3.1.
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
    async def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict | None:
        """Call LLM with tool calling and return parsed tool arguments."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "submit_state_groups"}},
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

    async def _get_state_merge_groups(
        self,
        task_instruction: str,
        state_entries: list[dict],
    ) -> dict[str, str]:
        """Call LLM to group states by task progress + state abstraction.

        Args:
            task_instruction: The task being solved.
            state_entries: List of {"state_id": ..., "plan_id": ..., "step": ...,
                           "description": ..., "predicted_state": ...}

        Returns:
            Mapping from state_id -> canonical_id (the first state_id in its group).
        """
        # Build dynamic tools with enum-constrained state_ids
        state_ids = [e["state_id"] for e in state_entries]
        tools = _build_merge_tools(state_ids)

        # Build user message with all states
        states_text = json.dumps(state_entries, indent=2)
        user_message = (
            f"# Task\n{task_instruction}\n\n"
            f"# Predicted States\n{states_text}"
        )

        messages = add_anthropic_caching(
            [
                {"role": "system", "content": self._prompt_template},
                {"role": "user", "content": user_message},
            ],
            self.model_name,
        )

        result = await self._call_llm(messages, tools)
        if result is None:
            logger.warning("[TAPE Graph] LLM state merge returned no tool call")
            return {}

        # Parse groups into a mapping: state_id -> canonical_id
        valid_ids = set(e["state_id"] for e in state_entries)
        merge_map: dict[str, str] = {}
        for group in result.get("groups", []):
            # Filter out any state_ids the LLM hallucinated
            valid_group = [sid for sid in group if sid in valid_ids]
            if not valid_group:
                continue
            canonical = valid_group[0]
            for state_id in valid_group:
                merge_map[state_id] = canonical
        return merge_map

    async def build_graph(
        self,
        plans: list[Plan],
        initial_state: str,
        task_instruction: str,
    ) -> PlanGraphData:
        """Merge plans into a DAG using LLM-based state abstraction.

        Algorithm:
        1. Collect all predicted_states from all plans with unique IDs.
        2. Call LLM to group states by task progress + state abstraction.
        3. Build DAG: merged states become shared nodes, subgoals become edges.
        """
        graph = PlanGraphData()

        # Create start node
        start_id = "start"
        graph.nodes[start_id] = PlanNode(
            node_id=start_id,
            state_description=initial_state,
            is_start=True,
        )
        graph.start_node = start_id

        # Step 1: Collect all states with IDs (state_p{m}_s{k})
        state_entries = []
        state_id_to_info: dict[str, dict] = {}  # state_id -> {plan_id, step_idx, subgoal}

        for plan in plans:
            for i, subgoal in enumerate(plan.subgoals):
                state_id = f"state_p{plan.plan_id}_s{i}"
                state_entries.append({
                    "state_id": state_id,
                    "plan_id": plan.plan_id,
                    "step": i,
                    "subgoal_description": subgoal.description,
                    "predicted_state": subgoal.predicted_state,
                })
                state_id_to_info[state_id] = {
                    "plan_id": plan.plan_id,
                    "step_idx": i,
                    "subgoal": subgoal,
                    "is_last": i == len(plan.subgoals) - 1,
                }

        # Step 2: LLM-based state grouping
        try:
            merge_map = await self._get_state_merge_groups(
                task_instruction, state_entries,
            )
        except Exception as e:
            logger.warning("[TAPE Graph] LLM merge failed, falling back to no merging: %s", e)
            merge_map = {s["state_id"]: s["state_id"] for s in state_entries}

        # Fallback: any state_id not in merge_map maps to itself
        for entry in state_entries:
            sid = entry["state_id"]
            if sid not in merge_map:
                merge_map[sid] = sid

        # Step 3: Build DAG
        # Create nodes with sequential IDs: node_0, node_1, ...
        # (node IDs are plan-agnostic since they represent merged states)
        canonical_to_node_id: dict[str, str] = {}
        node_counter = 0

        for state_id, canonical_id in merge_map.items():
            # Guard: if LLM returned a canonical_id not in state_id_to_info,
            # fall back to using state_id itself as canonical
            if canonical_id not in state_id_to_info:
                logger.warning(
                    "[TAPE Graph] canonical_id %s not found, falling back to %s",
                    canonical_id, state_id,
                )
                canonical_id = state_id
                merge_map[state_id] = state_id

            if canonical_id not in canonical_to_node_id:
                info = state_id_to_info[canonical_id]
                node_id = f"node_{node_counter}"
                node_counter += 1
                is_goal = info["is_last"]
                canonical_to_node_id[canonical_id] = node_id
                graph.nodes[node_id] = PlanNode(
                    node_id=node_id,
                    state_description=info["subgoal"].predicted_state,
                    is_goal=is_goal,
                )
                if is_goal:
                    graph.goal_nodes.append(node_id)
            else:
                # If any merged state is a goal state, mark the node as goal
                info = state_id_to_info[state_id]
                node_id = canonical_to_node_id[canonical_id]
                if info["is_last"] and not graph.nodes[node_id].is_goal:
                    graph.nodes[node_id].is_goal = True
                    graph.goal_nodes.append(node_id)

        # Create edges
        edge_counter = 0
        seen_edges: set[tuple[str, str, str]] = set()  # (from_node, to_node, tool_type)

        for plan in plans:
            prev_node_id = start_id

            for i, subgoal in enumerate(plan.subgoals):
                state_id = f"state_p{plan.plan_id}_s{i}"
                canonical_id = merge_map[state_id]
                target_node_id = canonical_to_node_id[canonical_id]

                # Deduplicate edges with same from/to/tool
                edge_key = (prev_node_id, target_node_id, subgoal.predicted_tool.value)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edge_id = f"edge_{edge_counter}"
                    edge_counter += 1
                    graph.edges[edge_id] = PlanEdge(
                        edge_id=edge_id,
                        from_node=prev_node_id,
                        to_node=target_node_id,
                        subgoal=subgoal,
                        reward=subgoal.reward,
                        cost=subgoal.estimated_duration,
                    )

                prev_node_id = target_node_id

        logger.info(
            "[TAPE Graph] Built graph: %d nodes, %d edges, %d goal nodes",
            len(graph.nodes), len(graph.edges), len(graph.goal_nodes),
        )
        return graph
