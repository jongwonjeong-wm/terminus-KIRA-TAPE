"""Shared data types for TAPE (Tool-guided Adaptive Planning with constrained Execution)."""

from dataclasses import dataclass, field
from enum import Enum


class ToolType(Enum):
    """Tool types available in TerminusKira."""

    EXECUTE_COMMANDS = "execute_commands"
    TASK_COMPLETE = "task_complete"
    IMAGE_READ = "image_read"


@dataclass
class Subgoal:
    """A single step in a high-level plan.

    Represents an edge (action) in the plan graph: tool + subgoal description.
    The predicted_state, success_probability, and estimated_duration are
    populated by the Simulator after initial plan generation.
    """

    id: str  # e.g., "subgoal_p0_s2"
    description: str  # natural language subgoal
    predicted_tool: ToolType  # which tool the simulator expects

    # Populated by Simulator
    predicted_state: str = ""  # what terminal should look like after
    success_probability: float = 0.0  # 0.0–1.0
    estimated_duration: float = 1.0  # seconds (cost for ILP)


@dataclass
class Plan:
    """A complete candidate plan — ordered sequence of subgoals."""

    plan_id: int
    subgoals: list[Subgoal] = field(default_factory=list)
    total_success_prob: float = 0.0  # product of step probs
    total_estimated_duration: float = 0.0  # sum of step durations


@dataclass
class PlanNode:
    """Node in the plan DAG. Represents a state."""

    node_id: str
    state_description: str
    is_start: bool = False
    is_goal: bool = False


@dataclass
class PlanEdge:
    """Edge in the plan DAG. Represents a subgoal/action transition."""

    edge_id: str
    from_node: str  # node_id
    to_node: str  # node_id
    subgoal: Subgoal
    reward: float  # success_probability
    cost: float  # estimated_duration


@dataclass
class PlanGraphData:
    """The merged DAG of all candidate plans."""

    nodes: dict[str, PlanNode] = field(default_factory=dict)
    edges: dict[str, PlanEdge] = field(default_factory=dict)
    start_node: str = ""
    goal_nodes: list[str] = field(default_factory=list)


@dataclass
class SelectedPath:
    """Result from ILP solver — the optimal path through the DAG."""

    edges: list[PlanEdge] = field(default_factory=list)
    total_reward: float = 0.0
    total_cost: float = 0.0
    current_step_idx: int = 0  # tracks execution progress

    @property
    def current_subgoal(self) -> Subgoal | None:
        """Get the current subgoal to execute."""
        if self.current_step_idx < len(self.edges):
            return self.edges[self.current_step_idx].subgoal
        return None

    @property
    def is_complete(self) -> bool:
        """Whether all subgoals have been executed."""
        return self.current_step_idx >= len(self.edges)

    @property
    def total_steps(self) -> int:
        return len(self.edges)
