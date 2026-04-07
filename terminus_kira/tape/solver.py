"""ILPSolver — PuLP-based optimal path selection on the plan graph.

Implements the time-expanded ILP formulation from the TAPE paper (Section 3.2),
adapted for the plan graph structure used in Terminus-Kira.

Objective: maximize Σ log(p_e) * x_e  (= maximize path success probability)
           with small cost penalty λ for tie-breaking (prefer faster paths).
"""

import logging

from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpMaximize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

from terminus_kira.tape.types import PlanEdge, PlanGraphData, SelectedPath

logger = logging.getLogger(__name__)


class ILPSolver:
    """Select the optimal path through the plan graph via Integer Linear Programming.

    Formulation:
        Variables: x_e in {0, 1} for each edge e
        Objective: maximize sum_e (log(p_e) - λ * cost_e) * x_e
                   where p_e = subgoal success probability (from simulator)
        Constraints:
            1. Flow out of start = 1
            2. Flow into virtual sink = 1
            3. Flow conservation at intermediate nodes
            4. At most one incoming edge per node (simple path)
            5. Budget: sum_e (cost_e * x_e) <= time_budget (if set)
    """

    def __init__(
        self,
        time_budget: float | None = None,
        cost_penalty: float = 0.001,
        solver_timeout: int = 10,
    ):
        self.time_budget = time_budget
        self.cost_penalty = cost_penalty
        self.solver_timeout = solver_timeout

    def solve(self, graph: PlanGraphData) -> SelectedPath | None:
        """Solve for the optimal path through the plan graph.

        Returns SelectedPath or None if infeasible.
        """
        if not graph.edges:
            logger.warning("[TAPE Solver] Empty graph, no edges to solve")
            return None

        if not graph.goal_nodes:
            logger.warning("[TAPE Solver] No goal nodes in graph")
            return None

        # Add virtual sink node with zero-cost edges from all goal nodes
        sink_id = "__sink__"
        sink_edges: dict[str, PlanEdge] = {}
        for i, goal_id in enumerate(graph.goal_nodes):
            sink_edge_id = f"__sink_edge_{i}__"
            # Create a dummy subgoal for the sink edge
            sink_edges[sink_edge_id] = PlanEdge(
                edge_id=sink_edge_id,
                from_node=goal_id,
                to_node=sink_id,
                subgoal=None,  # type: ignore — sink edges aren't real actions
                reward=0.0,
                cost=0.0,
            )

        all_edges = {**graph.edges, **sink_edges}

        # Collect all node IDs
        all_nodes = set(graph.nodes.keys()) | {sink_id}

        # Build adjacency structures
        outgoing: dict[str, list[str]] = {n: [] for n in all_nodes}
        incoming: dict[str, list[str]] = {n: [] for n in all_nodes}
        for eid, edge in all_edges.items():
            outgoing[edge.from_node].append(eid)
            incoming[edge.to_node].append(eid)

        # Create ILP problem
        prob = LpProblem("TAPE_PathSelection", LpMaximize)

        # Decision variables: x_e for each edge
        x = {
            eid: LpVariable(f"x_{eid}", cat=LpBinary)
            for eid in all_edges
        }

        # Objective: maximize total reward with cost penalty
        # reward: 1=goal, 0=normal, negative=risky (from simulator)
        # -λ * cost: prefer faster paths as tie-breaker
        prob += lpSum(
            (all_edges[eid].reward - self.cost_penalty * all_edges[eid].cost) * x[eid]
            for eid in all_edges
        ), "TotalReward"

        # Constraint 1: Flow out of start = 1
        prob += (
            lpSum(x[eid] for eid in outgoing.get(graph.start_node, [])) == 1,
            "FlowOutOfStart",
        )

        # Constraint 2: Flow into sink = 1
        prob += (
            lpSum(x[eid] for eid in incoming.get(sink_id, [])) == 1,
            "FlowIntoSink",
        )

        # Constraint 3: Flow conservation at intermediate nodes
        for node_id in all_nodes:
            if node_id == graph.start_node or node_id == sink_id:
                continue
            in_edges = incoming.get(node_id, [])
            out_edges = outgoing.get(node_id, [])
            if in_edges or out_edges:
                prob += (
                    lpSum(x[eid] for eid in in_edges)
                    == lpSum(x[eid] for eid in out_edges),
                    f"FlowConservation_{node_id}",
                )

        # Constraint 4: At most one incoming edge per node (simple path)
        for node_id in all_nodes:
            if node_id == graph.start_node:
                continue
            in_edges = incoming.get(node_id, [])
            if len(in_edges) > 1:
                prob += (
                    lpSum(x[eid] for eid in in_edges) <= 1,
                    f"SimplePath_{node_id}",
                )

        # Constraint 5: Budget constraint (optional)
        if self.time_budget is not None:
            prob += (
                lpSum(
                    all_edges[eid].cost * x[eid]
                    for eid in all_edges
                )
                <= self.time_budget,
                "BudgetConstraint",
            )

        # Solve
        solver = PULP_CBC_CMD(msg=0, timeLimit=self.solver_timeout)
        try:
            prob.solve(solver)
        except Exception as e:
            logger.warning(f"[TAPE Solver] Solver error: {e}")
            return None

        # Check feasibility
        if prob.status != 1:  # 1 = Optimal
            logger.warning(
                f"[TAPE Solver] No optimal solution found (status={prob.status})"
            )
            return None

        # Extract selected edges (excluding sink edges), in path order
        selected_edge_ids = {
            eid
            for eid, var in x.items()
            if value(var) is not None and value(var) > 0.5  # type: ignore
        }

        # Remove sink edges
        selected_edge_ids -= set(sink_edges.keys())

        # Order edges by walking from start
        ordered_edges = self._order_edges(
            graph, selected_edge_ids,
        )

        if not ordered_edges:
            logger.warning("[TAPE Solver] Could not reconstruct path from solution")
            return None

        total_reward = sum(e.reward for e in ordered_edges)
        total_cost = sum(e.cost for e in ordered_edges)

        logger.info(
            f"[TAPE Solver] Selected path: {len(ordered_edges)} steps, "
            f"reward={total_reward:.3f}, cost={total_cost:.1f}s"
        )

        return SelectedPath(
            edges=ordered_edges,
            total_reward=total_reward,
            total_cost=total_cost,
        )

    def _order_edges(
        self,
        graph: PlanGraphData,
        selected_edge_ids: set[str],
    ) -> list[PlanEdge]:
        """Order selected edges by walking the path from start node."""
        # Build from_node -> edge lookup for selected edges
        from_lookup: dict[str, PlanEdge] = {}
        for eid in selected_edge_ids:
            edge = graph.edges[eid]
            from_lookup[edge.from_node] = edge

        ordered = []
        current = graph.start_node
        visited = set()

        while current in from_lookup and current not in visited:
            visited.add(current)
            edge = from_lookup[current]
            ordered.append(edge)
            current = edge.to_node

        return ordered
