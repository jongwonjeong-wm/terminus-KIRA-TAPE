"""Quick sanity check for TAPE components.

Run: module load conda && conda activate tb && python tests/test_tape.py
     python tests/test_tape.py --verbose   # dump all LLM input/output to tests/tape_llm_log.jsonl
Optional: set ANTHROPIC_API_KEY or OPENAI_API_KEY to test LLM-dependent components.
"""

import asyncio
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── LLM call logger (activated with --verbose) ──────────────────────
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv
_LOG_PATH = Path(__file__).parent / "tape_llm_log.jsonl"
_call_counter = 0


def _setup_llm_logger():
    """Install a litellm callback that logs every LLM call to a JSONL file."""
    import litellm

    if _LOG_PATH.exists():
        _LOG_PATH.unlink()

    class TAPELogger(litellm.integrations.custom_logger.CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._do_log(kwargs, response_obj, start_time, end_time)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._do_log(kwargs, response_obj, start_time, end_time)

        def _do_log(self, kwargs, response_obj, start_time, end_time):
            global _call_counter
            _call_counter += 1
            call_id = _call_counter

            messages = kwargs.get("messages", [])
            tools = kwargs.get("tools", [])
            tool_choice = kwargs.get("tool_choice", None)
            model = kwargs.get("model", "")

            # Extract response
            resp_content = ""
            resp_tool_calls = []
            try:
                msg = response_obj.choices[0].message
                resp_content = msg.content or ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                pass
                        resp_tool_calls.append({
                            "id": tc.id,
                            "function": tc.function.name,
                            "arguments": args,
                        })
            except Exception:
                pass

            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Usage
            usage = {}
            try:
                u = response_obj.usage
                usage = {
                    "input_tokens": u.prompt_tokens,
                    "output_tokens": u.completion_tokens,
                }
            except Exception:
                pass

            entry = {
                "call_id": call_id,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "duration_ms": duration_ms,
                "usage": usage,
                "input": {
                    "messages": messages,
                    "tools": [t["function"]["name"] for t in tools] if tools else [],
                    "tool_choice": tool_choice,
                },
                "output": {
                    "content": resp_content,
                    "tool_calls": resp_tool_calls,
                },
            }

            # Write to file
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Print summary to terminal
            tool_names = [tc["function"] for tc in resp_tool_calls]
            args_preview = ""
            if resp_tool_calls:
                args_raw = resp_tool_calls[0]["arguments"]
                args_str = json.dumps(args_raw, ensure_ascii=False) if isinstance(args_raw, dict) else str(args_raw)
                args_preview = args_str[:200] + "..." if len(args_str) > 200 else args_str

            print(f"\n  📡 LLM Call #{call_id} [{model}] {duration_ms}ms")
            print(f"     Input: {len(messages)} messages, tool_choice={tool_choice}")
            for i, m in enumerate(messages):
                role = m.get("role", "?")
                content = m.get("content", "")
                if isinstance(content, str):
                    preview = content[:150].replace("\n", "\\n")
                    if len(content) > 150:
                        preview += "..."
                else:
                    preview = str(content)[:150]
                print(f"     [{i}] {role}: {preview}")
            print(f"     Output: tools={tool_names} {usage}")
            if args_preview:
                print(f"     Args: {args_preview}")

    litellm.callbacks = [TAPELogger()]
    print(f"  LLM logging enabled → {_LOG_PATH}\n")

from terminus_kira.tape.types import (
    Plan, PlanEdge, PlanGraphData, PlanNode, SelectedPath, Subgoal, ToolType,
)
from terminus_kira.tape.graph import PlanGraphBuilder
from terminus_kira.tape.solver import ILPSolver


def make_subgoal(plan_id, step, desc, state, prob=0.9, dur=2.0, tool=ToolType.EXECUTE_COMMANDS):
    return Subgoal(
        id=f"subgoal_p{plan_id}_s{step}",
        description=desc,
        predicted_tool=tool,
        predicted_state=state,
        success_probability=prob,
        estimated_duration=dur,
    )


def make_test_plans():
    """Create 3 synthetic plans with some overlap."""
    plan0 = Plan(plan_id=0, subgoals=[
        make_subgoal(0, 0, "read the source file main.py", "file contents visible with solve() function", 0.95, 1.0),
        make_subgoal(0, 1, "identify the bug in solve function", "found bug on line 42: off-by-one error", 0.8, 2.0),
        make_subgoal(0, 2, "fix the bug with sed", "file modified successfully", 0.7, 1.5),
        make_subgoal(0, 3, "run the tests", "all tests passed", 0.85, 3.0),
        make_subgoal(0, 4, "mark task complete", "task done", 1.0, 0.1, ToolType.TASK_COMPLETE),
    ])

    plan1 = Plan(plan_id=1, subgoals=[
        make_subgoal(1, 0, "read source file main.py", "file contents visible with solve() function", 0.95, 1.0),
        make_subgoal(1, 1, "run the tests first to see failures", "test output shows 2 failures in test_solve", 0.9, 3.0),
        make_subgoal(1, 2, "identify bug from test output", "found bug: off-by-one in solve()", 0.75, 2.0),
        make_subgoal(1, 3, "fix the bug with python script", "file modified successfully", 0.65, 2.0),
        make_subgoal(1, 4, "rerun the tests", "all tests passed", 0.85, 3.0),
        make_subgoal(1, 5, "mark task complete", "task done", 1.0, 0.1, ToolType.TASK_COMPLETE),
    ])

    plan2 = Plan(plan_id=2, subgoals=[
        make_subgoal(2, 0, "read the source file main.py", "file contents visible with solve() function", 0.95, 1.0),
        make_subgoal(2, 1, "grep for error pattern in source", "found potential bug locations", 0.85, 1.0),
        make_subgoal(2, 2, "fix the bug with sed command", "file modified successfully", 0.7, 1.5),
        make_subgoal(2, 3, "run tests to verify", "all tests passed", 0.85, 3.0),
        make_subgoal(2, 4, "mark task complete", "task done", 1.0, 0.1, ToolType.TASK_COMPLETE),
    ])

    for p in [plan0, plan1, plan2]:
        p.total_success_prob = math.prod(sg.success_probability for sg in p.subgoals)
        p.total_estimated_duration = sum(sg.estimated_duration for sg in p.subgoals)

    return [plan0, plan1, plan2]


def _mock_merge_response():
    """Simulate LLM state merge response for test plans.

    Groups states that represent the same task progress:
    - p0_s0, p1_s0, p2_s0 -> all read main.py (same state)
    - p0_s2, p1_s3, p2_s2 -> all fixed the bug (same state)
    - p0_s3, p1_s4, p2_s3 -> all tests passed (same state)
    - p0_s4, p1_s5, p2_s4 -> all task complete (same state)
    - p0_s1, p1_s1, p1_s2, p2_s1 -> different approaches, NOT merged
    """
    return {
        "groups": [
            ["state_p0_s0", "state_p1_s0", "state_p2_s0"],  # read source file
            ["state_p0_s1"],                                  # identify bug directly
            ["state_p1_s1"],                                  # run tests first
            ["state_p1_s2"],                                  # identify bug from test output
            ["state_p2_s1"],                                  # grep for error pattern
            ["state_p0_s2", "state_p1_s3", "state_p2_s2"],   # file modified (bug fixed)
            ["state_p0_s3", "state_p1_s4", "state_p2_s3"],   # all tests passed
            ["state_p0_s4", "state_p1_s5", "state_p2_s4"],   # task done
        ]
    }


async def test_graph_builder():
    """Test graph construction with mocked LLM state merging."""
    print("=" * 60)
    print("TEST: PlanGraphBuilder (mocked LLM)")
    print("=" * 60)

    plans = make_test_plans()

    builder = PlanGraphBuilder(model_name="mock-model")

    # Mock the LLM call to return our predefined merge groups
    with patch.object(builder, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _mock_merge_response()

        graph = await builder.build_graph(
            plans,
            initial_state="$ ls\nmain.py test_main.py",
            task_instruction="Fix the bug in main.py",
        )

    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Goal nodes: {len(graph.goal_nodes)}")
    print(f"  Start node: {graph.start_node}")

    print("\n  Nodes:")
    for nid, node in graph.nodes.items():
        tag = ""
        if node.is_start:
            tag = " [START]"
        if node.is_goal:
            tag = " [GOAL]"
        print(f"    {nid}{tag}: {node.state_description[:60]}...")

    print("\n  Edges:")
    for eid, edge in graph.edges.items():
        print(
            f"    {eid}: {edge.from_node} -> {edge.to_node} "
            f"[r={edge.reward:.2f}, c={edge.cost:.1f}] "
            f"{edge.subgoal.description[:40]}..."
        )

    # Verify graph invariants
    assert graph.start_node in graph.nodes, "Start node missing"
    assert len(graph.goal_nodes) > 0, "No goal nodes"
    for gid in graph.goal_nodes:
        assert gid in graph.nodes, f"Goal node {gid} missing"
    for eid, edge in graph.edges.items():
        assert edge.from_node in graph.nodes, f"Edge {eid} from_node missing"
        assert edge.to_node in graph.nodes, f"Edge {eid} to_node missing"

    # Verify merging happened: should have fewer nodes than total subgoals
    total_subgoals = sum(len(p.subgoals) for p in plans)
    print(f"\n  Total subgoals across plans: {total_subgoals}")
    # 8 groups + 1 start = 9 nodes (merged from 16 subgoals)
    assert len(graph.nodes) < total_subgoals, "No merging happened"
    print(f"  Merged to {len(graph.nodes)} nodes (including start)")

    print("\n  ✓ Graph invariants OK")
    return graph


def test_solver(graph):
    """Test ILP solver on the constructed graph."""
    print("\n" + "=" * 60)
    print("TEST: ILPSolver")
    print("=" * 60)

    # Test 1: Unconstrained
    solver = ILPSolver(time_budget=None)
    path = solver.solve(graph)

    if path is None:
        print("  ✗ Solver returned None (infeasible)")
        return None

    print(f"  Selected path: {path.total_steps} steps")
    print(f"  Total reward: {path.total_reward:.3f}")
    print(f"  Total cost: {path.total_cost:.1f}s")
    print("\n  Path:")
    for i, edge in enumerate(path.edges):
        sg = edge.subgoal
        print(
            f"    Step {i + 1}: [{sg.predicted_tool.value}] {sg.description} "
            f"(r={edge.reward:.2f}, c={edge.cost:.1f}s)"
        )

    assert path.total_steps > 0, "Empty path"
    assert path.edges[0].from_node == graph.start_node, "Path doesn't start at start node"
    assert path.edges[-1].to_node in graph.goal_nodes, "Path doesn't end at goal"
    print("\n  ✓ Unconstrained solve OK")

    # Test 2: With tight budget
    print("\n  --- Budget constrained (5s budget) ---")
    solver_budget = ILPSolver(time_budget=5.0)
    path_budget = solver_budget.solve(graph)
    if path_budget is None:
        print("  ✓ Correctly returned infeasible for tight budget")
    else:
        print(f"  Path cost: {path_budget.total_cost:.1f}s (budget: 5.0s)")
        assert path_budget.total_cost <= 5.0 + 0.01, "Budget violated"
        print("  ✓ Budget constraint satisfied")

    return path


def test_selected_path():
    """Test SelectedPath tracking."""
    print("\n" + "=" * 60)
    print("TEST: SelectedPath")
    print("=" * 60)

    sg = make_subgoal(0, 0, "test", "state", 0.9, 1.0)
    edge = PlanEdge("e0", "start", "n0", sg, 0.9, 1.0)
    path = SelectedPath(edges=[edge, edge], total_reward=1.8, total_cost=2.0)

    assert not path.is_complete
    assert path.current_subgoal is not None
    assert path.current_step_idx == 0

    path.current_step_idx = 1
    assert not path.is_complete

    path.current_step_idx = 2
    assert path.is_complete
    assert path.current_subgoal is None

    print("  ✓ SelectedPath tracking OK")


async def test_planner_llm():
    """Test planner with real LLM (requires API key)."""
    print("\n" + "=" * 60)
    print("TEST: TAPEPlanner (LLM)")
    print("=" * 60)

    from terminus_kira.tape.planner import TAPEPlanner

    model = os.environ.get("TAPE_TEST_MODEL", "anthropic/claude-haiku-4-5-20251001")
    planner = TAPEPlanner(model_name=model, M=2, temperature=0.8)

    plans = await planner.generate_plans(
        task_instruction="Create a file called hello.txt containing 'Hello World'",
        current_terminal_state="$ pwd\n/home/user\n$ ",
    )

    print(f"  Generated {len(plans)} plans")
    for plan in plans:
        print(f"\n  Plan {plan.plan_id} ({len(plan.subgoals)} subgoals):")
        for sg in plan.subgoals:
            print(f"    - [{sg.predicted_tool.value}] {sg.description}")

    assert len(plans) > 0, "No plans generated"
    print("\n  ✓ Planner OK")
    return plans


async def test_simulator_llm(plans):
    """Test simulator with real LLM."""
    print("\n" + "=" * 60)
    print("TEST: TAPESimulator (LLM)")
    print("=" * 60)

    from terminus_kira.tape.simulator import TAPESimulator

    model = os.environ.get("TAPE_TEST_MODEL", "anthropic/claude-haiku-4-5-20251001")
    sim = TAPESimulator(model_name=model)

    simulated = await sim.simulate_all(
        plans,
        task_instruction="Create a file called hello.txt containing 'Hello World'",
        initial_state="$ pwd\n/home/user\n$ ",
    )

    print(f"  Simulated {len(simulated)} plans")
    for plan in simulated:
        print(f"\n  Plan {plan.plan_id} (prob={plan.total_success_prob:.3f}, dur={plan.total_estimated_duration:.1f}s):")
        for sg in plan.subgoals:
            print(f"    - {sg.description}")
            print(f"      state: {sg.predicted_state[:80]}...")
            print(f"      prob={sg.success_probability:.2f}, dur={sg.estimated_duration:.1f}s")

    assert len(simulated) > 0, "No simulations succeeded"
    print("\n  ✓ Simulator OK")
    return simulated


async def test_mismatch_checker_llm():
    """Test mismatch checker with real LLM — both match and mismatch cases."""
    print("\n" + "=" * 60)
    print("TEST: MismatchChecker (LLM)")
    print("=" * 60)

    from terminus_kira.tape.mismatch import MismatchChecker, SubgoalStatus

    model = os.environ.get("TAPE_TEST_MODEL", "anthropic/claude-haiku-4-5-20251001")
    checker = MismatchChecker(model_name=model)

    task = "Create a file called hello.txt containing 'Hello World'"

    # --- Case 1: MATCH (subgoal completed, state matches) ---
    print("\n  --- Case 1: Completed + Match ---")
    status1 = await checker.check(
        terminal_history="$ echo 'Hello World' > hello.txt\n$",
        current_terminal_output="$ cat hello.txt\nHello World\n$",
        subgoal_description="Verify that hello.txt contains 'Hello World'",
        predicted_state="$ cat hello.txt\nHello World",
        task_instruction=task,
    )
    print(f"  Status: {status1}")
    assert status1 == SubgoalStatus.COMPLETED_MATCH, f"Expected COMPLETED_MATCH, got {status1}"
    print("  ✓ Match case OK")

    # --- Case 2: MISMATCH (subgoal completed, but state diverged) ---
    print("\n  --- Case 2: Completed + Mismatch ---")
    status2 = await checker.check(
        terminal_history=(
            "$ echo 'Hello World' > hello.txt\n"
            "bash: permission denied: hello.txt\n$\n"
            "$ ls hello.txt\n"
            "ls: cannot access 'hello.txt': No such file or directory\n$"
        ),
        current_terminal_output="$ ls hello.txt\nls: cannot access 'hello.txt': No such file or directory\n$",
        subgoal_description="Create hello.txt with content 'Hello World' using echo",
        predicted_state="$ echo 'Hello World' > hello.txt\n$ (file created successfully)",
        task_instruction=task,
    )
    print(f"  Status: {status2}")
    assert status2 == SubgoalStatus.COMPLETED_MISMATCH, f"Expected COMPLETED_MISMATCH, got {status2}"
    print("  ✓ Mismatch case OK")

    # --- Case 3: IN_PROGRESS (subgoal not done yet) ---
    print("\n  --- Case 3: In Progress ---")
    status3 = await checker.check(
        terminal_history="",
        current_terminal_output="$ cd /home/user/project\n$",
        subgoal_description="Read the source file main.py and identify the buggy function",
        predicted_state="file contents visible with solve() function, bug on line 42",
        task_instruction="Fix the bug in main.py",
    )
    print(f"  Status: {status3}")
    assert status3 == SubgoalStatus.IN_PROGRESS, f"Expected IN_PROGRESS, got {status3}"
    print("  ✓ In-progress case OK")

    print("\n  ✓ MismatchChecker OK")


async def test_full_pipeline_llm():
    """Test full pipeline: plan -> simulate -> graph -> solve."""
    print("\n" + "=" * 60)
    print("TEST: Full Pipeline (LLM)")
    print("=" * 60)

    model = os.environ.get("TAPE_TEST_MODEL", "anthropic/claude-haiku-4-5-20251001")

    plans = await test_planner_llm()
    simulated = await test_simulator_llm(plans)

    builder = PlanGraphBuilder(model_name=model)
    graph = await builder.build_graph(
        simulated,
        "$ pwd\n/home/user\n$ ",
        task_instruction="Create a file called hello.txt containing 'Hello World'",
    )
    print(f"\n  Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    solver = ILPSolver()
    path = solver.solve(graph)
    if path:
        print(f"  Selected path: {path.total_steps} steps, reward={path.total_reward:.3f}")
        for i, e in enumerate(path.edges):
            print(f"    Step {i+1}: {e.subgoal.description}")
        print("\n  ✓ Full pipeline OK")
    else:
        print("  ✗ Solver returned None")


async def test_execution_loop():
    """Test TerminusKiraTAPE execution loop with mocked terminal + mocked LLM.

    Simulates a 3-subgoal task:
      Subgoal 1: Create hello.txt       → execute → COMPLETED_MATCH → advance
      Subgoal 2: Verify file content     → execute → IN_PROGRESS → execute again → COMPLETED_MATCH → advance
      Subgoal 3: Mark task complete       → task_complete → done

    This tests:
      - TAPE planning pipeline → constrained execution loop
      - Subgoal injection on first attempt (tool_choice + tool description + observation)
      - IN_PROGRESS: no injection on continuation episode
      - COMPLETED_MATCH → advance to next subgoal
      - Task completion handling
    """
    print("\n" + "=" * 60)
    print("TEST: TerminusKiraTAPE Execution Loop (mocked)")
    print("=" * 60)

    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

    from terminus_kira.terminus_kira_tape import TerminusKiraTAPE
    from terminus_kira.terminus_kira import ToolCallResponse, Command
    from terminus_kira.tape.types import (
        Plan, PlanEdge, SelectedPath, Subgoal, ToolType,
    )
    from terminus_kira.tape.mismatch import SubgoalStatus
    from harbor.llms.base import LLMResponse
    from harbor.models.agent.context import AgentContext
    from harbor.models.metric import UsageInfo

    # ── Build a predefined 3-step SelectedPath ────────────────────────
    sg1 = Subgoal(
        id="subgoal_p0_s0",
        description="Create hello.txt with content 'Hello World'",
        predicted_tool=ToolType.EXECUTE_COMMANDS,
        predicted_state="$ echo 'Hello World' > hello.txt\n$",
        success_probability=0.95,
        estimated_duration=1.0,
    )
    sg2 = Subgoal(
        id="subgoal_p0_s1",
        description="Verify hello.txt contains correct content",
        predicted_tool=ToolType.EXECUTE_COMMANDS,
        predicted_state="$ cat hello.txt\nHello World\n$",
        success_probability=0.9,
        estimated_duration=1.0,
    )
    sg3 = Subgoal(
        id="subgoal_p0_s2",
        description="Mark the task as complete",
        predicted_tool=ToolType.TASK_COMPLETE,
        predicted_state="task completed",
        success_probability=1.0,
        estimated_duration=0.1,
    )

    edges = [
        PlanEdge("e0", "start", "node_0", sg1, reward=0.95, cost=1.0),
        PlanEdge("e1", "node_0", "node_1", sg2, reward=0.9, cost=1.0),
        PlanEdge("e2", "node_1", "node_2", sg3, reward=1.0, cost=0.1),
    ]
    selected_path = SelectedPath(edges=edges, total_reward=0.855, total_cost=2.1)

    # ── Create agent with minimal mocking ─────────────────────────────
    tmp_dir = Path("/tmp/tape_test_exec")
    tmp_dir.mkdir(exist_ok=True)

    agent = TerminusKiraTAPE(
        logs_dir=tmp_dir,
        model_name="anthropic/claude-haiku-4-5-20251001",
        max_turns=10,
        enable_summarize=False,
        record_terminal_session=False,
        suppress_max_turns_warning=True,
    )

    # Mock session
    mock_session = MagicMock()
    mock_session.is_session_alive = AsyncMock(return_value=True)
    mock_session.capture_pane = AsyncMock(return_value="$ ")
    mock_session.get_incremental_output = AsyncMock(return_value="")
    mock_session.send_keys = AsyncMock()
    agent._session = mock_session

    # Mock context
    agent._context = AgentContext()

    # Mock chat with a real-ish object
    mock_chat = MagicMock()
    mock_chat.messages = []
    mock_chat.total_input_tokens = 0
    mock_chat.total_output_tokens = 0
    mock_chat.total_cache_tokens = 0
    mock_chat.total_cost = 0.0
    mock_chat.rollout_details = []
    mock_chat._messages = []

    # ── Track what happens during execution ───────────────────────────
    call_log = []  # (episode, subgoal_idx, was_injected, prompt_snippet)

    # Define episode-by-episode behavior:
    # Episode 0: sg1 injected, agent runs echo cmd → COMPLETED_MATCH
    # Episode 1: sg2 injected, agent runs cat cmd → IN_PROGRESS
    # Episode 2: sg2 continues (no injection), agent runs cat again → COMPLETED_MATCH
    # Episode 3: sg3 injected, agent calls task_complete → first confirmation
    # Episode 4: sg3 continues (no injection), agent calls task_complete → double confirm → done

    episode_counter = [0]

    terminal_outputs = [
        "$ echo 'Hello World' > hello.txt\n$",          # ep0: create file
        "$ cat hello.txt\n$",                            # ep1: cat but no output yet (in progress)
        "$ cat hello.txt\nHello World\n$",               # ep2: cat with output
        "$ ",                                             # ep3: task_complete (first)
        "$ ",                                             # ep4: task_complete (confirm)
    ]

    # _handle_llm_interaction returns:
    # (commands, is_task_complete, feedback, analysis, plan, llm_response, image_read)
    dummy_response = LLMResponse(content="executing", reasoning_content=None, usage=None)

    async def mock_handle_llm(chat, prompt, logging_paths, original_instruction, session):
        ep = episode_counter[0]
        was_injected = "[SUBGOAL]" in prompt
        subgoal_idx = selected_path.current_step_idx

        # Check if task_complete is blocked in tools
        task_complete_blocked = False
        if agent._tape_tools is not None:
            tool_names = [t["function"]["name"] for t in agent._tape_tools]
            task_complete_blocked = "task_complete" not in tool_names

        call_log.append({
            "episode": ep,
            "subgoal_idx": subgoal_idx,
            "was_injected": was_injected,
            "tool_choice_active": agent._tape_tool_choice is not None,
            "tools_modified": agent._tape_tools is not None,
            "task_complete_blocked": task_complete_blocked,
        })

        print(f"    Episode {ep}: subgoal_idx={subgoal_idx}, "
              f"injected={was_injected}, "
              f"tool_choice={agent._tape_tool_choice is not None}")

        if ep <= 2:
            # Execute commands episodes
            cmds = [Command(keystrokes="echo test\n", duration_sec=0.1)]
            return cmds, False, "", "analysis", "plan", dummy_response, None
        else:
            # Task complete episodes
            return [], True, "", "analysis", "plan", dummy_response, None

    async def mock_execute_commands(commands, session):
        ep = episode_counter[0]
        output = terminal_outputs[min(ep, len(terminal_outputs) - 1)]
        return False, output

    # Mismatch checker behavior per episode
    mismatch_results = [
        SubgoalStatus.COMPLETED_MATCH,      # ep0: sg1 done
        SubgoalStatus.IN_PROGRESS,           # ep1: sg2 still going
        SubgoalStatus.COMPLETED_MATCH,       # ep2: sg2 done
        # ep3 & ep4: task_complete, mismatch checker not called
    ]
    mismatch_call_counter = [0]

    async def mock_mismatch_check(**kwargs):
        idx = mismatch_call_counter[0]
        mismatch_call_counter[0] += 1
        result = mismatch_results[min(idx, len(mismatch_results) - 1)]
        print(f"    Mismatch check #{idx}: {result}")
        return result

    # ── Patch and run ─────────────────────────────────────────────────
    # Pre-set TAPE components so _init_tape_components doesn't overwrite them
    mock_mismatch = MagicMock()
    mock_mismatch.check = AsyncMock(side_effect=mock_mismatch_check)
    agent._tape_planner = MagicMock()
    agent._tape_simulator = MagicMock()
    agent._tape_graph_builder = MagicMock()
    agent._tape_solver = MagicMock()
    agent._tape_mismatch_checker = mock_mismatch

    with (
        patch.object(agent, "_init_tape_components", return_value=None),
        patch.object(agent, "_tape_plan_and_select", new_callable=AsyncMock) as mock_plan,
        patch.object(agent, "_handle_llm_interaction", side_effect=mock_handle_llm),
        patch.object(agent, "_execute_commands", side_effect=mock_execute_commands),
        patch.object(agent, "_setup_episode_logging", return_value=(None, None, None)),
        patch.object(agent, "_dump_trajectory", return_value=None),
        patch.object(agent, "_record_asciinema_marker", return_value=None),
    ):
        # Planning returns our predefined path
        mock_plan.return_value = selected_path

        # Track episode progression
        original_run = agent._run_agent_loop.__func__

        # We need to track episode counter in mock_handle_llm
        # Patch _build_observation to also increment counter
        original_build_obs = agent._build_observation

        def tracking_build_obs(is_task_complete, feedback, terminal_output):
            result = original_build_obs(is_task_complete, feedback, terminal_output)
            episode_counter[0] += 1
            return result

        agent._build_observation = tracking_build_obs

        result = await agent._run_agent_loop(
            initial_prompt="Create hello.txt containing 'Hello World'",
            chat=mock_chat,
            logging_dir=None,
            original_instruction="Create hello.txt containing 'Hello World'",
        )

    print(f"\n  Agent loop returned after {result} episodes")
    print(f"  Call log ({len(call_log)} episodes):")
    for entry in call_log:
        print(f"    ep={entry['episode']}: subgoal={entry['subgoal_idx']}, "
              f"injected={entry['was_injected']}, "
              f"tool_choice={entry['tool_choice_active']}, "
              f"tools_mod={entry['tools_modified']}")

    # ── Assertions ────────────────────────────────────────────────────
    assert len(call_log) == 5, f"Expected 5 episodes, got {len(call_log)}"

    # Episode 0: sg1, injected=True, tool_choice=True → COMPLETED_MATCH → advance
    assert call_log[0]["subgoal_idx"] == 0, "ep0 should be on subgoal 0"
    assert call_log[0]["was_injected"], "ep0 should have subgoal injected in prompt"
    assert call_log[0]["tool_choice_active"], "ep0 should have tool_choice set"
    assert call_log[0]["tools_modified"], "ep0 should have modified tools"

    # Episode 1: sg2 (advanced after match), injected=True (first attempt at sg2) → IN_PROGRESS
    assert call_log[1]["subgoal_idx"] == 1, "ep1 should be on subgoal 1"
    assert call_log[1]["was_injected"], "ep1 should have subgoal injected (first attempt at sg2)"
    assert call_log[1]["tool_choice_active"], "ep1 should have tool_choice set"

    # Episode 2: sg2 continues (IN_PROGRESS), injected=False → COMPLETED_MATCH → advance
    assert call_log[2]["subgoal_idx"] == 1, "ep2 should still be on subgoal 1"
    assert not call_log[2]["was_injected"], "ep2 should NOT have subgoal injected (continuation)"
    assert not call_log[2]["tool_choice_active"], "ep2 should NOT have tool_choice (continuation)"
    assert call_log[2]["task_complete_blocked"], "ep2 should block task_complete (not on last subgoal)"

    # Episode 3: sg3 (advanced), injected=True → task_complete first confirmation
    assert call_log[3]["subgoal_idx"] == 2, "ep3 should be on subgoal 2"
    assert call_log[3]["was_injected"], "ep3 should have subgoal injected (first attempt at sg3)"

    # Episode 4: sg3 double-confirm → tool_choice=None (free choice), LLM confirms → returns
    assert call_log[4]["subgoal_idx"] == 2, "ep4 should still be on subgoal 2"
    assert not call_log[4]["was_injected"], "ep4 should NOT have subgoal injected (double-confirm)"
    assert not call_log[4]["tool_choice_active"], "ep4 should NOT force tool_choice (free choice on double-confirm)"

    assert result == 5, f"Expected agent to return 5 episodes, got {result}"

    print(f"\n  Final path step: {selected_path.current_step_idx}")

    print("\n  ✓ Execution loop test PASSED")
    print("    - Subgoal injection works on first attempt only")
    print("    - IN_PROGRESS continues without injection")
    print("    - COMPLETED_MATCH advances to next subgoal")
    print("    - Task completion with double-confirmation works")
    print("    - Found and fixed bug: task_complete no longer advances path prematurely")


async def test_e2e_with_real_llm():
    """End-to-end test: real LLM planning + execution + mismatch checking.

    Terminal is simulated (mock), but ALL LLM calls are real:
    - Planner generates plans
    - Simulator predicts states
    - Graph builder merges states
    - ILP solver selects path
    - Agent LLM decides commands (via _call_llm_with_tools)
    - Mismatch checker evaluates completion + state match

    Task: "Create a file called hello.txt containing 'Hello World'"
    Mock terminal responds realistically to the LLM's chosen commands.
    """
    print("\n" + "=" * 60)
    print("TEST: End-to-End with Real LLM (mock terminal)")
    print("=" * 60)

    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from terminus_kira.terminus_kira_tape import TerminusKiraTAPE
    from terminus_kira.terminus_kira import Command
    from harbor.models.agent.context import AgentContext

    model = os.environ.get("TAPE_TEST_MODEL", "anthropic/claude-haiku-4-5-20251001")
    tmp_dir = Path("/tmp/tape_test_e2e")
    tmp_dir.mkdir(exist_ok=True)

    agent = TerminusKiraTAPE(
        logs_dir=tmp_dir,
        model_name=model,
        max_turns=15,
        enable_summarize=False,
        record_terminal_session=False,
        suppress_max_turns_warning=True,
    )

    # Mock session — simulates terminal that responds to commands
    mock_session = MagicMock()
    mock_session.is_session_alive = AsyncMock(return_value=True)
    mock_session.capture_pane = AsyncMock(return_value="$ pwd\n/home/user\n$ ")
    mock_session.send_keys = AsyncMock()
    agent._session = mock_session
    agent._context = AgentContext()

    # Mock chat — messages must accumulate like real Chat so LLM sees history
    mock_chat = MagicMock()
    mock_chat._messages = []
    mock_chat._cumulative_input_tokens = 0
    mock_chat._cumulative_output_tokens = 0
    mock_chat._cumulative_cache_tokens = 0
    mock_chat._cumulative_cost = 0.0
    # Properties that mirror real Chat behavior
    type(mock_chat).messages = property(lambda self: self._messages)
    type(mock_chat).total_input_tokens = property(
        lambda self: self._cumulative_input_tokens
    )
    type(mock_chat).total_output_tokens = property(
        lambda self: self._cumulative_output_tokens
    )
    type(mock_chat).total_cache_tokens = property(
        lambda self: self._cumulative_cache_tokens
    )
    type(mock_chat).total_cost = property(
        lambda self: self._cumulative_cost
    )
    mock_chat.rollout_details = []
    mock_chat.reset_response_chain = MagicMock()

    # ── Simulated terminal state machine ─────────────────────────────
    # The terminal knows about the "file system" and responds accordingly
    terminal_state = {"files": {}, "cwd": "/home/user"}
    episode_log = []

    async def mock_execute_commands(commands, session):
        """Simulate terminal: parse commands and produce realistic output."""
        output_lines = []
        for cmd in commands:
            ks = cmd.keystrokes.strip()
            if not ks:
                continue

            # echo 'X' > file
            if ">" in ks and "echo" in ks:
                import re
                m = re.match(r"""echo\s+['"](.*?)['"]\s*>\s*(\S+)""", ks)
                if m:
                    content, fname = m.group(1), m.group(2)
                    terminal_state["files"][fname] = content
                    output_lines.append(f"$ {ks}")
                    output_lines.append("$ ")
                else:
                    output_lines.append(f"$ {ks}")
                    output_lines.append("$ ")

            # cat file
            elif ks.startswith("cat "):
                fname = ks.split()[-1]
                if fname in terminal_state["files"]:
                    output_lines.append(f"$ {ks}")
                    output_lines.append(terminal_state["files"][fname])
                    output_lines.append("$ ")
                else:
                    output_lines.append(f"$ {ks}")
                    output_lines.append(f"cat: {fname}: No such file or directory")
                    output_lines.append("$ ")

            # ls
            elif ks.startswith("ls"):
                fname = ks.split()[-1] if len(ks.split()) > 1 else None
                if fname and fname.startswith("-"):
                    fname = ks.split()[-1] if len(ks.split()) > 2 else None
                if fname and fname in terminal_state["files"]:
                    output_lines.append(f"$ {ks}")
                    output_lines.append(fname)
                    output_lines.append("$ ")
                elif terminal_state["files"]:
                    output_lines.append(f"$ {ks}")
                    output_lines.append("  ".join(terminal_state["files"].keys()))
                    output_lines.append("$ ")
                else:
                    output_lines.append(f"$ {ks}")
                    output_lines.append("$ ")

            else:
                output_lines.append(f"$ {ks}")
                output_lines.append("$ ")

        output = "\n".join(output_lines) if output_lines else "$ "
        return False, output

    # Track episodes
    episode_counter = [0]
    original_build_obs = agent._build_observation

    def tracking_build_obs(is_task_complete, feedback, terminal_output):
        result = original_build_obs(is_task_complete, feedback, terminal_output)
        ep = episode_counter[0]
        episode_log.append({
            "episode": ep,
            "is_task_complete": is_task_complete,
            "subgoal_idx": getattr(agent, '_tape_subgoal_injected', None),
            "terminal_snippet": terminal_output[:100],
        })
        episode_counter[0] += 1
        return result

    agent._build_observation = tracking_build_obs

    # ── Run with real LLM, mock terminal ─────────────────────────────
    with (
        patch.object(agent, "_execute_commands", side_effect=mock_execute_commands),
        patch.object(agent, "_setup_episode_logging", return_value=(None, None, None)),
        patch.object(agent, "_dump_trajectory", return_value=None),
        patch.object(agent, "_record_asciinema_marker", return_value=None),
    ):
        result = await agent._run_agent_loop(
            initial_prompt="You are in /home/user. Create a file called hello.txt containing 'Hello World'.\n\n$ pwd\n/home/user\n$ ",
            chat=mock_chat,
            logging_dir=None,
            original_instruction="Create a file called hello.txt containing 'Hello World'",
        )

    print(f"\n  Agent completed in {result} episodes")
    print(f"  Terminal state: {terminal_state}")
    print(f"\n  Episode log:")
    for entry in episode_log:
        print(f"    ep={entry['episode']}: task_complete={entry['is_task_complete']}, "
              f"terminal={entry['terminal_snippet'][:60]}...")

    # Verify the task was actually accomplished
    assert "hello.txt" in terminal_state["files"], "hello.txt was not created!"
    assert terminal_state["files"]["hello.txt"] == "Hello World", (
        f"Wrong content: {terminal_state['files']['hello.txt']!r}"
    )
    print(f"\n  ✓ File created: hello.txt = {terminal_state['files']['hello.txt']!r}")
    print("  ✓ End-to-end test PASSED")


def main():
    print("TAPE Component Tests\n")

    # Tests that don't need LLM (graph builder uses mocked LLM)
    test_selected_path()
    graph = asyncio.run(test_graph_builder())
    test_solver(graph)

    # Execution loop test (mocked terminal + LLM)
    asyncio.run(test_execution_loop())

    # LLM tests (optional)
    has_key = any(os.environ.get(k) for k in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
    if has_key:
        print("\n\n>>> API key detected, running LLM tests <<<\n")
        if VERBOSE:
            _setup_llm_logger()
        asyncio.run(test_mismatch_checker_llm())
        asyncio.run(test_full_pipeline_llm())
        asyncio.run(test_e2e_with_real_llm())
    else:
        print("\n\n>>> No API key found, skipping LLM tests <<<")
        print(">>> Set ANTHROPIC_API_KEY or OPENAI_API_KEY to test planner/simulator <<<")
        print(">>> Optionally set TAPE_TEST_MODEL (default: anthropic/claude-haiku-4-5-20251001) <<<")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
