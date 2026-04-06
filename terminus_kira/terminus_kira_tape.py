"""TerminusKiraTAPE — TAPE-wrapped TerminusKira agent.

Overrides _run_agent_loop to add:
1. Plan generation + simulation + graph construction + ILP solving (before execution)
2. Constrained execution (subgoal injection + tool_choice forcing)
3. Two-stage mismatch checking (completion + state comparison) + replanning
"""

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import litellm
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
    BadRequestError,
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from anthropic_caching import add_anthropic_caching
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError
from harbor.llms.chat import Chat
from harbor.models.trajectories import Metrics, Observation, ObservationResult, Step, ToolCall

from terminus_kira.terminus_kira import TOOLS, TerminusKira, ToolCallResponse
from terminus_kira.tape.types import Plan, PlanEdge, SelectedPath, Subgoal, ToolType
from terminus_kira.tape.planner import TAPEPlanner
from terminus_kira.tape.simulator import TAPESimulator
from terminus_kira.tape.graph import PlanGraphBuilder
from terminus_kira.tape.solver import ILPSolver
from terminus_kira.tape.mismatch import MismatchChecker, SubgoalStatus

logger = logging.getLogger(__name__)


class TerminusKiraTAPE(TerminusKira):
    """TAPE-wrapped TerminusKira agent.

    Wraps the vanilla TerminusKira with TAPE's planning pipeline:
    Planner -> Simulator -> Graph -> ILP -> Constrained Execution -> Mismatch Check
    """

    def __init__(self, *args, tape_config: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = tape_config or {}
        # Hyperparameters: tape_config dict -> env var -> default
        self._tape_m = int(os.environ.get("TAPE_M", cfg.get("M", 4)))
        self._tape_time_budget_override = (
            float(os.environ["TAPE_TIME_BUDGET"])
            if "TAPE_TIME_BUDGET" in os.environ
            else cfg.get("time_budget", None)
        )
        # max_replans: None = unlimited (default), set via env or config to limit
        _max_replans_raw = os.environ.get(
            "TAPE_MAX_REPLANS", cfg.get("max_replans", None)
        )
        self._tape_max_replans: int | None = (
            int(_max_replans_raw) if _max_replans_raw is not None else None
        )
        self._tape_planner_temperature = float(
            os.environ.get("TAPE_PLANNER_TEMPERATURE", cfg.get("planner_temperature", 0.8))
        )

        # TAPE components (initialized lazily in _init_tape_components)
        self._tape_planner: TAPEPlanner | None = None
        self._tape_simulator: TAPESimulator | None = None
        self._tape_graph_builder: PlanGraphBuilder | None = None  # now requires model_name
        self._tape_solver: ILPSolver | None = None
        self._tape_mismatch_checker: MismatchChecker | None = None

        # Runtime state for constrained execution
        self._tape_tool_choice: dict | None = None
        self._tape_tools: list[dict] | None = None
        self._tape_replan_count: int = 0
        self._tape_terminal_history: str = ""  # unused, kept for compat
        self._tape_subgoal_injected: bool = False  # whether subgoal was already injected for current step
        self._tape_recent_outputs: list[str] = []  # last N terminal outputs for judge context
        self._tape_double_conf_count: int = 0  # consecutive double-confirmation failures

    @staticmethod
    def name() -> str:
        return "terminus-kira-tape"

    def version(self) -> str | None:
        return "1.0.0"

    def _init_tape_components(self):
        """Initialize TAPE components once model_name is available."""
        # Ensure TAPE logs (info+) are visible in job.log
        tape_logger = logging.getLogger("terminus_kira.tape")
        if not tape_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            tape_logger.addHandler(handler)
            tape_logger.setLevel(logging.INFO)
        # Also ensure this module's logger is visible
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        api_base = getattr(self._llm, "_api_base", None) if self._llm else None
        self._tape_planner = TAPEPlanner(
            self._model_name, api_base, self._tape_m, self._tape_planner_temperature,
        )
        self._tape_simulator = TAPESimulator(self._model_name, api_base)
        self._tape_graph_builder = PlanGraphBuilder(self._model_name, api_base)
        self._tape_solver = ILPSolver(self._tape_time_budget_override)
        self._tape_mismatch_checker = MismatchChecker(self._model_name, api_base)

    # ------------------------------------------------------------------
    # TAPE Planning Pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def _build_history_context(chat: Chat | None, extra_context: str = "") -> str:
        """Build interaction history from chat messages for the planner.

        Assistant messages store executed commands in tool_calls, not in content.
        We extract function name + arguments from tool_calls to show what was executed.
        """
        parts = []

        if chat is not None:
            for msg in chat.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                if role == "system":
                    continue

                if role == "assistant":
                    # Extract agent's reasoning/content
                    agent_parts = []
                    if content and isinstance(content, str):
                        truncated = content[:800] + "...(truncated)" if len(content) > 800 else content
                        agent_parts.append(truncated)
                    # Extract executed tool calls (commands, task_complete, image_read)
                    if tool_calls:
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            func_name = func.get("name", "unknown")
                            args_raw = func.get("arguments", "{}")
                            try:
                                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                            except (json.JSONDecodeError, TypeError):
                                args = {"raw": str(args_raw)[:200]}
                            if func_name == "execute_commands":
                                cmds = args.get("commands", [])
                                cmd_strs = []
                                for c in cmds:
                                    if isinstance(c, dict):
                                        cmd_strs.append(c.get("keystrokes", "").rstrip("\n"))
                                    elif isinstance(c, str):
                                        cmd_strs.append(c.rstrip("\n"))
                                if cmd_strs:
                                    agent_parts.append(f"Commands: {cmd_strs}")
                            elif func_name == "task_complete":
                                agent_parts.append("Action: task_complete")
                            elif func_name == "image_read":
                                agent_parts.append(f"Action: image_read({args.get('file_path', '')})")
                    if agent_parts:
                        parts.append(f"[Agent]:\n" + "\n".join(agent_parts))

                elif role == "user":
                    if content and isinstance(content, str):
                        truncated = content[:1000] + "...(truncated)" if len(content) > 1000 else content
                        parts.append(f"[Observation]:\n{truncated}")

                # Skip "tool" role messages — they just say "executed"

        if extra_context:
            parts.append(f"[Note]: {extra_context}")

        if not parts:
            return ""

        # Keep total history under ~8000 chars to fit in planner context
        history = "\n\n".join(parts)
        if len(history) > 8000:
            history = "...(earlier history truncated)...\n\n" + history[-8000:]

        return history

    async def _tape_plan_and_select(
        self,
        task_instruction: str,
        current_terminal_state: str,
        chat_history_summary: str = "",
        chat: Chat | None = None,
        logging_dir: Path | None = None,
    ) -> SelectedPath | None:
        """Full TAPE planning pipeline: plan -> simulate -> graph -> ILP."""
        assert self._tape_planner is not None
        assert self._tape_simulator is not None
        assert self._tape_graph_builder is not None
        assert self._tape_solver is not None

        # Build history context from chat + any extra summary
        history_context = self._build_history_context(chat, chat_history_summary)

        # Step 1: Generate M candidate plans
        logger.info("[TAPE] Step 1/4: Generating %d candidate plans...", self._tape_m)
        plans = await self._tape_planner.generate_plans(
            task_instruction, current_terminal_state, history_context,
        )
        if not plans:
            logger.warning("[TAPE] No plans generated — falling back to vanilla")
            return None
        for p in plans:
            subgoal_names = [sg.description[:80] for sg in p.subgoals]
            logger.info("[TAPE]   Plan %d: %d subgoals — %s", p.plan_id, len(p.subgoals), subgoal_names)

        # Step 2: Simulate all plans
        logger.info("[TAPE] Step 2/4: Simulating %d plans...", len(plans))
        simulated_plans = await self._tape_simulator.simulate_all(
            plans, task_instruction, current_terminal_state, history_context,
        )
        if not simulated_plans:
            logger.warning("[TAPE] All simulations failed — falling back to vanilla")
            return None

        # Step 3: Build plan graph (LLM-based state merging)
        logger.info("[TAPE] Step 3/4: Building plan graph from %d simulated plans...", len(simulated_plans))
        graph = await self._tape_graph_builder.build_graph(
            simulated_plans, current_terminal_state, task_instruction,
        )
        logger.info("[TAPE]   Graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges))

        # Step 4: ILP solve
        logger.info("[TAPE] Step 4/4: Solving ILP...")
        selected = self._tape_solver.solve(graph)

        if selected is None:
            logger.warning("[TAPE] ILP infeasible, falling back to best single plan")
            best_plan = max(simulated_plans, key=lambda p: p.total_success_prob)
            selected = self._plan_to_path(best_plan)

        logger.info(
            "[TAPE] Selected path: %d steps, reward=%.3f, cost=%.1fs",
            selected.total_steps, selected.total_reward, selected.total_cost,
        )
        for i, edge in enumerate(selected.edges):
            sg = edge.subgoal
            if sg is not None:
                logger.info("[TAPE]   Step %d: [%s] %s", i + 1, sg.predicted_tool.value, sg.description[:100])

        # Dump TAPE pipeline results to logging_dir
        if logging_dir is not None:
            self._dump_tape_pipeline(
                logging_dir, plans, simulated_plans, graph, selected,
            )

        return selected

    @staticmethod
    def _plan_to_path(plan: Plan) -> SelectedPath:
        """Convert a single Plan to a SelectedPath (fallback when ILP fails)."""
        edges = []
        for i, sg in enumerate(plan.subgoals):
            from_node = "start" if i == 0 else f"node_{i - 1}"
            to_node = f"node_{i}"
            edges.append(
                PlanEdge(
                    edge_id=f"fallback_edge_{i}",
                    from_node=from_node,
                    to_node=to_node,
                    subgoal=sg,
                    reward=sg.success_probability,
                    cost=sg.estimated_duration,
                )
            )
        return SelectedPath(
            edges=edges,
            total_reward=plan.total_success_prob,
            total_cost=plan.total_estimated_duration,
        )

    # ------------------------------------------------------------------
    # Constrained Execution Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_subgoal_into_tools(subgoal: Subgoal) -> list[dict]:
        """Create modified TOOLS with subgoal injected into descriptions."""
        tools = copy.deepcopy(TOOLS)
        for tool in tools:
            fname = tool["function"]["name"]
            if fname == subgoal.predicted_tool.value:
                tool["function"]["description"] = (
                    tool["function"]["description"]
                    + f"\n\nCURRENT SUBGOAL: {subgoal.description}\n"
                    f"Execute to achieve this subgoal."
                )
                if fname == "execute_commands":
                    props = tool["function"]["parameters"]["properties"]
                    if "plan" in props:
                        props["plan"]["description"] = (
                            f"Your plan to accomplish: {subgoal.description}"
                        )
        return tools

    @staticmethod
    def _build_tool_choice(tool_type: ToolType) -> dict:
        """Build tool_choice dict for the given tool type."""
        return {
            "type": "function",
            "function": {"name": tool_type.value},
        }

    def _build_subgoal_prompt(
        self,
        terminal_output: str,
        selected_path: SelectedPath,
    ) -> str:
        """Build observation prompt with subgoal context injected."""
        subgoal = selected_path.current_subgoal
        if subgoal is None:
            return terminal_output

        step_num = selected_path.current_step_idx + 1
        total_steps = selected_path.total_steps

        subgoal_context = (
            f"\n\n[SUBGOAL]: {subgoal.description}\n"
            f"Expected outcome: {subgoal.predicted_state}\n"
        )

        return terminal_output + subgoal_context

    # ------------------------------------------------------------------
    # Override: _call_llm_with_tools (inject tool_choice + tools)
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=(
            retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (
                    BadRequestError,
                    LiteLLMAuthenticationError,
                    ContextLengthExceededError,
                    OutputLengthExceededError,
                )
            )
        ),
        reraise=True,
    )
    async def _call_llm_with_tools(
        self,
        messages: list[dict],
    ) -> ToolCallResponse:
        """Override to inject TAPE constraints (tool_choice, modified tools)."""
        messages = add_anthropic_caching(messages, self._model_name)

        tools = self._tape_tools if self._tape_tools is not None else TOOLS

        completion_kwargs = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
            "tools": tools,
            "timeout": 900,
            "drop_params": True,
        }

        if self._tape_tool_choice is not None:
            completion_kwargs["tool_choice"] = self._tape_tool_choice

        if hasattr(self._llm, "_api_base") and self._llm._api_base:  # noqa: B009
            completion_kwargs["api_base"] = self._llm._api_base  # noqa: B009

        if self._reasoning_effort:
            completion_kwargs["reasoning_effort"] = self._reasoning_effort
            completion_kwargs["temperature"] = 1

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except LiteLLMContextWindowExceededError as exc:
            raise ContextLengthExceededError() from exc

        message = response.choices[0].message
        content = message.content or ""
        tool_calls = self._extract_tool_calls(response)
        usage_info = self._extract_usage_info(response)

        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise OutputLengthExceededError(
                "Response was truncated due to max tokens limit",
                truncated_response=content,
            )

        reasoning_content = None
        if hasattr(message, "reasoning_content"):
            reasoning_content = message.reasoning_content

        return ToolCallResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            usage=usage_info,
        )

    # ------------------------------------------------------------------
    # Override: _run_agent_loop (TAPE-aware execution)
    # ------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        chat: Chat,
        logging_dir: Path | None = None,
        original_instruction: str = "",
    ) -> int:
        """TAPE-aware agent loop.

        1. TAPE planning phase (plan -> simulate -> graph -> ILP)
        2. Constrained execution (subgoal injection + tool_choice forcing)
        3. Two-stage check: subgoal completion + state mismatch
        4. Adaptive replanning on mismatch
        """
        if self._context is None or self._session is None:
            raise RuntimeError("Context/session not set")

        self._init_tape_components()

        # Get initial terminal state
        initial_terminal = await self._with_block_timeout(
            self._session.capture_pane(capture_entire=False)
        )

        # TAPE planning phase
        selected_path = await self._tape_plan_and_select(
            original_instruction,
            initial_terminal or "",
            chat=chat,
            logging_dir=logging_dir,
        )

        if selected_path is None:
            logger.info("[TAPE] Planning failed, running vanilla TerminusKira")
            return await super()._run_agent_loop(
                initial_prompt, chat, logging_dir, original_instruction,
            )

        # Initialize loop state
        prompt = initial_prompt
        self._tape_terminal_history = initial_terminal or ""
        self._context.n_input_tokens = 0
        self._context.n_output_tokens = 0
        self._context.n_cache_tokens = 0
        self._context.cost_usd = None
        episode = 0

        while episode < self._max_episodes:
            if selected_path.is_complete:
                logger.info("[TAPE] All subgoals completed")
                break

            self._n_episodes = episode + 1

            if not await self._with_block_timeout(self._session.is_session_alive()):
                logger.debug("[TAPE] Session ended")
                break

            # Proactive summarization (inherited)
            if original_instruction and self._enable_summarize:
                proactive_result = await self._with_block_timeout(
                    self._check_proactive_summarization(
                        chat, original_instruction, self._session,
                    )
                )
                if proactive_result:
                    prompt, subagent_refs = proactive_result
                    self._pending_subagent_refs = subagent_refs
                    self._pending_handoff_prompt = prompt

            # --- CONSTRAINED EXECUTION ---
            current_subgoal = selected_path.current_subgoal
            assert current_subgoal is not None

            if self._pending_completion:
                # Double-confirmation: agent already called task_complete,
                # let it freely decide — confirm or back out
                constrained_prompt = prompt
                self._tape_tool_choice = None
                self._tape_tools = None
            elif not self._tape_subgoal_injected:
                # First attempt at this subgoal — inject constraints
                # 1. Inject subgoal into observation
                constrained_prompt = self._build_subgoal_prompt(prompt, selected_path)
                # 2. Set tool_choice forcing
                self._tape_tool_choice = self._build_tool_choice(
                    current_subgoal.predicted_tool
                )
                # 3. Modify tool descriptions
                self._tape_tools = self._inject_subgoal_into_tools(current_subgoal)
                self._tape_subgoal_injected = True
            else:
                # IN_PROGRESS continuation — no tool_choice forcing
                constrained_prompt = prompt
                self._tape_tool_choice = None
                # Block task_complete unless current subgoal is TASK_COMPLETE
                if current_subgoal.predicted_tool != ToolType.TASK_COMPLETE:
                    self._tape_tools = [
                        t for t in TOOLS
                        if t["function"]["name"] != "task_complete"
                    ]
                else:
                    self._tape_tools = None

            # --- LLM INTERACTION (reuse parent's full logic) ---
            logging_paths = self._setup_episode_logging(logging_dir, episode)
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            (
                commands,
                is_task_complete,
                feedback,
                analysis,
                plan,
                llm_response,
                image_read,
            ) = await self._handle_llm_interaction(
                chat, constrained_prompt, logging_paths,
                original_instruction, self._session,
            )

            # Clear TAPE constraints
            self._tape_tool_choice = None
            self._tape_tools = None

            # Handle pending subagent refs / handoff (inherited logic)
            if self._pending_subagent_refs:
                self._trajectory_steps.append(
                    Step(
                        step_id=len(self._trajectory_steps) + 1,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="system",
                        message="Performed context summarization and handoff.",
                        observation=Observation(
                            results=[
                                ObservationResult(
                                    subagent_trajectory_ref=self._pending_subagent_refs
                                )
                            ]
                        ),
                    )
                )
                self._pending_subagent_refs = None

            if self._pending_handoff_prompt:
                if self._linear_history:
                    self._split_trajectory_on_summarization(
                        self._pending_handoff_prompt
                    )
                else:
                    self._trajectory_steps.append(
                        Step(
                            step_id=len(self._trajectory_steps) + 1,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            source="user",
                            message=self._pending_handoff_prompt,
                        )
                    )
                self._pending_handoff_prompt = None

            # Build message content for trajectory
            if self._save_raw_content_in_trajectory:
                message_content = llm_response.content
            else:
                message_parts = []
                if analysis:
                    message_parts.append(f"Analysis: {analysis}")
                if plan:
                    message_parts.append(f"Plan: {plan}")
                message_content = "\n".join(message_parts) if message_parts else ""

            # Update context metrics
            self._context.n_input_tokens = chat.total_input_tokens
            self._context.n_output_tokens = chat.total_output_tokens
            self._context.n_cache_tokens = chat.total_cache_tokens
            self._context.cost_usd = chat.total_cost if chat.total_cost > 0 else None

            self._record_asciinema_marker(
                f"Episode {episode} [TAPE {selected_path.current_step_idx + 1}"
                f"/{selected_path.total_steps}]: {len(commands)} commands"
                + (" (image_read)" if image_read else ""),
            )

            # Handle feedback errors — retry without advancing subgoal
            if feedback and "ERROR:" in feedback:
                prompt = (
                    f"Previous response had parsing errors:\n{feedback}\n\n"
                    f"Please fix these issues and provide a proper "
                    f"{self._get_error_response_type()}."
                )
                self._record_trajectory_step(
                    llm_response, prompt, message_content,
                    tokens_before_input, tokens_before_output,
                    tokens_before_cache, cost_before, chat,
                )
                episode += 1
                continue

            # --- EXECUTE ---
            if image_read is not None:
                terminal_output = await self._execute_image_read(
                    image_read, chat, original_instruction
                )
            else:
                _, terminal_output = await self._with_block_timeout(
                    self._execute_commands(commands, self._session)
                )

            # terminal_history accumulation removed — step judge uses current output only

            # Handle task completion with double-confirmation
            was_pending_completion = self._pending_completion
            observation = self._build_observation(
                is_task_complete, feedback, terminal_output,
            )

            # Record trajectory step
            self._record_trajectory_step_with_tools(
                llm_response, observation, message_content,
                tokens_before_input, tokens_before_output,
                tokens_before_cache, cost_before, chat,
                episode, commands, image_read, is_task_complete,
            )

            # Handle task completion
            if is_task_complete and was_pending_completion:
                self._tape_double_conf_count = 0
                return episode + 1

            if is_task_complete:
                # If agent already failed double-confirmation once, accept immediately
                if self._tape_double_conf_count >= 1:
                    logger.info("[TAPE] Agent retried task_complete after prior decline, accepting")
                    self._tape_double_conf_count = 0
                    return episode + 1
                # First confirmation sent — don't advance path, wait for re-confirm
                self._tape_subgoal_injected = True  # suppress re-injection on next episode
                prompt = observation
                episode += 1
                continue

            if was_pending_completion and not is_task_complete:
                # Agent backed out of task_complete — replan with context
                self._tape_double_conf_count += 1
                logger.info(
                    "[TAPE] Agent declined double-confirmation (#%d), replanning with context",
                    self._tape_double_conf_count,
                )
                self._pending_completion = False  # reset before replan
                # Add context about why we're replanning
                replan_context = (
                    "The agent attempted to mark the task as complete but then "
                    "determined it is NOT ready. The agent's latest response "
                    "indicates remaining work is needed. "
                    "Plan what additional steps are required to finish the task."
                )
                observation = f"{observation}\n\n[REPLAN CONTEXT] {replan_context}"
                selected_path, episode = await self._handle_replan(
                    selected_path, current_subgoal,
                    terminal_output, observation,
                    original_instruction, chat, logging_dir, episode,
                )
                if selected_path is None:
                    return episode
                prompt = observation
                episode += 1
                continue

            # --- STEP JUDGMENT: COMPLETE / IN_PROGRESS / REPLAN ---
            if self._tape_mismatch_checker is not None:
                status = await self._tape_mismatch_checker.check(
                    terminal_history="",
                    current_terminal_output=terminal_output,
                    subgoal_description=current_subgoal.description,
                    predicted_state=current_subgoal.predicted_state,
                    task_instruction=original_instruction,
                    agent_analysis=analysis,
                    agent_plan=plan,
                    recent_outputs=self._tape_recent_outputs[-2:],
                )
                # Track recent outputs (keep last 2)
                self._tape_recent_outputs.append(terminal_output)
                if len(self._tape_recent_outputs) > 2:
                    self._tape_recent_outputs.pop(0)

                if status == SubgoalStatus.REPLAN:
                    # REPLAN — state diverged
                    logger.info(
                        "[TAPE] REPLAN at step %d/%d (replan #%d): %s",
                        selected_path.current_step_idx + 1,
                        selected_path.total_steps,
                        self._tape_replan_count + 1,
                        current_subgoal.description,
                    )
                    self._dump_tape_judgment(
                        logging_dir, episode, current_subgoal.description,
                        "replan", terminal_output,
                    )
                    selected_path, episode = await self._handle_replan(
                        selected_path, current_subgoal,
                        terminal_output, observation,
                        original_instruction, chat, logging_dir, episode,
                    )
                    if selected_path is None:
                        return episode
                    prompt = observation
                    episode += 1
                    continue

                if status == SubgoalStatus.IN_PROGRESS:
                    # IN_PROGRESS — stay on same subgoal
                    self._dump_tape_judgment(
                        logging_dir, episode, current_subgoal.description,
                        "in_progress", terminal_output,
                    )
                    logger.info(
                        "[TAPE] IN_PROGRESS step %d/%d: %s",
                        selected_path.current_step_idx + 1,
                        selected_path.total_steps,
                        current_subgoal.description,
                    )
                    prompt = observation
                    episode += 1
                    continue

            # COMPLETE — subgoal achieved, move to next subgoal
            self._dump_tape_judgment(
                logging_dir, episode, current_subgoal.description,
                "complete", terminal_output,
            )
            logger.info(
                "[TAPE] COMPLETE step %d/%d: %s",
                selected_path.current_step_idx + 1,
                selected_path.total_steps,
                current_subgoal.description,
            )
            selected_path.current_step_idx += 1
            self._tape_subgoal_injected = False
            prompt = observation
            episode += 1

        return episode

    # ------------------------------------------------------------------
    # Helper methods (extracted to reduce _run_agent_loop complexity)
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        is_task_complete: bool,
        feedback: str,
        terminal_output: str,
    ) -> str:
        """Build observation string, handling task completion and warnings."""
        if is_task_complete:
            if self._pending_completion:
                return terminal_output
            self._pending_completion = True
            return self._get_completion_confirmation_message(terminal_output)

        self._pending_completion = False
        if feedback and "WARNINGS:" in feedback:
            return (
                f"Previous response had warnings:\n{feedback}\n\n"
                f"{self._limit_output_length(terminal_output)}"
            )
        return self._limit_output_length(terminal_output)

    def _record_trajectory_step(
        self, llm_response, observation, message_content,
        tokens_before_input, tokens_before_output,
        tokens_before_cache, cost_before, chat,
    ):
        """Record a trajectory step (for error/retry cases without tool calls)."""
        cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
        step_cost = chat.total_cost - cost_before
        self._trajectory_steps.append(
            Step(
                step_id=len(self._trajectory_steps) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=self._model_name,
                message=llm_response.content,
                reasoning_content=llm_response.reasoning_content,
                observation=Observation(
                    results=[ObservationResult(content=observation)]
                ),
                metrics=Metrics(
                    prompt_tokens=chat.total_input_tokens - tokens_before_input,
                    completion_tokens=chat.total_output_tokens - tokens_before_output,
                    cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                    cost_usd=step_cost if step_cost > 0 else None,
                ),
            )
        )

    def _record_trajectory_step_with_tools(
        self, llm_response, observation, message_content,
        tokens_before_input, tokens_before_output,
        tokens_before_cache, cost_before, chat,
        episode, commands, image_read, is_task_complete,
    ):
        """Record a trajectory step with tool call details."""
        cache_tokens_used = chat.total_cache_tokens - tokens_before_cache
        step_cost = chat.total_cost - cost_before

        tool_calls_list: list[ToolCall] = []
        if not self._save_raw_content_in_trajectory:
            if image_read is not None:
                tool_calls_list.append(
                    ToolCall(
                        tool_call_id=f"call_{episode}_image_read",
                        function_name="image_read",
                        arguments={
                            "file_path": image_read.file_path,
                            "image_read_instruction": image_read.image_read_instruction,
                        },
                    )
                )
            elif commands:
                for i, cmd in enumerate(commands):
                    tool_calls_list.append(
                        ToolCall(
                            tool_call_id=f"call_{episode}_{i + 1}",
                            function_name="bash_command",
                            arguments={
                                "keystrokes": cmd.keystrokes,
                                "duration": cmd.duration_sec,
                            },
                        )
                    )
            if is_task_complete:
                tool_calls_list.append(
                    ToolCall(
                        tool_call_id=f"call_{episode}_task_complete",
                        function_name="mark_task_complete",
                        arguments={},
                    )
                )

        self._trajectory_steps.append(
            Step(
                step_id=len(self._trajectory_steps) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=self._model_name,
                message=message_content,
                reasoning_content=llm_response.reasoning_content,
                tool_calls=tool_calls_list or None,
                observation=Observation(
                    results=[ObservationResult(content=observation)]
                ),
                metrics=Metrics(
                    prompt_tokens=chat.total_input_tokens - tokens_before_input,
                    completion_tokens=chat.total_output_tokens - tokens_before_output,
                    cached_tokens=cache_tokens_used if cache_tokens_used > 0 else None,
                    cost_usd=step_cost if step_cost > 0 else None,
                    prompt_token_ids=llm_response.prompt_token_ids,
                    completion_token_ids=llm_response.completion_token_ids,
                    logprobs=llm_response.logprobs,
                ),
            )
        )
        self._dump_trajectory()

    async def _handle_replan(
        self, selected_path, current_subgoal,
        terminal_output, observation,
        original_instruction, chat, logging_dir, episode,
    ) -> tuple[SelectedPath | None, int]:
        """Handle replanning after a mismatch. Returns (new_path, episode)."""
        self._tape_replan_count += 1

        if self._tape_max_replans is not None and self._tape_replan_count > self._tape_max_replans:
            logger.warning("[TAPE] Max replans exceeded, switching to vanilla")
            episode += 1
            remaining = self._max_episodes - episode
            if remaining > 0:
                episode += await self._run_vanilla_remaining(
                    observation, chat, logging_dir,
                    original_instruction, remaining,
                )
            return None, episode

        logger.info("[TAPE] Replanning...")
        new_path = await self._tape_plan_and_select(
            original_instruction,
            terminal_output,
            chat_history_summary=(
                f"Previous plan failed at step: "
                f"{current_subgoal.description}."
            ),
            chat=chat,
            logging_dir=logging_dir,
        )

        if new_path is None:
            logger.warning("[TAPE] Replan failed, switching to vanilla")
            episode += 1
            remaining = self._max_episodes - episode
            if remaining > 0:
                episode += await self._run_vanilla_remaining(
                    observation, chat, logging_dir,
                    original_instruction, remaining,
                )
            return None, episode

        self._tape_terminal_history = ""  # reset for new plan
        self._tape_subgoal_injected = False  # reset for new plan
        return new_path, episode

    async def _run_vanilla_remaining(
        self,
        prompt: str,
        chat: Chat,
        logging_dir: Path | None,
        original_instruction: str,
        max_episodes: int,
    ) -> int:
        """Run vanilla TerminusKira loop for remaining episodes."""
        original_max = self._max_episodes
        self._max_episodes = max_episodes
        try:
            result = await super()._run_agent_loop(
                prompt, chat, logging_dir, original_instruction,
            )
        finally:
            self._max_episodes = original_max
        return result

    # ------------------------------------------------------------------
    # TAPE Logging
    # ------------------------------------------------------------------

    def _dump_tape_pipeline(
        self,
        logging_dir: Path,
        plans: list,
        simulated_plans: list,
        graph,
        selected: SelectedPath,
    ) -> None:
        """Dump full TAPE pipeline results to tape/ subdirectory."""
        tape_dir = logging_dir / f"tape-plan-{self._tape_replan_count}"
        tape_dir.mkdir(parents=True, exist_ok=True)

        # 1. Raw plans from planner
        plans_data = []
        for p in plans:
            plans_data.append({
                "plan_id": p.plan_id,
                "subgoals": [
                    {
                        "id": sg.id,
                        "description": sg.description,
                        "predicted_tool": sg.predicted_tool.value,
                    }
                    for sg in p.subgoals
                ],
            })
        (tape_dir / "1_plans.json").write_text(
            json.dumps(plans_data, indent=2, ensure_ascii=False)
        )

        # 2. Simulated plans (with predicted states and probabilities)
        sim_data = []
        for p in simulated_plans:
            sim_data.append({
                "plan_id": p.plan_id,
                "total_success_prob": p.total_success_prob,
                "total_estimated_duration": p.total_estimated_duration,
                "subgoals": [
                    {
                        "id": sg.id,
                        "description": sg.description,
                        "predicted_tool": sg.predicted_tool.value,
                        "predicted_state": sg.predicted_state,
                        "success_probability": sg.success_probability,
                        "estimated_duration": sg.estimated_duration,
                    }
                    for sg in p.subgoals
                ],
            })
        (tape_dir / "2_simulated.json").write_text(
            json.dumps(sim_data, indent=2, ensure_ascii=False)
        )

        # 3. Graph structure
        graph_data = {
            "start_node": graph.start_node,
            "goal_nodes": graph.goal_nodes,
            "nodes": {
                nid: {
                    "state_description": n.state_description,
                    "is_start": n.is_start,
                    "is_goal": n.is_goal,
                }
                for nid, n in graph.nodes.items()
            },
            "edges": {
                eid: {
                    "from": e.from_node,
                    "to": e.to_node,
                    "reward": e.reward,
                    "cost": e.cost,
                    "subgoal": e.subgoal.description if e.subgoal else None,
                }
                for eid, e in graph.edges.items()
            },
        }
        (tape_dir / "3_graph.json").write_text(
            json.dumps(graph_data, indent=2, ensure_ascii=False)
        )

        # 4. Selected path
        path_data = {
            "total_steps": selected.total_steps,
            "total_reward": selected.total_reward,
            "total_cost": selected.total_cost,
            "steps": [
                {
                    "edge_id": e.edge_id,
                    "from": e.from_node,
                    "to": e.to_node,
                    "subgoal": e.subgoal.description if e.subgoal else None,
                    "predicted_tool": e.subgoal.predicted_tool.value if e.subgoal else None,
                    "predicted_state": e.subgoal.predicted_state if e.subgoal else None,
                    "reward": e.reward,
                    "cost": e.cost,
                }
                for e in selected.edges
            ],
        }
        (tape_dir / "4_selected_path.json").write_text(
            json.dumps(path_data, indent=2, ensure_ascii=False)
        )

        logger.info("[TAPE] Pipeline dumped to %s", tape_dir)

    def _dump_tape_judgment(
        self,
        logging_dir: Path | None,
        episode: int,
        subgoal_description: str,
        decision: str,
        terminal_output: str,
    ) -> None:
        """Dump step judge result alongside the episode directory."""
        if logging_dir is None:
            return
        judge_file = logging_dir / f"episode-{episode}" / "tape_judgment.json"
        judge_file.parent.mkdir(parents=True, exist_ok=True)
        judge_data = {
            "episode": episode,
            "subgoal": subgoal_description,
            "decision": decision,
            "terminal_output_snippet": terminal_output[-1000:] if len(terminal_output) > 1000 else terminal_output,
        }
        judge_file.write_text(json.dumps(judge_data, indent=2, ensure_ascii=False))
