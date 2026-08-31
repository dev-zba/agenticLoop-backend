"""LangGraph: spec loop + implementation loop + blocked checkpoint."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from app.agents.adversary import run_adversary
from app.agents.builder import run_builder
from app.agents.events import EventCallback, noop_event
from app.agents.explorer import run_explorer
from app.agents.metrics import AgentMetrics
from app.agents.evidence import run_evidence
from app.agents.spec_detective import run_spec_detective
from app.agents.verifier import run_verifier
from app.security import sanitize_error_message
from app.state import MAX_BUILD_ITERATIONS, MAX_SPEC_ITERATIONS, Requirement, WorkflowState
from app.tools.repo_tools import Sandbox
from app.trajectories import TrajectoryRecorder


@dataclass
class PipelineResult:
    specification: list[Requirement] = field(default_factory=list)
    rejected_requirements: list[Requirement] = field(default_factory=list)
    revision_log: list[dict[str, Any]] = field(default_factory=list)
    explorer_findings: dict[str, Any] = field(default_factory=dict)
    evidence_report: dict[str, Any] = field(default_factory=dict)
    adversary_findings: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    implementation_diff: str | None = None
    verification: dict[str, Any] | None = None
    tests_passed: int = 0
    tests_failed: int = 0
    test_output: str = ""
    spec_iteration: int = 0
    build_iteration: int = 0
    runtime_seconds: float = 0.0
    token_cost: float = 0.0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "success"
    error: str | None = None
    trajectory_dir: str | None = None
    run_id: str | None = None
    checkpoint: dict[str, Any] | None = None


def build_graph(sandbox: Sandbox, emit: EventCallback, metrics: AgentMetrics):
    graph = StateGraph(WorkflowState)

    def explorer_node(state: WorkflowState) -> dict[str, Any]:
        findings = run_explorer(state["request"], sandbox, emit, metrics)
        return {"explorer_findings": findings, "status": "running"}

    def spec_detective_node(state: WorkflowState) -> dict[str, Any]:
        iteration = int(state.get("spec_iteration") or 0) + 1
        conflicts = list(state.get("conflicts") or [])
        prior = list(state.get("specification") or [])
        # Spec-problem from Verifier also carries conflicts
        verification = state.get("verification") or {}
        if verification.get("failure_class") == "spec" and verification.get("spec_feedback"):
            conflicts = list(conflicts) + list(verification.get("spec_feedback") or [])
        spec = run_spec_detective(
            state["request"],
            state.get("explorer_findings") or {},
            sandbox,
            emit,
            metrics,
            conflicts=conflicts if iteration > 1 or verification.get("failure_class") == "spec" else None,
            prior_specification=prior if iteration > 1 or verification.get("failure_class") == "spec" else None,
            spec_iteration=iteration,
        )
        return {
            "specification": spec,
            "spec_iteration": iteration,
            "conflicts": [],
            "status": "running",
            "verification": {**(verification or {}), "failure_class": None, "spec_feedback": None}
            if verification
            else None,
        }

    def evidence_node(state: WorkflowState) -> dict[str, Any]:
        spec, report = run_evidence(
            state["request"],
            state.get("specification") or [],
            sandbox,
            emit,
            metrics,
        )
        return {"specification": spec, "evidence_report": report, "status": "running"}

    def adversary_node(state: WorkflowState) -> dict[str, Any]:
        iteration = int(state.get("spec_iteration") or 1)
        findings, conflicts = run_adversary(
            state["request"],
            state.get("specification") or [],
            sandbox,
            emit,
            metrics,
            spec_iteration=iteration,
        )
        # Forced blocked-demo marker file (ignored after human accept_assumption)
        force = sandbox.worktree_path / "FORCE_SPEC_CONFLICT"
        req_l = (state.get("request") or "").lower()
        human_cleared = (
            "accept remaining assumptions" in req_l
            or "human clarification:" in req_l
            or "human checkpoint:" in req_l
        )
        if force.exists() and not human_cleared:
            conflicts = list(conflicts) + [
                {
                    "requirement_id": "FORCE",
                    "summary": "FORCE_SPEC_CONFLICT marker present — unresolved adversarial block",
                    "detail": (
                        "This eval case keeps a permanent conflict until max spec iterations "
                        "to exercise the BLOCKED human-checkpoint path."
                    ),
                    "evidence": ["FORCE_SPEC_CONFLICT:1"],
                }
            ]
        updates: dict[str, Any] = {
            "adversary_findings": findings,
            "conflicts": conflicts,
            "status": "running",
        }
        if conflicts and iteration < MAX_SPEC_ITERATIONS:
            log = list(state.get("revision_log") or [])
            log.append(
                {
                    "from_iteration": iteration,
                    "to_iteration": iteration + 1,
                    "action": "sent_back_to_spec_detective",
                    "conflicts": conflicts,
                    "draft_snapshot": [
                        {
                            "id": r.get("id"),
                            "text": r.get("text"),
                            "status": r.get("status"),
                            "evidence": r.get("evidence"),
                        }
                        for r in (state.get("specification") or [])
                    ],
                }
            )
            updates["revision_log"] = log
        return updates

    def accept_spec_node(state: WorkflowState) -> dict[str, Any]:
        accepted: list[Requirement] = []
        rejected: list[Requirement] = []
        for req in state.get("specification") or []:
            status = req.get("status")
            if status == "supported":
                accepted.append({**req, "status": "accepted"})  # type: ignore[misc]
            elif status == "accepted":
                accepted.append(req)
            else:
                reason = {
                    "contradicted": "Rejected — contradicted by repo evidence",
                    "insufficient_evidence": "Rejected — insufficient evidence (not in repo)",
                    "proposed": "Rejected — never verified",
                }.get(str(status), f"Rejected — status={status}")
                item = {**req, "status": status}
                item["rejection_reason"] = reason  # type: ignore[typeddict-unknown-key]
                rejected.append(item)  # type: ignore[arg-type]
        emit(
            "spec_updated",
            {
                "count": len(accepted),
                "requirements": accepted,
                "rejected": rejected,
                "source": "accepted",
                "spec_iteration": state.get("spec_iteration"),
            },
        )
        if not accepted:
            emit(
                "checkpoint_needed",
                {
                    "reason": "no_accepted_requirements",
                    "message": "No supported requirements survived Evidence/Adversary. Human checkpoint required.",
                    "rejected": len(rejected),
                },
            )
            return {
                "specification": accepted,
                "rejected_requirements": rejected,
                "status": "blocked",
                "conflicts": [],
            }
        return {
            "specification": accepted,
            "rejected_requirements": rejected,
            "status": "running",
            "conflicts": [],
        }

    def builder_node(state: WorkflowState) -> dict[str, Any]:
        build_i = int(state.get("build_iteration") or 0) + 1
        result = run_builder(
            state["request"],
            state.get("specification") or [],
            sandbox,
            emit,
            metrics,
            build_iteration=build_i,
        )
        prev = state.get("verification") or {}
        return {
            "implementation_diff": result.get("diff") or "",
            "verification": {
                **prev,
                "tests_passed": result.get("tests_passed", 0),
                "tests_failed": result.get("tests_failed", 0),
                "test_output": result.get("test_output") or "",
                "builder_error": result.get("error"),
                "used_fallback": result.get("used_fallback", False),
            },
            "build_iteration": build_i,
            "status": "running",
        }

    def verifier_node(state: WorkflowState) -> dict[str, Any]:
        build_i = int(state.get("build_iteration") or 1)
        result = run_verifier(
            state["request"],
            state.get("specification") or [],
            sandbox,
            emit,
            metrics,
            build_iteration=build_i,
        )
        prev = state.get("verification") or {}
        verification = {
            **prev,
            **result,
            "tests_passed": result.get("tests_passed", prev.get("tests_passed", 0)),
            "tests_failed": result.get("tests_failed", prev.get("tests_failed", 0)),
            "test_output": result.get("test_output") or prev.get("test_output") or "",
        }
        if result.get("overall_pass"):
            return {"verification": verification, "status": "success", "implementation_diff": result.get("diff") or state.get("implementation_diff")}
        # Attach spec feedback for Loop B
        if result.get("failure_class") == "spec":
            verification["spec_feedback"] = [
                {
                    "requirement_id": "VERIFIER",
                    "summary": "Verifier classified failure as specification problem",
                    "detail": result.get("rationale") or "Accepted spec conflicted with repo contracts after build",
                    "evidence": [],
                }
            ]
        return {
            "verification": verification,
            "implementation_diff": result.get("diff") or state.get("implementation_diff"),
            "status": "running",
        }

    def blocked_node(state: WorkflowState) -> dict[str, Any]:
        emit(
            "checkpoint_needed",
            {
                "reason": "max_spec_iterations",
                "message": (
                    f"Could not resolve a non-contradicted specification after "
                    f"{state.get('spec_iteration')} iterations. Awaiting human checkpoint."
                ),
                "conflicts": state.get("conflicts") or [],
                "spec_iteration": state.get("spec_iteration"),
            },
        )
        return {"status": "blocked"}

    def max_iter_node(state: WorkflowState) -> dict[str, Any]:
        return {"status": "max_iterations_reached"}

    def route_after_adversary(
        state: WorkflowState,
    ) -> Literal["accept_spec", "spec_detective", "blocked"]:
        conflicts = state.get("conflicts") or []
        iteration = int(state.get("spec_iteration") or 1)
        if not conflicts:
            return "accept_spec"
        # Checkpoint demo: permanent FORCE conflict enters BLOCKED after 2 passes
        # (still exercises loop-back once, without burning 4 full LLM revisions).
        force_block = any(c.get("requirement_id") == "FORCE" for c in conflicts)
        if iteration >= MAX_SPEC_ITERATIONS or (force_block and iteration >= 2):
            return "blocked"
        emit(
            "conflict_found",
            {
                "agent": "router",
                "action": "loop_back",
                "spec_iteration": iteration,
                "next_iteration": iteration + 1,
                "conflict_count": len(conflicts),
                "label": f"↻ Spec Detective (iteration {iteration + 1})",
                "conflicts": conflicts,
            },
        )
        return "spec_detective"

    def route_after_accept(
        state: WorkflowState,
    ) -> Literal["builder", "blocked"]:
        if state.get("status") == "blocked":
            return "blocked"
        return "builder"

    def route_after_verifier(
        state: WorkflowState,
    ) -> Literal["success_end", "builder", "spec_detective", "max_iter"]:
        verification = state.get("verification") or {}
        if verification.get("overall_pass") or state.get("status") == "success":
            return "success_end"
        failure = verification.get("failure_class") or "code"
        build_i = int(state.get("build_iteration") or 0)
        spec_i = int(state.get("spec_iteration") or 0)
        if failure == "code":
            if build_i >= MAX_BUILD_ITERATIONS:
                return "max_iter"
            emit(
                "conflict_found",
                {
                    "agent": "router",
                    "action": "retry_builder",
                    "build_iteration": build_i,
                    "label": f"↻ Builder (build {build_i + 1})",
                },
            )
            return "builder"
        # spec problem → full specification loop
        if spec_i >= MAX_SPEC_ITERATIONS:
            return "max_iter"
        # Seed conflicts for Spec Detective and clear accepted-only filter by keeping current spec as prior
        emit(
            "conflict_found",
            {
                "agent": "router",
                "action": "spec_problem_loop",
                "label": f"↻ Spec Detective (spec problem, iteration {spec_i + 1})",
                "detail": verification.get("rationale"),
            },
        )
        return "spec_detective"

    def success_end_node(state: WorkflowState) -> dict[str, Any]:
        return {"status": "success"}

    graph.add_node("explorer", explorer_node)
    graph.add_node("spec_detective", spec_detective_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("adversary", adversary_node)
    graph.add_node("accept_spec", accept_spec_node)
    graph.add_node("builder", builder_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("blocked", blocked_node)
    graph.add_node("max_iter", max_iter_node)
    graph.add_node("success_end", success_end_node)

    graph.set_entry_point("explorer")
    graph.add_edge("explorer", "spec_detective")
    graph.add_edge("spec_detective", "evidence")
    graph.add_edge("evidence", "adversary")
    graph.add_conditional_edges(
        "adversary",
        route_after_adversary,
        {
            "accept_spec": "accept_spec",
            "spec_detective": "spec_detective",
            "blocked": "blocked",
        },
    )
    graph.add_conditional_edges(
        "accept_spec",
        route_after_accept,
        {"builder": "builder", "blocked": "blocked"},
    )
    graph.add_edge("builder", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "success_end": "success_end",
            "builder": "builder",
            "spec_detective": "spec_detective",
            "max_iter": "max_iter",
        },
    )
    graph.add_edge("success_end", END)
    graph.add_edge("blocked", END)
    graph.add_edge("max_iter", END)
    return graph.compile()


def run_pipeline(
    repo_path: str,
    request: str,
    on_event: EventCallback | None = None,
    *,
    run_id: str | None = None,
    resume_state: dict[str, Any] | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    run_id = run_id or str(uuid.uuid4())
    recorder = TrajectoryRecorder(run_id, on_event)
    emit = recorder.emit
    metrics = AgentMetrics()

    initial: WorkflowState = {
        "request": request,
        "repo_path": repo_path,
        "explorer_findings": {},
        "specification": [],
        "evidence_report": {},
        "adversary_findings": [],
        "conflicts": [],
        "rejected_requirements": [],
        "revision_log": [],
        "implementation_diff": None,
        "verification": None,
        "spec_iteration": 0,
        "build_iteration": 0,
        "status": "running",
    }
    if resume_state:
        initial.update(resume_state)  # type: ignore[arg-type]

    try:
        with Sandbox.create(repo_path) as sandbox:
            app = build_graph(sandbox, emit, metrics)
            final = app.invoke(initial)
        runtime = time.perf_counter() - started
        verification = final.get("verification") or {}
        status = str(final.get("status") or "success")
        traj = recorder.finalize(
            status,
            {
                "runtime_seconds": round(runtime, 3),
                "token_cost": round(metrics.cost_usd, 6),
                "spec_iteration": final.get("spec_iteration"),
                "build_iteration": final.get("build_iteration"),
            },
        )
        checkpoint = None
        if status == "blocked":
            checkpoint = {
                "run_id": run_id,
                "status": "blocked",
                "message": "Human checkpoint required before continuing.",
                "conflicts": final.get("conflicts") or [],
                "rejected_requirements": final.get("rejected_requirements") or [],
                "spec_iteration": final.get("spec_iteration"),
            }
        else:
            emit(
                "run_completed",
                {
                    "status": status,
                    "requirements": len(final.get("specification") or []),
                    "runtime_seconds": round(runtime, 3),
                    "token_cost": round(metrics.cost_usd, 6),
                    "spec_iteration": final.get("spec_iteration"),
                    "build_iteration": final.get("build_iteration"),
                    "tests_passed": verification.get("tests_passed", 0),
                    "tests_failed": verification.get("tests_failed", 0),
                },
            )
        return PipelineResult(
            specification=final.get("specification") or [],
            rejected_requirements=list(final.get("rejected_requirements") or []),
            revision_log=list(final.get("revision_log") or []),
            explorer_findings=final.get("explorer_findings") or {},
            evidence_report=final.get("evidence_report") or {},
            adversary_findings=list(final.get("adversary_findings") or []),
            conflicts=list(final.get("conflicts") or []),
            implementation_diff=final.get("implementation_diff"),
            verification=verification,
            tests_passed=int(verification.get("tests_passed") or 0),
            tests_failed=int(verification.get("tests_failed") or 0),
            test_output=str(verification.get("test_output") or ""),
            spec_iteration=int(final.get("spec_iteration") or 0),
            build_iteration=int(final.get("build_iteration") or 0),
            runtime_seconds=round(runtime, 3),
            token_cost=round(metrics.cost_usd, 6),
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            status=status,
            error=sanitize_error_message(str(verification.get("builder_error")))
            if verification.get("builder_error")
            else None,
            trajectory_dir=str(traj),
            run_id=run_id,
            checkpoint=checkpoint,
        )
    except Exception as exc:
        runtime = time.perf_counter() - started
        safe = sanitize_error_message(str(exc))
        emit("run_completed", {"status": "failed", "error": safe})
        recorder.finalize("failed", {"error": safe})
        return PipelineResult(
            runtime_seconds=round(runtime, 3),
            token_cost=round(metrics.cost_usd, 6),
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            status="failed",
            error=safe,
            run_id=run_id,
            trajectory_dir=str(recorder.dir),
        )
