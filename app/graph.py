"""LangGraph orchestration: Explorer → Spec Detective → Evidence → Adversary ↔ loop → Builder."""

from __future__ import annotations

import time
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
from app.security import sanitize_error_message
from app.state import MAX_SPEC_ITERATIONS, Requirement, WorkflowState
from app.tools.repo_tools import Sandbox


@dataclass
class PipelineResult:
    specification: list[Requirement] = field(default_factory=list)
    explorer_findings: dict[str, Any] = field(default_factory=dict)
    evidence_report: dict[str, Any] = field(default_factory=dict)
    adversary_findings: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    implementation_diff: str | None = None
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


def build_graph(sandbox: Sandbox, emit: EventCallback, metrics: AgentMetrics):
    graph = StateGraph(WorkflowState)

    def explorer_node(state: WorkflowState) -> dict[str, Any]:
        findings = run_explorer(state["request"], sandbox, emit, metrics)
        return {"explorer_findings": findings, "status": "running"}

    def spec_detective_node(state: WorkflowState) -> dict[str, Any]:
        iteration = int(state.get("spec_iteration") or 0) + 1
        conflicts = list(state.get("conflicts") or [])
        prior = list(state.get("specification") or [])
        spec = run_spec_detective(
            state["request"],
            state.get("explorer_findings") or {},
            sandbox,
            emit,
            metrics,
            conflicts=conflicts if iteration > 1 else None,
            prior_specification=prior if iteration > 1 else None,
            spec_iteration=iteration,
        )
        return {
            "specification": spec,
            "spec_iteration": iteration,
            "conflicts": [],  # clear resolved; adversary will re-fill
            "status": "running",
        }

    def evidence_node(state: WorkflowState) -> dict[str, Any]:
        spec, report = run_evidence(
            state["request"],
            state.get("specification") or [],
            sandbox,
            emit,
            metrics,
        )
        return {
            "specification": spec,
            "evidence_report": report,
            "status": "running",
        }

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
        return {
            "adversary_findings": findings,
            "conflicts": conflicts,
            "status": "running",
        }

    def accept_spec_node(state: WorkflowState) -> dict[str, Any]:
        accepted: list[Requirement] = []
        for req in state.get("specification") or []:
            status = req.get("status")
            if status == "supported":
                accepted.append({**req, "status": "accepted"})  # type: ignore[misc]
            elif status == "accepted":
                accepted.append(req)
            # Drop contradicted / insufficient from the implementable set.
        emit(
            "spec_updated",
            {
                "count": len(accepted),
                "requirements": accepted,
                "source": "accepted",
                "spec_iteration": state.get("spec_iteration"),
            },
        )
        return {"specification": accepted, "status": "running", "conflicts": []}

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
        return {
            "implementation_diff": result.get("diff") or "",
            "verification": {
                "tests_passed": result.get("tests_passed", 0),
                "tests_failed": result.get("tests_failed", 0),
                "test_output": result.get("test_output") or "",
                "builder_error": result.get("error"),
                "used_fallback": result.get("used_fallback", False),
            },
            "build_iteration": build_i,
            "status": result.get("status") or "success",
        }

    def spec_conflict_node(state: WorkflowState) -> dict[str, Any]:
        return {"status": "spec_conflict"}

    def route_after_adversary(
        state: WorkflowState,
    ) -> Literal["accept_spec", "spec_detective", "spec_conflict"]:
        conflicts = state.get("conflicts") or []
        iteration = int(state.get("spec_iteration") or 1)
        if not conflicts:
            return "accept_spec"
        if iteration >= MAX_SPEC_ITERATIONS:
            return "spec_conflict"
        emit(
            "conflict_found",
            {
                "agent": "router",
                "action": "loop_back",
                "spec_iteration": iteration,
                "next_iteration": iteration + 1,
                "conflict_count": len(conflicts),
                "label": f"↻ Spec Detective (iteration {iteration + 1})",
            },
        )
        return "spec_detective"

    graph.add_node("explorer", explorer_node)
    graph.add_node("spec_detective", spec_detective_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("adversary", adversary_node)
    graph.add_node("accept_spec", accept_spec_node)
    graph.add_node("builder", builder_node)
    graph.add_node("spec_conflict", spec_conflict_node)

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
            "spec_conflict": "spec_conflict",
        },
    )
    graph.add_edge("accept_spec", "builder")
    graph.add_edge("builder", END)
    graph.add_edge("spec_conflict", END)
    return graph.compile()


def run_pipeline(
    repo_path: str,
    request: str,
    on_event: EventCallback | None = None,
) -> PipelineResult:
    started = time.perf_counter()
    emit = on_event or noop_event
    metrics = AgentMetrics()

    initial: WorkflowState = {
        "request": request,
        "repo_path": repo_path,
        "explorer_findings": {},
        "specification": [],
        "evidence_report": {},
        "adversary_findings": [],
        "conflicts": [],
        "implementation_diff": None,
        "verification": None,
        "spec_iteration": 0,
        "build_iteration": 0,
        "status": "running",
    }

    try:
        with Sandbox.create(repo_path) as sandbox:
            app = build_graph(sandbox, emit, metrics)
            final = app.invoke(initial)
        runtime = time.perf_counter() - started
        verification = final.get("verification") or {}
        status = str(final.get("status") or "success")
        emit(
            "run_completed",
            {
                "status": status,
                "requirements": len(final.get("specification") or []),
                "runtime_seconds": round(runtime, 3),
                "token_cost": round(metrics.cost_usd, 6),
                "spec_iteration": final.get("spec_iteration"),
                "tests_passed": verification.get("tests_passed", 0),
                "tests_failed": verification.get("tests_failed", 0),
            },
        )
        return PipelineResult(
            specification=final.get("specification") or [],
            explorer_findings=final.get("explorer_findings") or {},
            evidence_report=final.get("evidence_report") or {},
            adversary_findings=list(final.get("adversary_findings") or []),
            conflicts=list(final.get("conflicts") or []),
            implementation_diff=final.get("implementation_diff"),
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
        )
    except Exception as exc:
        runtime = time.perf_counter() - started
        safe = sanitize_error_message(str(exc))
        emit("run_completed", {"status": "failed", "error": safe})
        return PipelineResult(
            runtime_seconds=round(runtime, 3),
            token_cost=round(metrics.cost_usd, 6),
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            status="failed",
            error=safe,
        )
