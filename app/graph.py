"""LangGraph orchestration: Explorer → Spec Detective."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.events import EventCallback, noop_event
from app.agents.explorer import run_explorer
from app.agents.metrics import AgentMetrics
from app.agents.spec_detective import run_spec_detective
from app.security import sanitize_error_message
from app.state import Requirement, WorkflowState
from app.tools.repo_tools import Sandbox


@dataclass
class PipelineResult:
    specification: list[Requirement] = field(default_factory=list)
    explorer_findings: dict[str, Any] = field(default_factory=dict)
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
        return {"explorer_findings": findings}

    def spec_detective_node(state: WorkflowState) -> dict[str, Any]:
        spec = run_spec_detective(
            state["request"],
            state.get("explorer_findings") or {},
            sandbox,
            emit,
            metrics,
        )
        return {"specification": spec, "status": "success", "spec_iteration": 1}

    graph.add_node("explorer", explorer_node)
    graph.add_node("spec_detective", spec_detective_node)
    graph.set_entry_point("explorer")
    graph.add_edge("explorer", "spec_detective")
    graph.add_edge("spec_detective", END)
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
        emit(
            "run_completed",
            {
                "status": final.get("status", "success"),
                "requirements": len(final.get("specification") or []),
                "runtime_seconds": round(runtime, 3),
                "token_cost": round(metrics.cost_usd, 6),
            },
        )
        return PipelineResult(
            specification=final.get("specification") or [],
            explorer_findings=final.get("explorer_findings") or {},
            runtime_seconds=round(runtime, 3),
            token_cost=round(metrics.cost_usd, 6),
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            status=str(final.get("status") or "success"),
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
