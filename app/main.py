"""FastAPI app: baseline + pipeline runs + SSE + human checkpoint."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.baseline import BaselineResult, run_baseline
from app.graph import PipelineResult, run_pipeline
from app.security import sanitize_error_message
from app.tools.repo_tools import SandboxError, apply_diff_to_repo

load_dotenv()

app = FastAPI(title="Spec Detective", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS: dict[str, dict[str, Any]] = {}
EVAL_ROOT = Path(__file__).resolve().parents[2] / "eval" / "results"


class RunRequest(BaseModel):
    repo_path: str = Field(..., min_length=1)
    request: str = Field(..., min_length=1)
    # Explicit human approval at run start to copy sandbox diff onto the real repo
    apply_on_success: bool = True


class CheckpointRequest(BaseModel):
    action: Literal["clarify", "accept_assumption", "stop"]
    note: str = ""


class RunResponse(BaseModel):
    id: str
    mode: str = "baseline"
    diff: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    runtime_seconds: float = 0.0
    token_cost: float = 0.0
    test_output: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    files_in_context: list[str] = []
    error: str | None = None
    status: str = "completed"
    specification: list[dict[str, Any]] = []
    explorer_findings: dict[str, Any] = {}
    evidence_report: dict[str, Any] = {}
    adversary_findings: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    rejected_requirements: list[dict[str, Any]] = []
    revision_log: list[dict[str, Any]] = []
    verification: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    trajectory_dir: str | None = None
    spec_iteration: int = 0
    build_iteration: int = 0
    applied_to_repo: bool = False
    apply_message: str | None = None


class RunStartedResponse(BaseModel):
    id: str
    mode: Literal["baseline", "pipeline"]
    status: str = "running"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _append_event(run_id: str, event_type: str, data: dict) -> None:
    run = RUNS.get(run_id)
    if not run:
        return
    run["events"].append(
        {
            "type": event_type,
            "data": data,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )
    # Open the checkpoint gate as soon as the graph asks — don't wait for
    # invoke() to return (UI used to race and POST while status was still running).
    if event_type == "checkpoint_needed":
        run["status"] = "blocked"
        run["awaiting_checkpoint"] = True
        run["checkpoint"] = {
            "run_id": run_id,
            "status": "blocked",
            "message": data.get("message") or "Human checkpoint required.",
            "reason": data.get("reason"),
            "conflicts": data.get("conflicts") or [],
        }


def _pipeline_payload(run_id: str, result: PipelineResult) -> dict[str, Any]:
    return {
        "id": run_id,
        "mode": "pipeline",
        "status": result.status,
        "runtime_seconds": result.runtime_seconds,
        "token_cost": result.token_cost,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "specification": result.specification,
        "explorer_findings": result.explorer_findings,
        "evidence_report": result.evidence_report,
        "adversary_findings": result.adversary_findings,
        "conflicts": result.conflicts,
        "rejected_requirements": result.rejected_requirements,
        "revision_log": result.revision_log,
        "verification": result.verification,
        "checkpoint": result.checkpoint,
        "trajectory_dir": result.trajectory_dir,
        "diff": result.implementation_diff or "",
        "tests_passed": result.tests_passed,
        "tests_failed": result.tests_failed,
        "test_output": result.test_output,
        "spec_iteration": result.spec_iteration,
        "build_iteration": result.build_iteration,
        "error": result.error,
        "applied_to_repo": False,
        "apply_message": None,
    }


async def _execute_pipeline(
    run_id: str,
    repo_path: str,
    request: str,
    *,
    resume_state: dict[str, Any] | None = None,
) -> None:
    def on_event(event_type: str, data: dict) -> None:
        _append_event(run_id, event_type, data)

    try:
        result: PipelineResult = await asyncio.to_thread(
            run_pipeline,
            repo_path,
            request,
            on_event,
            run_id=run_id,
            resume_state=resume_state,
        )
        payload = _pipeline_payload(run_id, result)
        RUNS[run_id]["status"] = result.status
        RUNS[run_id]["result"] = payload
        RUNS[run_id]["checkpoint"] = result.checkpoint
        if result.error:
            RUNS[run_id]["error"] = result.error

        if (
            result.status == "success"
            and RUNS[run_id].get("apply_on_success")
            and (result.implementation_diff or "").strip()
        ):
            try:
                msg = apply_diff_to_repo(repo_path, result.implementation_diff or "")
                payload["applied_to_repo"] = True
                payload["apply_message"] = msg
                RUNS[run_id]["result"] = payload
                _append_event(run_id, "applied_to_repo", {"ok": True, "message": msg})
            except Exception as exc:
                safe = sanitize_error_message(str(exc))
                payload["applied_to_repo"] = False
                payload["apply_message"] = safe
                RUNS[run_id]["result"] = payload
                _append_event(run_id, "applied_to_repo", {"ok": False, "message": safe})

        if result.status == "blocked":
            _append_event(
                run_id,
                "run_completed",
                {"status": "blocked", "checkpoint": True, "message": (result.checkpoint or {}).get("message")},
            )
    except Exception as exc:
        safe = sanitize_error_message(str(exc))
        RUNS[run_id]["status"] = "failed"
        RUNS[run_id]["error"] = safe
        _append_event(run_id, "run_completed", {"status": "failed", "error": safe})


@app.post("/runs")
async def create_run(
    body: RunRequest,
    mode: Literal["baseline", "pipeline"] = Query(default="baseline"),
):
    run_id = str(uuid.uuid4())
    RUNS[run_id] = {
        "id": run_id,
        "mode": mode,
        "status": "running",
        "repo_path": body.repo_path,
        "request": body.request,
        "apply_on_success": body.apply_on_success,
        "events": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if mode == "pipeline":
        asyncio.create_task(_execute_pipeline(run_id, body.repo_path, body.request))
        return RunStartedResponse(id=run_id, mode="pipeline", status="running")

    def on_event(event_type: str, data: dict) -> None:
        _append_event(run_id, event_type, data)

    try:
        result: BaselineResult = await asyncio.to_thread(
            run_baseline, body.repo_path, body.request, on_event
        )
    except Exception as exc:
        safe = sanitize_error_message(str(exc))
        RUNS[run_id]["status"] = "failed"
        RUNS[run_id]["error"] = safe
        _append_event(run_id, "run_completed", {"status": "failed", "error": safe})
        raise HTTPException(status_code=500, detail=safe) from exc

    payload = RunResponse(
        id=run_id,
        mode="baseline",
        diff=result.diff,
        tests_passed=result.tests_passed,
        tests_failed=result.tests_failed,
        runtime_seconds=result.runtime_seconds,
        token_cost=result.token_cost,
        test_output=result.test_output,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        files_in_context=result.files_in_context,
        error=result.error,
        status="completed",
    )
    if body.apply_on_success and (result.diff or "").strip():
        try:
            msg = apply_diff_to_repo(body.repo_path, result.diff)
            payload = payload.model_copy(update={"applied_to_repo": True, "apply_message": msg})
            _append_event(run_id, "applied_to_repo", {"ok": True, "message": msg})
        except Exception as exc:
            safe = sanitize_error_message(str(exc))
            payload = payload.model_copy(update={"applied_to_repo": False, "apply_message": safe})
            _append_event(run_id, "applied_to_repo", {"ok": False, "message": safe})
    RUNS[run_id]["status"] = "completed"
    RUNS[run_id]["result"] = payload.model_dump()
    _append_event(
        run_id,
        "run_completed",
        {"status": "completed", "mode": "baseline", "tests_passed": result.tests_passed},
    )
    return payload


@app.post("/runs/{run_id}/apply")
async def apply_run_to_repo(run_id: str) -> dict[str, Any]:
    """Human-approved: copy the run's sandbox diff onto the original repo."""
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    result = run.get("result") or {}
    diff = result.get("diff") or ""
    if not str(diff).strip():
        raise HTTPException(status_code=400, detail="run has no diff to apply")
    if run.get("status") not in {"success", "completed", "max_iterations_reached"}:
        raise HTTPException(status_code=400, detail="run is not in an applyable state")
    try:
        msg = apply_diff_to_repo(run["repo_path"], str(diff))
    except SandboxError as exc:
        raise HTTPException(status_code=400, detail=sanitize_error_message(str(exc))) from exc
    result = dict(result)
    result["applied_to_repo"] = True
    result["apply_message"] = msg
    run["result"] = result
    _append_event(run_id, "applied_to_repo", {"ok": True, "message": msg})
    return {"id": run_id, "applied_to_repo": True, "apply_message": msg}


@app.post("/runs/{run_id}/checkpoint")
async def post_checkpoint(run_id: str, body: CheckpointRequest) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if run.get("status") != "blocked" and not run.get("awaiting_checkpoint"):
        raise HTTPException(
            status_code=400,
            detail="run is not awaiting a checkpoint (wait until status is blocked, then try again)",
        )

    run["awaiting_checkpoint"] = False
    _append_event(
        run_id,
        "checkpoint_response",
        {"action": body.action, "note": body.note},
    )

    if body.action == "stop":
        run["status"] = "stopped"
        result = run.get("result") or {}
        result["status"] = "stopped"
        result["checkpoint_response"] = body.model_dump()
        result["checkpoint"] = run.get("checkpoint")
        run["result"] = result
        _append_event(run_id, "run_completed", {"status": "stopped", "action": "stop"})
        return {"id": run_id, "status": "stopped", "action": "stop"}

    repo_path = run["repo_path"]
    request = run["request"]
    if body.action == "clarify":
        # Note optional — empty clarify still retries and clears FORCE_SPEC_CONFLICT
        if body.note.strip():
            request = f"{request}\n\nHuman clarification: {body.note.strip()}"
        else:
            request = (
                f"{request}\n\nHuman clarification: proceed with supported requirements only; "
                "ignore unresolved adversarial FORCE conflicts and implement farewell."
            )
        run["request"] = request

    if body.action == "accept_assumption":
        request = (
            f"{request}\n\nHuman checkpoint: accept remaining assumptions and proceed to implement "
            f"only non-contradicted supported requirements. Note: {body.note or 'none'}"
        )
        run["request"] = request

    run["status"] = "running"
    run["checkpoint"] = None
    asyncio.create_task(_execute_pipeline(run_id, repo_path, request))
    return {"id": run_id, "status": "running", "action": body.action}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/runs/{run_id}/results")
def get_run_results(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    status = run.get("status")
    if status == "running":
        return {"id": run_id, "status": "running", "mode": run.get("mode", "pipeline")}
    result = dict(run.get("result") or {})
    # Early BLOCKED (before invoke finished writing full result)
    if status == "blocked":
        result.setdefault("status", "blocked")
        result.setdefault("checkpoint", run.get("checkpoint"))
        result.setdefault("mode", "pipeline")
    return {"id": run_id, "status": status, **result}


@app.get("/eval/results/latest")
def eval_results_latest() -> dict[str, Any]:
    path = EVAL_ROOT / "harness_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no harness_summary.json yet — run eval/harness.py")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> EventSourceResponse:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run not found")

    async def generator():
        sent = 0
        idle_rounds = 0
        saw_run_completed = False
        while True:
            run = RUNS.get(run_id)
            if not run:
                break
            events = run["events"]
            while sent < len(events):
                event = events[sent]
                sent += 1
                idle_rounds = 0
                if event["type"] == "run_completed":
                    saw_run_completed = True
                yield {
                    "event": event["type"],
                    "data": json.dumps(event),
                }
            # Do not close on status=blocked alone — checkpoint_needed sets that
            # early, before invoke() finishes and appends run_completed. Closing
            # early made the UI report "SSE connection lost".
            if saw_run_completed and sent >= len(events):
                break
            if run["status"] in {
                "completed",
                "success",
                "failed",
                "spec_conflict",
                "implementation_failed",
                "max_iterations_reached",
                "stopped",
                "verification_failed",
            } and sent >= len(events):
                break
            idle_rounds += 1
            if idle_rounds > 1200:
                break
            await asyncio.sleep(0.25)

    return EventSourceResponse(generator())
