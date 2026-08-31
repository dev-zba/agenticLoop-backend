"""FastAPI app: baseline + pipeline runs + SSE events."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.baseline import BaselineResult, run_baseline
from app.graph import PipelineResult, run_pipeline
from app.security import sanitize_error_message

load_dotenv()

app = FastAPI(title="Spec Detective", version="0.2.0")
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


class RunRequest(BaseModel):
    repo_path: str = Field(..., min_length=1)
    request: str = Field(..., min_length=1)


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
    spec_iteration: int = 0
    build_iteration: int = 0


class RunStartedResponse(BaseModel):
    id: str
    mode: Literal["baseline", "pipeline"]
    status: str = "running"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _append_event(run_id: str, event_type: str, data: dict) -> None:
    RUNS[run_id]["events"].append(
        {
            "type": event_type,
            "data": data,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )


async def _execute_pipeline(run_id: str, repo_path: str, request: str) -> None:
    def on_event(event_type: str, data: dict) -> None:
        _append_event(run_id, event_type, data)

    try:
        result: PipelineResult = await asyncio.to_thread(run_pipeline, repo_path, request, on_event)
        payload = {
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
            "diff": result.implementation_diff or "",
            "tests_passed": result.tests_passed,
            "tests_failed": result.tests_failed,
            "test_output": result.test_output,
            "spec_iteration": result.spec_iteration,
            "build_iteration": result.build_iteration,
            "error": result.error,
        }
        RUNS[run_id]["status"] = result.status
        RUNS[run_id]["result"] = payload
        if result.error:
            RUNS[run_id]["error"] = result.error
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
    RUNS[run_id]["status"] = "completed"
    RUNS[run_id]["result"] = payload.model_dump()
    _append_event(
        run_id,
        "run_completed",
        {"status": "completed", "mode": "baseline", "tests_passed": result.tests_passed},
    )
    return payload


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
    if run.get("status") == "running":
        return {"id": run_id, "status": "running", "mode": run.get("mode", "pipeline")}
    result = run.get("result") or {}
    return {"id": run_id, "status": run.get("status"), **result}


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> EventSourceResponse:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run not found")

    async def generator():
        sent = 0
        idle_rounds = 0
        while True:
            run = RUNS.get(run_id)
            if not run:
                break
            events = run["events"]
            while sent < len(events):
                event = events[sent]
                sent += 1
                idle_rounds = 0
                yield {
                    "event": event["type"],
                    "data": json.dumps(event),
                }
            if run["status"] in {
                "completed",
                "success",
                "failed",
                "spec_conflict",
                "implementation_failed",
                "max_iterations_reached",
                "blocked",
                "verification_failed",
            } and sent >= len(events):
                break
            idle_rounds += 1
            if idle_rounds > 1200:
                break
            await asyncio.sleep(0.25)

    return EventSourceResponse(generator())
