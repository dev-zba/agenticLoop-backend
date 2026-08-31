"""FastAPI app: baseline run + SSE events."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.baseline import BaselineResult, run_baseline

load_dotenv()

app = FastAPI(title="Spec Detective", version="0.1.0")
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

# In-memory run store. Fine for the local skeleton.
RUNS: dict[str, dict[str, Any]] = {}


class RunRequest(BaseModel):
    repo_path: str = Field(..., min_length=1)
    request: str = Field(..., min_length=1)


class RunResponse(BaseModel):
    id: str
    diff: str
    tests_passed: int
    tests_failed: int
    runtime_seconds: float
    token_cost: float
    test_output: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    files_in_context: list[str] = []
    error: str | None = None
    status: str = "completed"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunResponse)
async def create_run(body: RunRequest) -> RunResponse:
    run_id = str(uuid.uuid4())
    RUNS[run_id] = {
        "id": run_id,
        "status": "running",
        "repo_path": body.repo_path,
        "request": body.request,
        "events": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    def on_event(event_type: str, data: dict) -> None:
        RUNS[run_id]["events"].append(
            {
                "type": event_type,
                "data": data,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )

    try:
        result: BaselineResult = await asyncio.to_thread(
            run_baseline, body.repo_path, body.request, on_event
        )
    except Exception as exc:
        RUNS[run_id]["status"] = "failed"
        RUNS[run_id]["error"] = str(exc)
        RUNS[run_id]["events"].append(
            {
                "type": "completed",
                "data": {"error": str(exc)},
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = RunResponse(
        id=run_id,
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
    return payload


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> EventSourceResponse:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="run not found")

    async def generator():
        # Replay whatever we have (started / completed for this phase).
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
            if run["status"] in {"completed", "failed"} and sent >= len(events):
                break
            idle_rounds += 1
            if idle_rounds > 600:
                break
            await asyncio.sleep(0.25)

    return EventSourceResponse(generator())
