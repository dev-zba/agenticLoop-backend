# backend/app/state.py
from typing import Literal, TypedDict

class Requirement(TypedDict):
    id: str
    text: str
    evidence: list[str]          # e.g. ["auth/verification.py:54-91"]
    confidence: Literal["high", "medium", "low"]
    status: Literal["proposed", "supported", "contradicted", "insufficient_evidence", "accepted"]

class WorkflowState(TypedDict):
    request: str
    repo_path: str
    explorer_findings: dict
    specification: list[Requirement]
    evidence_report: dict
    adversary_findings: list[dict]
    conflicts: list[dict]
    implementation_diff: str | None
    verification: dict | None
    spec_iteration: int
    build_iteration: int
    status: Literal[
        "running", "blocked", "spec_conflict",
        "implementation_failed", "verification_failed",
        "max_iterations_reached", "success"
    ]

MAX_SPEC_ITERATIONS = 4
MAX_BUILD_ITERATIONS = 3
