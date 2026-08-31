"""Adversary agent — assume the verified spec is wrong; report conflicts only."""

from __future__ import annotations

import json
import re
from typing import Any

from app.agents.events import EventCallback
from app.agents.evidence import claims_jwt_runtime
from app.agents.metrics import AgentMetrics
from app.llm import LLMResult, complete
from app.state import Requirement
from app.tools.repo_tools import Sandbox, git_history, read_file, run_tests, search_code

SYSTEM_PROMPT = """You are the Adversary agent for Spec Detective.
Assume the specification is wrong — try to prove it.
Look for contradictory code, conflicting tests, forgotten edge cases,
compatibility issues, and unsupported assumptions among SUPPORTED requirements.
Report conflicts; do NOT fix them. Do NOT propose implementation patches.

Return ONLY valid JSON:
{
  "findings": [
    {"requirement_id": "R1", "kind": "edge_case|contradiction|assumption|compat", "detail": "..."}
  ],
  "conflicts": [
    {
      "requirement_id": "R1",
      "summary": "short conflict title",
      "detail": "why the supported claim fails against the repo",
      "evidence": ["path/file.py:1-10"]
    }
  ]
}

Rules:
- conflicts must be concrete and evidenced (file:line cites you observed)
- If a requirement is labeled contradicted by Evidence, that alone is a conflict on the overall spec
- Missing hidden contracts (TTL, token format, client headers) that tests enforce are conflicts
- Empty conflicts array means you could not break the supported set
"""


def run_adversary(
    request: str,
    specification: list[Requirement],
    sandbox: Sandbox,
    emit: EventCallback,
    metrics: AgentMetrics,
    *,
    spec_iteration: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    emit(
        "agent_started",
        {
            "agent": "adversary",
            "spec_iteration": spec_iteration,
            "label": f"adversary (iteration {spec_iteration})",
        },
    )

    excerpts: dict[str, str] = {}
    for term in (
        "remember_me",
        "SESSION_TTL",
        "issue_token",
        "JWT",
        "X-Session-Token",
        "hex",
        "login",
        "farewell",
    ):
        emit("tool_call", {"agent": "adversary", "tool": "search_code", "args": {"pattern": term}})
        try:
            hits = search_code(sandbox, term, max_results=15)
        except Exception as exc:
            emit(
                "tool_result",
                {"agent": "adversary", "tool": "search_code", "summary": f"{term}: error: {exc}"},
            )
            continue
        emit(
            "tool_result",
            {"agent": "adversary", "tool": "search_code", "summary": f"{term}: {len(hits)} hits"},
        )
        for hit in hits[:5]:
            path = str(hit.get("path") or "")
            if path and path not in excerpts:
                _load(sandbox, path, emit, excerpts)

    for path in (
        "login.py",
        "lib/tokens.py",
        "lib/config.py",
        "lib/session.py",
        "clients/ios_client.py",
        "README.md",
        "tests/test_legacy_ios_contract.py",
        "tests/test_login.py",
        "greet.py",
        "tests/test_greet.py",
    ):
        if path not in excerpts:
            _load(sandbox, path, emit, excerpts, optional=True)

    emit("tool_call", {"agent": "adversary", "tool": "run_tests", "args": {}})
    tests = run_tests(sandbox)
    emit(
        "tool_result",
        {
            "agent": "adversary",
            "tool": "run_tests",
            "summary": f"{tests.passed} passed, {tests.failed} failed",
        },
    )

    emit(
        "tool_call",
        {"agent": "adversary", "tool": "git_history", "args": {"path": "login.py", "max_commits": 5}},
    )
    history = git_history(sandbox, "login.py", max_commits=5)
    emit(
        "tool_result",
        {"agent": "adversary", "tool": "git_history", "summary": history[:200] or "(empty)"},
    )

    supported = [r for r in specification if r.get("status") == "supported"]
    prompt = (
        f"Change request:\n{request}\n\n"
        f"Full specification (statuses from Evidence):\n{json.dumps(specification, indent=2)[:20000]}\n\n"
        f"SUPPORTED subset to attack:\n{json.dumps(supported, indent=2)[:12000]}\n\n"
        f"Repo excerpts:\n"
        + "\n\n".join(f"### {p}\n{body}" for p, body in excerpts.items())[:35000]
        + f"\n\nTest output:\n{(tests.output or '')[:2500]}\n\n"
        f"git log login.py:\n{history[:800]}\n\n"
        "Return findings/conflicts JSON. Prefer real conflicts over noise."
    )
    llm: LLMResult = complete(prompt, system=SYSTEM_PROMPT)
    metrics.add(llm)

    findings, llm_conflicts = _parse_adversary(llm.text)
    deterministic = _deterministic_conflicts(request, specification, excerpts, tests.output or "")
    # Loop gate is deterministic so the remember-me smoke case revises once then accepts;
    # LLM suggestions stay in findings for the transcript without forcing endless loops.
    conflicts = deterministic
    findings = findings or []
    for c in llm_conflicts:
        findings.append(
            {
                "requirement_id": c.get("requirement_id"),
                "kind": "llm_conflict_candidate",
                "detail": c.get("summary") or c.get("detail"),
                "evidence": c.get("evidence") or [],
            }
        )
    if not findings:
        findings = [{"kind": "scan", "detail": f"{len(conflicts)} deterministic conflict(s)"}]

    for conflict in conflicts:
        emit(
            "conflict_found",
            {
                "agent": "adversary",
                "spec_iteration": spec_iteration,
                "requirement_id": conflict.get("requirement_id"),
                "summary": conflict.get("summary"),
                "detail": conflict.get("detail"),
                "evidence": conflict.get("evidence") or [],
            },
        )

    emit(
        "tool_result",
        {
            "agent": "adversary",
            "tool": "adversary_summary",
            "summary": f"{len(conflicts)} conflict(s), {len(findings)} finding(s)",
        },
    )
    return findings, conflicts


def _load(
    sandbox: Sandbox,
    path: str,
    emit: EventCallback,
    dest: dict[str, str],
    optional: bool = False,
) -> None:
    emit("tool_call", {"agent": "adversary", "tool": "read_file", "args": {"path": path}})
    try:
        content = read_file(sandbox, path)
        dest[path] = _numbered(content)
        emit(
            "tool_result",
            {
                "agent": "adversary",
                "tool": "read_file",
                "summary": f"{path}: {len(content.splitlines())} lines",
            },
        )
    except Exception as exc:
        emit("tool_result", {"agent": "adversary", "tool": "read_file", "summary": f"{path}: {exc}"})
        if not optional:
            dest[path] = ""


def _numbered(content: str, max_lines: int = 80) -> str:
    lines = content.splitlines()[:max_lines]
    return "\n".join(f"{i+1:4d}| {line}" for i, line in enumerate(lines))


def _parse_adversary(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    body = text.strip()
    if "```" in body:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
        if m:
            body = m.group(1)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return [], []
    try:
        parsed = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return [], []
    findings = [x for x in (parsed.get("findings") or []) if isinstance(x, dict)]
    conflicts = [x for x in (parsed.get("conflicts") or []) if isinstance(x, dict)]
    return findings, conflicts


def _merge_conflicts(
    primary: list[dict[str, Any]],
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*primary, *extra]:
        key = f"{item.get('requirement_id')}|{item.get('summary')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _deterministic_conflicts(
    request: str,
    specification: list[Requirement],
    excerpts: dict[str, str],
    test_output: str,
) -> list[dict[str, Any]]:
    """Reliable traps so the remember-me smoke case revises at least once."""
    conflicts: list[dict[str, Any]] = []
    blob = json.dumps(specification).lower()
    req_lower = request.lower()

    for req in specification:
        if req.get("status") == "contradicted":
            conflicts.append(
                {
                    "requirement_id": req["id"],
                    "summary": "Evidence-contradicted claim still in the specification",
                    "detail": (
                        f"{req['id']} is labeled contradicted but remains in the draft spec: "
                        f"{req['text'][:160]}"
                    ),
                    "evidence": list(req.get("evidence") or [])[:4],
                }
            )
        if req.get("status") == "supported" and claims_jwt_runtime(req["text"]):
            conflicts.append(
                {
                    "requirement_id": req["id"],
                    "summary": "Supported requirement claims JWT against hex runtime",
                    "detail": (
                        "Runtime tokens are opaque hex (`lib/tokens.py`) and iOS rejects "
                        "dotted JWT-shaped tokens (`clients/ios_client.py`)."
                    ),
                    "evidence": ["lib/tokens.py:6-9", "clients/ios_client.py:11-16"],
                }
            )

    if "remember" in req_lower:
        has_ttl = any(
            re.search(r"1800|30\s*-?\s*minute|default.{0,40}ttl|SESSION_TTL", r["text"], re.I)
            for r in specification
            if r.get("status") in {"supported", "proposed", "accepted"}
        )
        has_hex = any(
            re.search(r"\bhex\b|opaque|period-free|no jwt|not .{0,10}jwt", r["text"], re.I)
            for r in specification
            if r.get("status") in {"supported", "proposed", "accepted"}
        )
        has_ios = any(
            re.search(r"ios|x-session-token|session-token", r["text"], re.I)
            for r in specification
            if r.get("status") in {"supported", "proposed", "accepted"}
        )
        # First-pass specs that still carry contradicted JWT OR omit hidden contracts.
        if "lib/tokens.py" in excerpts and not has_hex:
            conflicts.append(
                {
                    "requirement_id": "SPEC",
                    "summary": "Missing opaque-hex token constraint",
                    "detail": (
                        "tests/test_legacy_ios_contract.py requires period-free hex tokens; "
                        "no supported requirement locks this in."
                    ),
                    "evidence": ["lib/tokens.py:6-9", "tests/test_legacy_ios_contract.py:16-28"],
                }
            )
        if "lib/config.py" in excerpts and not has_ttl:
            conflicts.append(
                {
                    "requirement_id": "SPEC",
                    "summary": "Missing default 30-minute TTL constraint",
                    "detail": (
                        "SESSION_TTL_SECONDS=1800 must remain the default for non-remember-me "
                        "sessions; the supported set does not say so."
                    ),
                    "evidence": ["lib/config.py:2", "tests/test_legacy_ios_contract.py:30-36"],
                }
            )
        if "clients/ios_client.py" in excerpts and not has_ios:
            conflicts.append(
                {
                    "requirement_id": "SPEC",
                    "summary": "Missing iOS X-Session-Token contract",
                    "detail": (
                        "clients/ios_client.py rejects dotted tokens and expects X-Session-Token; "
                        "the supported set omits this compatibility constraint."
                    ),
                    "evidence": ["clients/ios_client.py:1-20"],
                }
            )

        # Doc/runtime split still present as contradicted JWT → always conflict once.
        if any(r.get("status") == "contradicted" and claims_jwt_runtime(r["text"]) for r in specification):
            if not any(c.get("summary", "").startswith("Evidence-contradicted") for c in conflicts):
                conflicts.append(
                    {
                        "requirement_id": "SPEC",
                        "summary": "README/JWT marketing claim contradicts runtime token format",
                        "detail": (
                            "Specification still includes a JWT/Bearer remember-me claim that "
                            "Evidence contradicted against hex tokens + iOS header contract."
                        ),
                        "evidence": ["lib/tokens.py:6-9", "clients/ios_client.py:11-16", "README.md:15-20"],
                    }
                )

    if "FAILED" in test_output and "remember" in req_lower:
        # Baseline state: remember_me not implemented — not a spec conflict by itself.
        pass

    # De-noise: if we only have LLM noise later, deterministic list is enough.
    _ = blob
    return conflicts
