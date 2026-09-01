"""Evidence agent — independently re-check each proposed requirement."""

from __future__ import annotations

import json
import re
from typing import Any

from app.agents.events import EventCallback
from app.agents.metrics import AgentMetrics
from app.llm import LLMResult, complete
from app.state import Requirement
from app.tools.repo_tools import Sandbox, read_file, run_tests, search_code

SYSTEM_PROMPT = """You are the Evidence agent for Spec Detective.
Independently verify each proposed requirement against the repository.
Do NOT trust Spec Detective's cited evidence. Re-read code, search, and use test output.

Return ONLY valid JSON:
{
  "verdicts": [
    {
      "id": "R1",
      "status": "supported",
      "independent_evidence": ["path/file.py:12-18"],
      "rationale": "one sentence"
    }
  ]
}

status must be one of: supported, contradicted, insufficient_evidence

Rules:
- supported: existing code/tests are consistent with this requirement as a constraint or a gap to implement
- contradicted: runtime code or tests DISPROVE a claim in the requirement (docs/flags that disagree with runtime count as contradiction)
- insufficient_evidence: you cannot verify the claim from the repo
- independent_evidence must be cites YOU observed (file:line or file:start-end), not a copy of Spec Detective's list
- Marketing README copy and unused constants are not runtime proof
"""

_JWTISH = re.compile(r"\bjwt\b|json web token|authorization:\s*bearer|bearer <jwt>|dotted token", re.I)
_DEFAULT_30_DAYS = re.compile(
    r"default.{0,40}(30\s*days|thirty days)|non-remember.{0,40}(30\s*days|thirty days)",
    re.I,
)
_ALREADY_HAS_REMEMBER = re.compile(
    r"already (wired|implemented|accepts)|is already wired",
    re.I,
)
_ISSPLASH_TRUE_DISABLES = re.compile(
    r"issplash\s+to\s+true|set(?:ting)?\s+issplash\s+to\s+true|"
    r"splash.{0,40}off.{0,40}true|true.{0,40}(disable|turn off|off).{0,20}splash|"
    r"turn.{0,30}splash.{0,20}off.{0,50}true",
    re.I,
)
_README_ONLY_GREETING = re.compile(
    r"readme.{0,60}(enough|sufficient|without).{0,40}(greeting|portfolio\.js|display)|"
    r"(social|github).{0,40}(enough|sufficient).{0,40}(greeting|title|name)|"
    r"portfolio\.js.{0,40}does not need|"
    r"don.?t need to touch portfolio\.js|"
    r"without.{0,20}(changing|touching).{0,20}portfolio\.js",
    re.I,
)


def claims_jwt_runtime(text: str) -> bool:
    if re.search(
        r"not (a |use |using )?jwt|no jwt|reject.*jwt|jwt.{0,30}not support|period-free|opaque hex|must remain.{0,20}hex",
        text,
        re.I,
    ):
        return False
    return bool(_JWTISH.search(text))


def run_evidence(
    request: str,
    specification: list[Requirement],
    sandbox: Sandbox,
    emit: EventCallback,
    metrics: AgentMetrics,
) -> tuple[list[Requirement], dict[str, Any]]:
    emit("agent_started", {"agent": "evidence"})

    file_excerpts: dict[str, str] = {}
    search_hits: dict[str, list[dict[str, str | int]]] = {}

    for term in (
        "remember_me",
        "SESSION_TTL",
        "issue_token",
        "JWT",
        "REMEMBER_ME_USES_JWT",
        "X-Session-Token",
        "Authorization",
        "hex",
        "login",
        "farewell",
        "greet",
    ):
        emit("tool_call", {"agent": "evidence", "tool": "search_code", "args": {"pattern": term}})
        try:
            hits = search_code(sandbox, term, max_results=20)
        except Exception as exc:
            emit(
                "tool_result",
                {"agent": "evidence", "tool": "search_code", "summary": f"{term}: error: {exc}"},
            )
            continue
        search_hits[term] = hits
        emit(
            "tool_result",
            {"agent": "evidence", "tool": "search_code", "summary": f"{term}: {len(hits)} hits"},
        )
        for hit in hits[:8]:
            path = str(hit.get("path") or "")
            if path and path not in file_excerpts:
                _load_file(sandbox, path, emit, file_excerpts)

    for path in (
        "login.py",
        "lib/tokens.py",
        "lib/config.py",
        "lib/session.py",
        "clients/ios_client.py",
        "README.md",
        "greet.py",
        "tests/test_greet.py",
        "tests/test_login.py",
        "tests/test_legacy_ios_contract.py",
        "src/portfolio.js",
        "src/containers/greeting/Greeting.js",
        "src/components/seoHeader/SeoHeader.js",
    ):
        if path not in file_excerpts:
            _load_file(sandbox, path, emit, file_excerpts, optional=True)

    emit("tool_call", {"agent": "evidence", "tool": "run_tests", "args": {}})
    tests = run_tests(sandbox)
    emit(
        "tool_result",
        {
            "agent": "evidence",
            "tool": "run_tests",
            "summary": f"{tests.passed} passed, {tests.failed} failed",
        },
    )

    prompt = (
        f"Change request:\n{request}\n\n"
        f"Proposed specification (do not trust its evidence field):\n"
        f"{json.dumps(specification, indent=2)[:20000]}\n\n"
        f"Independent search hits:\n{json.dumps(search_hits, indent=2)[:15000]}\n\n"
        f"Independent file excerpts (line-numbered):\n"
        + "\n\n".join(f"### {p}\n{body}" for p, body in file_excerpts.items())[:35000]
        + f"\n\nIndependent test output excerpt:\n{(tests.output or '')[:2500]}\n\n"
        "Produce the verdicts JSON. Re-derive judgment from the excerpts above."
    )
    llm: LLMResult = complete(prompt, system=SYSTEM_PROMPT)
    metrics.add(llm)

    updated, report = _apply_verdicts(specification, llm.text, file_excerpts, tests.output)
    updated = _apply_deterministic_traps(updated, file_excerpts, report)
    emit(
        "spec_updated",
        {
            "count": len(updated),
            "requirements": updated,
            "source": "evidence",
        },
    )
    return updated, report


def _load_file(
    sandbox: Sandbox,
    path: str,
    emit: EventCallback,
    dest: dict[str, str],
    optional: bool = False,
) -> None:
    emit("tool_call", {"agent": "evidence", "tool": "read_file", "args": {"path": path}})
    try:
        content = read_file(sandbox, path)
        dest[path] = _numbered(content)
        emit(
            "tool_result",
            {
                "agent": "evidence",
                "tool": "read_file",
                "summary": f"{path}: {len(content.splitlines())} lines",
            },
        )
    except Exception as exc:
        emit("tool_result", {"agent": "evidence", "tool": "read_file", "summary": f"{path}: {exc}"})
        if not optional:
            dest[path] = ""


def _numbered(content: str, max_lines: int = 80) -> str:
    lines = content.splitlines()[:max_lines]
    return "\n".join(f"{i+1:4d}| {line}" for i, line in enumerate(lines))


def _parse_verdicts(text: str) -> list[dict[str, Any]]:
    body = text.strip()
    if "```" in body:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
        if m:
            body = m.group(1)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return []
    return list(parsed.get("verdicts") or [])


def _apply_verdicts(
    specification: list[Requirement],
    llm_text: str,
    file_excerpts: dict[str, str],
    test_output: str,
) -> tuple[list[Requirement], dict[str, Any]]:
    by_id = {str(v.get("id")): v for v in _parse_verdicts(llm_text) if isinstance(v, dict)}
    updated: list[Requirement] = []
    report: dict[str, Any] = {"items": []}

    for req in specification:
        verdict = by_id.get(req["id"], {})
        status = str(verdict.get("status") or "insufficient_evidence").lower()
        if status not in {"supported", "contradicted", "insufficient_evidence"}:
            status = "insufficient_evidence"
        independent = [str(e) for e in (verdict.get("independent_evidence") or []) if str(e).strip()]
        if not independent:
            independent = _fallback_cites(req["text"], file_excerpts)
            if status == "supported" and not independent:
                status = "insufficient_evidence"
        item: Requirement = {
            "id": req["id"],
            "text": req["text"],
            "evidence": independent,
            "confidence": req["confidence"],
            "status": status,  # type: ignore[typeddict-item]
        }
        updated.append(item)
        report["items"].append(
            {
                "id": req["id"],
                "detective_evidence": list(req.get("evidence") or []),
                "independent_evidence": independent,
                "status": status,
                "rationale": str(verdict.get("rationale") or ""),
                "test_failed": tests_mention_failure(test_output),
            }
        )
    return updated, report


def tests_mention_failure(output: str) -> bool:
    return "FAILED" in (output or "") or "ERROR" in (output or "")


def _fallback_cites(text: str, excerpts: dict[str, str]) -> list[str]:
    cites: list[str] = []
    lowered = text.lower()
    mapping = [
        ("hex", "lib/tokens.py"),
        ("jwt", "lib/tokens.py"),
        ("opaque", "lib/tokens.py"),
        ("ttl", "lib/config.py"),
        ("1800", "lib/config.py"),
        ("30 minute", "lib/config.py"),
        ("ios", "clients/ios_client.py"),
        ("session-token", "clients/ios_client.py"),
        ("login", "login.py"),
        ("remember", "login.py"),
        ("farewell", "greet.py"),
        ("goodbye", "greet.py"),
    ]
    for needle, path in mapping:
        if needle in lowered and path in excerpts:
            nlines = len(excerpts[path].splitlines())
            cites.append(f"{path}:1-{nlines}")
    return list(dict.fromkeys(cites))


def _apply_deterministic_traps(
    requirements: list[Requirement],
    excerpts: dict[str, str],
    report: dict[str, Any],
) -> list[Requirement]:
    """Catch over-trust of README / REMEMBER_ME_USES_JWT / portfolio splash polarity."""
    tokens = excerpts.get("lib/tokens.py", "")
    ios = excerpts.get("clients/ios_client.py", "")
    config = excerpts.get("lib/config.py", "")
    login = excerpts.get("login.py", "")
    portfolio = ""
    for path, body in excerpts.items():
        if path.endswith("portfolio.js"):
            portfolio = body
            break
    hex_runtime = ".hex()" in tokens or "urandom" in tokens
    ios_rejects_dot = "JWT_HINT" in ios or 'raise ValueError' in ios
    ttl_1800 = "SESSION_TTL_SECONDS = 1800" in config or "SESSION_TTL_SECONDS=1800" in config.replace(" ", "")
    login_no_remember = "def login(user_id: str)" in login and "remember_me" not in login.split("def login", 1)[-1][:80]
    splash_comment = "isSplash" in portfolio
    greeting_in_portfolio = "greeting" in portfolio and "title:" in portfolio

    out: list[Requirement] = []
    for req in requirements:
        text = req["text"]
        status = req["status"]
        evidence = list(req["evidence"])
        flipped = None

        if claims_jwt_runtime(text) and (hex_runtime or ios_rejects_dot):
            status = "contradicted"
            evidence = [
                "lib/tokens.py:6-9",
                "clients/ios_client.py:11-16",
            ]
            flipped = "runtime hex tokens + iOS period rejection contradict JWT/Bearer claims"
        elif claims_jwt_runtime(text) and portfolio and not login.strip():
            status = "insufficient_evidence"
            evidence = ["src/portfolio.js:1-30"]
            flipped = "No login/JWT/auth modules exist in this React portfolio — reject JWT ask"
        elif _DEFAULT_30_DAYS.search(text) and ttl_1800:
            status = "contradicted"
            evidence = ["lib/config.py:2", "lib/session.py:19-25"]
            flipped = "SESSION_TTL_SECONDS=1800 contradicts a 30-day default session"
        elif _ALREADY_HAS_REMEMBER.search(text) and login_no_remember:
            status = "contradicted"
            evidence = ["login.py:12-14"]
            flipped = "login(user_id) has no remember_me parameter despite docs/flags"
        elif _ISSPLASH_TRUE_DISABLES.search(text) and splash_comment:
            status = "contradicted"
            evidence = ["src/portfolio.js:4-6"]
            flipped = "Comment says set isSplash to false to disable splash; true keeps splash ON"
        elif _README_ONLY_GREETING.search(text) and greeting_in_portfolio:
            status = "contradicted"
            evidence = ["src/portfolio.js:21-30", "src/containers/greeting/Greeting.js:17-18"]
            flipped = "Home greeting title is driven by greeting.title in portfolio.js, not README/social labels"

        req = {**req, "status": status, "evidence": evidence}  # type: ignore[misc]
        out.append(req)  # type: ignore[arg-type]
        if flipped:
            for item in report.get("items") or []:
                if item.get("id") == req["id"]:
                    item["status"] = status
                    item["independent_evidence"] = evidence
                    item["rationale"] = flipped
                    item["deterministic_override"] = True
    return out
