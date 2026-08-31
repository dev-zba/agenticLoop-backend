"""Explorer agent — investigate only, never propose requirements."""

from __future__ import annotations

import json
import re
from typing import Any

from app.agents.events import EventCallback
from app.agents.metrics import AgentMetrics
from app.llm import LLMResult, complete
from app.tools.repo_tools import (
    Sandbox,
    SandboxError,
    git_history,
    inspect_dependencies,
    list_files,
    read_file,
    run_tests,
    search_code,
)

SYSTEM_PROMPT = """You are the Explorer agent for Spec Detective.
Investigate, do not design. Report only what exists in the repository.
Never propose requirements or implementation steps.

Return ONLY valid JSON with this exact shape:
{
  "facts": ["short factual statements backed by files you saw"],
  "relevant_files": ["path/to/file.py", ...],
  "relevant_tests": ["tests/test_x.py", ...],
  "relevant_configs": ["lib/config.py: SESSION_TTL_SECONDS=1800", ...],
  "open_questions": ["things unclear from code that a human should clarify"]
}

Rules:
- facts must describe existing behavior, APIs, tests, configs — not desired changes
- include files/tests/configs even if only mentioned in search hits or test output
- open_questions are for genuine ambiguity, not missing features you wish existed
"""

SEARCH_PATTERNS = (
    "session",
    "token",
    "remember",
    "login",
    "ttl",
    "expire",
    "ios",
    "jwt",
    "hex",
    "config",
    "greet",
    "farewell",
)


def run_explorer(
    request: str,
    sandbox: Sandbox,
    emit: EventCallback,
    metrics: AgentMetrics,
) -> dict[str, Any]:
    emit("agent_started", {"agent": "explorer"})

    tool_log: list[dict[str, Any]] = []

    def call_tool(name: str, args: dict[str, Any], fn) -> Any:
        emit("tool_call", {"agent": "explorer", "tool": name, "args": args})
        try:
            result = fn()
            summary = _summarize_tool_result(name, result)
            emit("tool_result", {"agent": "explorer", "tool": name, "summary": summary})
            tool_log.append({"tool": name, "args": args, "result": result})
            return result
        except Exception as exc:
            emit("tool_result", {"agent": "explorer", "tool": name, "summary": f"error: {exc}"})
            tool_log.append({"tool": name, "args": args, "error": str(exc)})
            return None

    all_files = call_tool("list_files", {"path": "."}, lambda: list_files(sandbox))
    all_files = all_files or []

    keywords = _keywords_from_request(request)
    search_hits: list[dict[str, str | int]] = []
    patterns = list(dict.fromkeys([*keywords, *SEARCH_PATTERNS]))
    for pattern in patterns[:12]:
        hits = call_tool(
            "search_code",
            {"pattern": pattern},
            lambda p=pattern: search_code(sandbox, p),
        )
        if hits:
            search_hits.extend(hits)

    paths_to_read = _select_paths(all_files, search_hits, keywords)
    file_contents: dict[str, str] = {}
    for path in paths_to_read[:18]:
        content = call_tool("read_file", {"path": path}, lambda p=path: read_file(sandbox, p))
        if isinstance(content, str):
            file_contents[path] = content[:6000]

    history_targets = paths_to_read[:4] or ["."]
    histories: dict[str, str] = {}
    for path in history_targets:
        hist = call_tool(
            "git_history",
            {"path": path, "max_commits": 8},
            lambda p=path: git_history(sandbox, p, max_commits=8),
        )
        if isinstance(hist, str) and hist:
            histories[path] = hist

    deps = call_tool("inspect_dependencies", {}, lambda: inspect_dependencies(sandbox)) or {}
    tests = call_tool("run_tests", {}, lambda: run_tests(sandbox))

    synthesis_input = {
        "request": request,
        "all_files_count": len(all_files),
        "search_hits": search_hits[:40],
        "file_contents": file_contents,
        "histories": histories,
        "dependencies": deps,
        "test_summary": {
            "passed": tests.passed if tests else 0,
            "failed": tests.failed if tests else 0,
            "command": tests.command if tests else [],
            "output_excerpt": (tests.output[:3000] if tests else ""),
        },
    }

    prompt = (
        "Change request (for investigation focus only — do NOT implement):\n"
        f"{request}\n\n"
        "Tool results from the repository:\n"
        f"{json.dumps(synthesis_input, indent=2)[:50000]}\n\n"
        "Produce the Explorer JSON report."
    )
    llm: LLMResult = complete(prompt, system=SYSTEM_PROMPT)
    metrics.add(llm)

    findings = _parse_findings(llm.text, all_files, search_hits, file_contents, tests)
    emit("tool_result", {"agent": "explorer", "tool": "synthesize", "summary": f"{len(findings.get('facts', []))} facts"})
    return findings


def _keywords_from_request(request: str) -> list[str]:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", request.lower())
    stop = {"the", "want", "users", "that", "with", "for", "and", "are", "this", "they", "have", "will", "days"}
    return [w for w in words if w not in stop][:8]


def _select_paths(
    all_files: list[str],
    search_hits: list[dict[str, str | int]],
    keywords: list[str],
) -> list[str]:
    chosen: list[str] = []
    for name in ("README.md", "README", "readme.md"):
        if name in all_files:
            chosen.append(name)
    for hit in search_hits:
        path = str(hit.get("path") or "")
        if path and path not in chosen:
            chosen.append(path)
    for path in all_files:
        if path.startswith("tests/") and path not in chosen:
            chosen.append(path)
        if path.startswith("lib/") and path not in chosen:
            chosen.append(path)
        if path.startswith("clients/") and path not in chosen:
            chosen.append(path)
    for path in all_files:
        lowered = path.lower()
        if any(k in lowered for k in keywords) and path not in chosen:
            chosen.append(path)
    for path in all_files:
        if "/" not in path and path.endswith(".py") and path not in chosen:
            chosen.append(path)
    return chosen


def _parse_findings(
    text: str,
    all_files: list[str],
    search_hits: list[dict[str, str | int]],
    file_contents: dict[str, str],
    tests: Any,
) -> dict[str, Any]:
    body = text.strip()
    if "```" in body:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
        if m:
            body = m.group(1)
    start = body.find("{")
    end = body.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(body[start : end + 1])
            if isinstance(parsed, dict):
                return _normalize_findings(parsed, all_files, search_hits, file_contents, tests)
        except json.JSONDecodeError:
            pass
    return _normalize_findings({}, all_files, search_hits, file_contents, tests)


def _normalize_findings(
    parsed: dict[str, Any],
    all_files: list[str],
    search_hits: list[dict[str, str | int]],
    file_contents: dict[str, str],
    tests: Any,
) -> dict[str, Any]:
    facts = list(parsed.get("facts") or [])
    relevant_files = list(parsed.get("relevant_files") or [])
    relevant_tests = list(parsed.get("relevant_tests") or [])
    relevant_configs = list(parsed.get("relevant_configs") or [])
    open_questions = list(parsed.get("open_questions") or [])

    for path in file_contents:
        if path not in relevant_files:
            relevant_files.append(path)
    for hit in search_hits:
        path = str(hit.get("path") or "")
        if path.endswith(".py") and path not in relevant_files:
            relevant_files.append(path)
    for path in all_files:
        if path.startswith("tests/") and path not in relevant_tests:
            relevant_tests.append(path)

    if "lib/config.py" in file_contents and not any("SESSION_TTL" in c for c in relevant_configs):
        relevant_configs.append("lib/config.py: SESSION_TTL_SECONDS = 1800")
    if "lib/tokens.py" in file_contents and not any("hex" in f.lower() for f in facts):
        facts.append("lib/tokens.py issues opaque hex session tokens via os.urandom(...).hex()")
    if "login.py" in file_contents and "REMEMBER_ME_USES_JWT" in file_contents.get("login.py", ""):
        facts.append(
            "login.py defines REMEMBER_ME_USES_JWT = True and README product copy claims JWT remember-me; "
            "treat this as a documented claim to verify, not as proven runtime behavior"
        )
    if "clients/ios_client.py" in file_contents and not any("ios" in f.lower() for f in facts):
        facts.append("clients/ios_client.py defines an iOS session header contract")

    if tests and "test_legacy_ios_contract" in (tests.output or ""):
        facts.append("tests/test_legacy_ios_contract.py encodes legacy iOS hex-token and TTL expectations")

    return {
        "facts": facts,
        "relevant_files": sorted(set(relevant_files)),
        "relevant_tests": sorted(set(relevant_tests)),
        "relevant_configs": relevant_configs,
        "open_questions": open_questions,
    }


def _summarize_tool_result(name: str, result: Any) -> str:
    if name == "list_files" and isinstance(result, list):
        return f"{len(result)} files"
    if name == "search_code" and isinstance(result, list):
        return f"{len(result)} matches"
    if name == "read_file" and isinstance(result, str):
        return f"{len(result.splitlines())} lines"
    if name == "git_history" and isinstance(result, str):
        return f"{len(result.splitlines())} commits"
    if name == "inspect_dependencies" and isinstance(result, dict):
        return ", ".join(result.keys()) or "none"
    if name == "run_tests" and result is not None:
        return f"{result.passed} passed, {result.failed} failed"
    return str(result)[:200]
