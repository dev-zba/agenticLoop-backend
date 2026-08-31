"""Single-LLM baseline: one prompt, apply diff in sandbox, run tests.

Intentionally no Explorer / Evidence / Adversary / agents. See PROJECT_BRIEF.md §2.6.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.llm import LLMResult, complete
from app.tools.repo_tools import (
    Sandbox,
    SandboxError,
    apply_diff,
    extract_unified_diff,
    git_diff,
    list_files,
    read_file,
    run_tests,
    write_file,
)

EventCallback = Callable[[str, dict], None]

STOPWORDS = {
    "the", "a", "an", "to", "add", "for", "and", "or", "of", "in", "on", "that",
    "this", "with", "from", "it", "be", "is", "are", "so", "can", "they", "them",
    "their", "want", "please", "just", "into", "by", "as", "at", "we", "you",
    "your", "our", "not", "but", "if", "then", "else", "when", "make", "need",
}

MAX_FILES = 16
MAX_FILE_BYTES = 8_000
MAX_TOTAL_BYTES = 40_000

SYSTEM_PROMPT = """You are a careful software engineer applying one change to an existing repository.
Implement ONLY the user's request. Prefer the smallest patch that satisfies it.
Return a unified diff that `git apply` can consume (diff --git / --- / +++ hunks).
Do not wrap the diff in commentary. Do not invent a new architecture if an existing module already owns the behavior.
"""


@dataclass
class BaselineResult:
    diff: str
    tests_passed: int
    tests_failed: int
    runtime_seconds: float
    token_cost: float
    test_output: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw_model_output: str = ""
    files_in_context: list[str] = field(default_factory=list)
    error: str | None = None


def run_baseline(
    repo_path: str,
    request: str,
    on_event: EventCallback | None = None,
) -> BaselineResult:
    started = time.perf_counter()

    def emit(event_type: str, data: dict | None = None) -> None:
        if on_event:
            on_event(event_type, data or {})

    emit("started", {"repo_path": repo_path, "request": request})

    with Sandbox.create(repo_path) as sandbox:
        context, selected = gather_context(sandbox, request)
        prompt = _build_prompt(request, context)

        llm: LLMResult | None = None
        apply_error: str | None = None
        try:
            llm = complete(prompt, system=SYSTEM_PROMPT)
            _apply_model_output(sandbox, llm.text)
        except Exception as exc:
            apply_error = str(exc)

        diff = ""
        try:
            diff = git_diff(sandbox)
        except SandboxError:
            diff = ""

        tests = run_tests(sandbox)
        runtime = time.perf_counter() - started
        result = BaselineResult(
            diff=diff,
            tests_passed=tests.passed,
            tests_failed=tests.failed,
            runtime_seconds=round(runtime, 3),
            token_cost=round(llm.cost_usd, 6) if llm else 0.0,
            test_output=tests.output,
            model=llm.model if llm else "",
            input_tokens=llm.input_tokens if llm else 0,
            output_tokens=llm.output_tokens if llm else 0,
            raw_model_output=llm.text if llm else "",
            files_in_context=selected,
            error=apply_error,
        )
        emit(
            "completed",
            {
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "runtime_seconds": result.runtime_seconds,
                "token_cost": result.token_cost,
            },
        )
        return result


def gather_context(sandbox: Sandbox, request: str) -> tuple[str, list[str]]:
    """README + top-level files + path-keyword matches. Not the whole repo."""
    keywords = _keywords(request)
    all_files = list_files(sandbox)
    selected: list[str] = []

    def consider(path: str) -> None:
        if path not in selected:
            selected.append(path)

    for name in ("README.md", "README", "readme.md", "README.txt"):
        if name in all_files:
            consider(name)

    for path in all_files:
        if "/" not in path:
            consider(path)

    if keywords:
        for path in all_files:
            lowered = path.lower()
            if any(kw in lowered for kw in keywords):
                consider(path)

    chunks: list[str] = []
    total = 0
    included: list[str] = []
    for path in selected[:MAX_FILES]:
        try:
            content = read_file(sandbox, path)
        except SandboxError:
            continue
        if len(content) > MAX_FILE_BYTES:
            content = content[:MAX_FILE_BYTES] + "\n... [truncated]\n"
        piece = f"## {path}\n```\n{content}\n```\n"
        if total + len(piece) > MAX_TOTAL_BYTES:
            break
        chunks.append(piece)
        included.append(path)
        total += len(piece)

    tree = "\n".join(all_files[:200])
    header = f"# File tree\n```\n{tree}\n```\n\n# Selected files\n"
    return header + "\n".join(chunks), included


def _apply_model_output(sandbox: Sandbox, text: str) -> None:
    diff = extract_unified_diff(text)
    if "diff --git" in diff or diff.lstrip().startswith("--- "):
        apply_diff(sandbox, diff)
        return

    files = _extract_file_rewrites(text)
    if files:
        for path, content in files:
            write_file(sandbox, path, content)
        return

    apply_diff(sandbox, text)


def _extract_file_rewrites(text: str) -> list[tuple[str, str]]:
    """Fallback: ```path blocks or 'FILE: path' fences."""
    found: list[tuple[str, str]] = []
    for match in re.finditer(
        r"(?:FILE:\s*|###\s*File:\s*)([^\n]+)\n```(?:\w+)?\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        found.append((match.group(1).strip(), match.group(2)))
    if found:
        return found
    for match in re.finditer(r"```([a-zA-Z0-9_./\\-]+)\n(.*?)```", text, flags=re.DOTALL):
        path = match.group(1).strip()
        if path in {"diff", "patch", "text", "json", "python", "py", "md"}:
            continue
        if "/" in path or Path(path).suffix:
            found.append((path, match.group(2)))
    return found


def _build_prompt(request: str, context: str) -> str:
    return (
        "Existing repository context (heuristic subset, not the full repo):\n\n"
        f"{context}\n\n"
        "Change request:\n"
        f"{request}\n\n"
        "Return a unified diff implementing this change against the files above. "
        "Keep existing public contracts unless the request explicitly changes them.\n"
    )


def _keywords(request: str) -> set[str]:
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", request.lower())
    return {w for w in words if w not in STOPWORDS}
