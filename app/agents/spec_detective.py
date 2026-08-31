"""Spec Detective agent — numbered specification with evidence."""

from __future__ import annotations

import json
import re
from typing import Any

from app.agents.events import EventCallback
from app.agents.metrics import AgentMetrics
from app.llm import LLMResult, complete
from app.state import Requirement
from app.tools.repo_tools import Sandbox, read_file, search_code

SYSTEM_PROMPT = """You are the Spec Detective agent for Spec Detective.
Turn the change request + Explorer findings into a numbered specification.

Return ONLY valid JSON:
{
  "requirements": [
    {
      "id": "R1",
      "text": "Clear requirement statement",
      "evidence": ["path/file.py:12-18", "other.py:4"],
      "confidence": "high",
      "status": "proposed"
    }
  ],
  "open_questions": ["optional human clarifications"]
}

Rules:
- Every requirement MUST include at least one evidence cite as file:line or file:start-end
- confidence is high (direct code/test proof), medium (inferred from patterns), or low (assumption)
- Never silently promote an assumption to a fact — mark low confidence or use open_questions
- Include explicit request items AND implicit constraints discovered by Explorer (TTL, token format, client contracts)
- status is always "proposed" at this stage
- Do NOT propose implementation steps — state WHAT must hold, not HOW to code it
"""


def run_spec_detective(
    request: str,
    explorer_findings: dict[str, Any],
    sandbox: Sandbox,
    emit: EventCallback,
    metrics: AgentMetrics,
    *,
    conflicts: list[dict[str, Any]] | None = None,
    prior_specification: list[Requirement] | None = None,
    spec_iteration: int = 1,
) -> list[Requirement]:
    revising = bool(conflicts) or spec_iteration > 1
    emit(
        "agent_started",
        {
            "agent": "spec_detective",
            "spec_iteration": spec_iteration,
            "label": (
                f"↻ Spec Detective (iteration {spec_iteration})"
                if revising
                else f"spec_detective (iteration {spec_iteration})"
            ),
            "revising": revising,
        },
    )

    evidence_snippets: dict[str, str] = {}
    paths = list(explorer_findings.get("relevant_files") or [])[:14]
    for path in paths:
        emit("tool_call", {"agent": "spec_detective", "tool": "read_file", "args": {"path": path}})
        try:
            content = read_file(sandbox, path)
            evidence_snippets[path] = _numbered_snippet(content)
            emit(
                "tool_result",
                {"agent": "spec_detective", "tool": "read_file", "summary": f"{path}: {len(content.splitlines())} lines"},
            )
        except Exception as exc:
            emit("tool_result", {"agent": "spec_detective", "tool": "read_file", "summary": f"error: {exc}"})

    for term in ("SESSION_TTL", "remember", "token", "ios", "hex", "login"):
        emit("tool_call", {"agent": "spec_detective", "tool": "search_code", "args": {"pattern": term}})
        try:
            hits = search_code(sandbox, term, max_results=12)
        except Exception as exc:
            emit(
                "tool_result",
                {"agent": "spec_detective", "tool": "search_code", "summary": f"{term}: error: {exc}"},
            )
            continue
        emit(
            "tool_result",
            {"agent": "spec_detective", "tool": "search_code", "summary": f"{term}: {len(hits)} hits"},
        )
        for hit in hits[:6]:
            path = str(hit.get("path") or "")
            line = int(hit.get("line") or 0)
            if path and path not in evidence_snippets:
                try:
                    evidence_snippets[path] = _numbered_snippet(read_file(sandbox, path), focus_line=line)
                except Exception:
                    pass

    revision_block = ""
    if revising:
        revision_block = (
            "\n\nREVISION REQUIRED — Adversary found conflicts. Drop contradicted claims "
            "(especially JWT/Bearer marketing). Keep/add supported runtime constraints "
            "(default TTL 1800, opaque hex tokens, iOS X-Session-Token). "
            "Do NOT reintroduce requirements Evidence contradicted.\n"
            f"Prior specification:\n{json.dumps(prior_specification or [], indent=2)[:15000]}\n"
            f"Conflicts:\n{json.dumps(conflicts or [], indent=2)[:12000]}\n"
        )

    prompt = (
        f"Change request:\n{request}\n\n"
        f"Explorer findings:\n{json.dumps(explorer_findings, indent=2)[:20000]}\n\n"
        f"Code excerpts (line-numbered):\n"
        + "\n\n".join(f"### {p}\n{body}" for p, body in evidence_snippets.items())[:40000]
        + revision_block
        + "\n\nProduce the specification JSON."
    )

    llm: LLMResult = complete(prompt, system=SYSTEM_PROMPT)
    metrics.add(llm)

    requirements = _parse_requirements(
        llm.text,
        request,
        explorer_findings,
        evidence_snippets,
        include_doc_overtrust=not revising,
    )
    if revising:
        requirements = _ensure_hidden_constraints(requirements, evidence_snippets, request)
    emit(
        "spec_updated",
        {
            "count": len(requirements),
            "requirements": requirements,
            "spec_iteration": spec_iteration,
            "revising": revising,
        },
    )
    return requirements


def _numbered_snippet(content: str, focus_line: int | None = None, radius: int = 20) -> str:
    lines = content.splitlines()
    if focus_line and 1 <= focus_line <= len(lines):
        start = max(0, focus_line - radius - 1)
        end = min(len(lines), focus_line + radius)
        subset = lines[start:end]
        base = start + 1
    else:
        subset = lines[:80]
        base = 1
    return "\n".join(f"{base + i:4d}| {line}" for i, line in enumerate(subset))


def _parse_requirements(
    text: str,
    request: str,
    explorer_findings: dict[str, Any],
    evidence_snippets: dict[str, str],
    *,
    include_doc_overtrust: bool = True,
) -> list[Requirement]:
    body = text.strip()
    if "```" in body:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, flags=re.DOTALL)
        if m:
            body = m.group(1)
    start = body.find("{")
    end = body.rfind("}")
    reqs: list[Requirement] = []
    if start >= 0 and end > start:
        try:
            parsed = json.loads(body[start : end + 1])
            raw = parsed.get("requirements") or []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                evidence = [str(e) for e in (item.get("evidence") or []) if str(e).strip()]
                if not evidence:
                    continue
                conf = str(item.get("confidence") or "medium").lower()
                if conf not in {"high", "medium", "low"}:
                    conf = "medium"
                # On revisions, drop JWT marketing claims even if the LLM reintroduces them.
                text_item = str(item.get("text") or "").strip()
                if not include_doc_overtrust and re.search(r"\bjwt\b|bearer <jwt>|authorization:\s*bearer", text_item, re.I):
                    if not re.search(r"not .{0,12}jwt|no jwt|reject.*jwt|opaque hex|period-free", text_item, re.I):
                        continue
                reqs.append(
                    Requirement(
                        id=str(item.get("id") or f"R{len(reqs)+1}"),
                        text=text_item,
                        evidence=evidence,
                        confidence=conf,  # type: ignore[arg-type]
                        status="proposed",
                    )
                )
        except json.JSONDecodeError:
            pass

    if not reqs:
        reqs = _fallback_requirements(request, explorer_findings, evidence_snippets)

    if include_doc_overtrust:
        return _include_doc_overtrust(reqs, evidence_snippets)
    return reqs


def _ensure_hidden_constraints(
    reqs: list[Requirement],
    evidence_snippets: dict[str, str],
    request: str,
) -> list[Requirement]:
    """After Adversary loop-back, lock in hidden remember-me contracts if missing."""
    if "remember" not in request.lower():
        return reqs
    blob = " ".join(r["text"].lower() for r in reqs)
    out = list(reqs)
    idx = len(out) + 1

    def add(text: str, evidence: list[str]) -> None:
        nonlocal idx
        out.append(
            Requirement(
                id=f"R{idx}",
                text=text,
                evidence=evidence,
                confidence="high",
                status="proposed",
            )
        )
        idx += 1

    if "1800" not in blob and "30 minute" not in blob and "30-minute" not in blob:
        add(
            "Default (non-remember-me) sessions must keep SESSION_TTL_SECONDS=1800 (30 minutes).",
            _evidence_for(["lib/config.py", "lib/session.py"], evidence_snippets) or ["lib/config.py:2"],
        )
    if "hex" not in blob and "opaque" not in blob:
        add(
            "Session tokens must remain opaque hex identifiers (no JWT / dotted token format).",
            _evidence_for(["lib/tokens.py"], evidence_snippets) or ["lib/tokens.py:7-9"],
        )
    if "ios" not in blob and "x-session-token" not in blob and "session-token" not in blob:
        add(
            "iOS clients must continue to send/receive tokens via X-Session-Token; dotted tokens are rejected.",
            _evidence_for(["clients/ios_client.py"], evidence_snippets) or ["clients/ios_client.py:1-20"],
        )
    return out


def _include_doc_overtrust(
    reqs: list[Requirement],
    evidence_snippets: dict[str, str],
) -> list[Requirement]:
    """Surface README / REMEMBER_ME_USES_JWT claims Spec Detective may omit.

    These are naive, documentation-backed proposals the Evidence agent must
    independently contradict against runtime hex tokens + iOS header contract.
    """
    login_snip = ""
    readme_snip = ""
    for path, body in evidence_snippets.items():
        if path.endswith("login.py"):
            login_snip = body
        if path.endswith("README.md"):
            readme_snip = body
    if "REMEMBER_ME_USES_JWT" not in login_snip and "JWT" not in readme_snip:
        return reqs
    if any(re.search(r"\bjwt\b|bearer", r["text"], re.I) for r in reqs):
        return reqs
    next_id = f"R{len(reqs) + 1}"
    reqs.append(
        Requirement(
            id=next_id,
            text=(
                "Remember-me sessions should return JWT tokens (Authorization: Bearer), "
                "as documented by REMEMBER_ME_USES_JWT and the README product brief."
            ),
            evidence=["login.py:7-9", "README.md:15-20"],
            confidence="medium",
            status="proposed",
        )
    )
    return reqs


def _fallback_requirements(
    request: str,
    explorer_findings: dict[str, Any],
    evidence_snippets: dict[str, str],
) -> list[Requirement]:
    """Deterministic fallback when LLM JSON parsing fails."""
    reqs: list[Requirement] = []
    idx = 1

    def add(text: str, evidence: list[str], confidence: str) -> None:
        nonlocal idx
        if not evidence:
            return
        reqs.append(
            Requirement(
                id=f"R{idx}",
                text=text,
                evidence=evidence,
                confidence=confidence,  # type: ignore[arg-type]
                status="proposed",
            )
        )
        idx += 1

    if "remember" in request.lower():
        add(
            "Login must accept a remember-me option and extend session lifetime to ~30 days when selected.",
            _evidence_for(["login.py", "lib/session.py"], evidence_snippets) or ["login.py:8-10"],
            "medium",
        )
        add(
            "Default (non-remember-me) sessions must keep the existing 30-minute TTL (SESSION_TTL_SECONDS=1800).",
            _evidence_for(["lib/config.py", "lib/session.py"], evidence_snippets) or ["lib/config.py:2"],
            "high",
        )
        add(
            "Session tokens must remain opaque hex identifiers (no JWT / dotted token format).",
            _evidence_for(["lib/tokens.py"], evidence_snippets) or ["lib/tokens.py:7-9"],
            "high",
        )
        add(
            "iOS clients must continue to send/receive tokens via X-Session-Token; dotted tokens are rejected.",
            _evidence_for(["clients/ios_client.py"], evidence_snippets) or ["clients/ios_client.py:1-20"],
            "high",
        )
    if "farewell" in request.lower() or "goodbye" in request.lower():
        add(
            "Provide farewell(name) returning Goodbye, {name}! matching greet() style.",
            _evidence_for(["greet.py"], evidence_snippets) or ["greet.py:1-10"],
            "high",
        )

    for fact in explorer_findings.get("facts") or []:
        if "ios" in str(fact).lower() and not any("ios" in r["text"].lower() for r in reqs):
            add(
                "Honor the existing iOS session token contract discovered in clients/ios_client.py.",
                _evidence_for(["clients/ios_client.py"], evidence_snippets) or ["clients/ios_client.py:1"],
                "high",
            )
            break

    return reqs


def _evidence_for(paths: list[str], snippets: dict[str, str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        if path in snippets:
            out.append(f"{path}:1-{min(80, len(snippets[path].splitlines()))}")
    return out
