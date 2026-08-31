# Spec Detective — Backend

Python/FastAPI service for **Spec Detective**: sandboxed repository tools, a single-LLM **baseline** runner, and (planned) a 6-agent LangGraph pipeline.

> **Phase 1 (current):** Repository tools layer + baseline implementation + REST/SSE API. Agents (Explorer → Verifier) are defined in [PROJECT_BRIEF.md](../PROJECT_BRIEF.md) and come in later phases.

## What this service does

1. **Sandboxed repo tools** — All mutating operations run inside a temporary git worktree. The original repo path is never modified.
2. **Baseline runner** — One LLM call with heuristic file context → unified diff → apply in sandbox → run tests.
3. **HTTP API** — `POST /runs` to start a job; `GET /runs/{id}/events` for SSE progress (started/completed events today).

This baseline is the comparison point for measuring improvement once the full agent pipeline ships.

## Tech stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.141.1 | HTTP + SSE |
| uvicorn | 0.52.4 | ASGI server |
| GitPython | 3.1.61 | Worktree sandboxing |
| pytest | 9.1.1 | Backend + target-repo tests |
| httpx | 0.28.1 | LLM provider calls |

Optional (later phases): `langgraph`, `langchain-anthropic` — see PROJECT_BRIEF.md.

## Prerequisites

| Tool | Why |
|------|-----|
| Python 3.11+ | Backend runtime |
| `git` | Worktree sandboxing, diffs |
| `ripgrep` (`rg`) | Fast code search (used by repo tools) |
| An LLM API key | Powers the baseline LLM call |

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your API key(s)
```

### Environment variables

Copy `.env.example` → `.env`. Set **at least one** provider key:

```bash
# One of:
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...

# Model for the chosen provider
MODEL_NAME=gemini-2.5-flash   # or claude-sonnet-4-6, gpt-4o, etc.

LOG_COST=true
```

Provider is auto-detected from `MODEL_NAME`, or falls back to the first configured key (Anthropic → OpenAI → Gemini).

## Run locally

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

## API

### `GET /health`

Returns `{"status": "ok"}`.

### `POST /runs`

Start a baseline run.

**Request body:**

```json
{
  "repo_path": "/absolute/path/to/repo",
  "request": "Add passwordless login using magic links"
}
```

**Response:**

```json
{
  "id": "uuid",
  "diff": "unified diff text",
  "tests_passed": 12,
  "tests_failed": 0,
  "runtime_seconds": 4.2,
  "token_cost": 0.0012,
  "test_output": "...",
  "model": "gemini-2.5-flash",
  "input_tokens": 8000,
  "output_tokens": 1200,
  "files_in_context": ["src/auth.py", "..."],
  "error": null,
  "status": "completed"
}
```

### `GET /runs/{run_id}`

Fetch run metadata and stored result.

### `GET /runs/{run_id}/events`

Server-Sent Events stream (`started`, `completed`, etc.) for live progress.

## Baseline flow

```
repo_path + request
    → Sandbox.create (git worktree in temp dir)
    → gather_context (heuristic file selection from request keywords)
    → LLM complete (single call, unified diff output)
    → apply_diff in sandbox
    → run_tests in sandbox
    → return diff + metrics
```

The original repo on disk is never written to. See `app/baseline.py` and `app/tools/repo_tools.py`.

## Project structure

```
backend/
├── app/
│   ├── main.py           # FastAPI routes + SSE
│   ├── baseline.py       # Single-LLM baseline runner
│   ├── llm.py            # Anthropic / OpenAI / Gemini client + cost tracking
│   └── tools/
│       └── repo_tools.py # Sandbox, list/read/write, diff, tests, search
├── tests/
│   └── test_sandbox.py   # Sandbox isolation guarantees
├── requirements.txt
└── .env.example
```

## Tests

```bash
source .venv/bin/activate
pytest
```

Key guarantee under test: mutating sandbox operations do **not** touch the original repository.

## Security notes

- **Never commit `.env`** — it contains API keys.
- All file writes and test runs happen inside a **git worktree** under a temp directory.
- The full 6-agent pipeline will add a human-approval checkpoint before merging sandbox changes (see PROJECT_BRIEF.md §3.4).

## Related

- [Frontend README](../frontend/README.md) — Next.js UI that calls this API
- [PROJECT_BRIEF.md](../PROJECT_BRIEF.md) — Agent architecture, evaluation dataset, submission rubric
