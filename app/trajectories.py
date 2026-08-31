"""Per-run trajectory capture — one file per agent for submission §1.6."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# agenticLoop/trajectories (sibling of backend/)
ROOT = Path(__file__).resolve().parents[2]
TRAJECTORIES = ROOT / "trajectories"

EventCallback = Callable[[str, dict], None]


class TrajectoryRecorder:
    """Wraps an emit callback and appends structured events per agent."""

    def __init__(self, run_id: str, on_event: EventCallback | None = None):
        self.run_id = run_id
        self.on_event = on_event
        self.dir = TRAJECTORIES / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        meta = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def emit(self, event_type: str, data: dict | None = None) -> None:
        data = data or {}
        agent = str(data.get("agent") or _infer_agent(event_type, data) or "system")
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data,
        }
        with self._lock:
            self._buffers.setdefault(agent, []).append(entry)
            self._flush_agent(agent)
        if self.on_event:
            self.on_event(event_type, data)

    def _flush_agent(self, agent: str) -> None:
        path = self.dir / f"{agent}.json"
        payload = {
            "agent": agent,
            "run_id": self.run_id,
            "events": self._buffers.get(agent, []),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def finalize(self, status: str, extra: dict | None = None) -> Path:
        summary = {
            "run_id": self.run_id,
            "status": status,
            "agents": sorted(self._buffers.keys()),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            **(extra or {}),
        }
        path = self.dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return self.dir


def _infer_agent(event_type: str, data: dict) -> str | None:
    if event_type in {"run_completed", "checkpoint_needed"}:
        return "system"
    if event_type == "conflict_found" and data.get("action") == "loop_back":
        return "router"
    return None
