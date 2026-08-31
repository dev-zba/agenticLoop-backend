"""Event emitter type for SSE streaming."""

from __future__ import annotations

from typing import Callable

EventCallback = Callable[[str, dict], None]


def noop_event(_type: str, _data: dict) -> None:
    pass
