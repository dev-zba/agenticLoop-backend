"""Strip secrets from strings before they reach logs or the UI."""

from __future__ import annotations

import re

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"key=[^&\s'\"]+", re.I), "key=[REDACTED]"),
    (re.compile(r"AIza[0-9A-Za-z\-_]+"), "[REDACTED]"),
    (re.compile(r"AQ\.[0-9A-Za-z\-_]+"), "[REDACTED]"),
    (re.compile(r"sk-ant-[0-9A-Za-z\-_]+"), "[REDACTED]"),
    (re.compile(r"sk-[0-9A-Za-z\-_]+"), "[REDACTED]"),
    (re.compile(r"Bearer\s+[0-9A-Za-z\-_.]+", re.I), "Bearer [REDACTED]"),
]


def sanitize_error_message(message: str) -> str:
    if not message:
        return message
    out = message
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out
