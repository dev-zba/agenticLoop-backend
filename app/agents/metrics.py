"""Per-run LLM token/cost aggregation for agent calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm import LLMResult


@dataclass
class AgentMetrics:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, result: LLMResult) -> None:
        self.model = result.model or self.model
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.cost_usd += result.cost_usd
        self.calls += 1
