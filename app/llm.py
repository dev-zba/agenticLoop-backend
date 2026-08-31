"""Provider-agnostic LLM client with token/cost tracking.

Supports Anthropic, OpenAI, and Gemini. The first configured API key wins
(Anthropic → OpenAI → Gemini), unless MODEL_NAME implies a provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

# USD per 1M tokens: (input, output). Used for the cost-per-task metric.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-001": (0.10, 0.40),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def complete(prompt: str, system: str | None = None, timeout: float = 180.0) -> LLMResult:
    model = os.getenv("MODEL_NAME", "").strip()
    provider = _detect_provider(model)

    if provider == "anthropic":
        model = model or "claude-sonnet-4-6"
        return _anthropic(prompt, system, model, timeout)
    if provider == "openai":
        model = model or "gpt-4o"
        return _openai(prompt, system, model, timeout)
    if provider == "gemini":
        model = model or "gemini-2.5-flash"
        return _gemini(prompt, system, model, timeout)
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
    )


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES_PER_MILLION.get(model, (1.00, 5.00))
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


def _detect_provider(model: str) -> str | None:
    lowered = (model or "").lower()
    if lowered.startswith("claude") and os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if lowered.startswith("gpt") and os.getenv("OPENAI_API_KEY"):
        return "openai"
    if lowered.startswith("gemini") and os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return None


def _anthropic(prompt: str, system: str | None, model: str, timeout: float) -> LLMResult:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    payload: dict = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    usage = data.get("usage") or {}
    inp = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    return LLMResult(text=text, model=model, input_tokens=inp, output_tokens=out, cost_usd=estimate_cost(model, inp, out))


def _openai(prompt: str, system: str | None, model: str, timeout: float) -> LLMResult:
    api_key = os.environ["OPENAI_API_KEY"]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json={"model": model, "messages": messages, "temperature": 0.2},
        )
        resp.raise_for_status()
        data = resp.json()
    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage") or {}
    inp = int(usage.get("prompt_tokens") or 0)
    out = int(usage.get("completion_tokens") or 0)
    return LLMResult(text=text, model=model, input_tokens=inp, output_tokens=out, cost_usd=estimate_cost(model, inp, out))


def _gemini(prompt: str, system: str | None, model: str, timeout: float) -> LLMResult:
    api_key = os.environ["GEMINI_API_KEY"]
    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, params={"key": api_key}, json=payload)
        resp.raise_for_status()
        data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    text = "".join(part.get("text", "") for part in parts)
    usage = data.get("usageMetadata") or {}
    inp = int(usage.get("promptTokenCount") or 0)
    out = int(usage.get("candidatesTokenCount") or 0)
    return LLMResult(text=text, model=model, input_tokens=inp, output_tokens=out, cost_usd=estimate_cost(model, inp, out))
