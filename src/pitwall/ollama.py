"""Run a bundle's GGUF in Ollama on CPU and read the router's answer as probabilities.

- CPU only (num_gpu 0): the GPU belongs to Gearbox's tier S model; a 0.6B router must be
  cheap enough to run beside it on the CPU.
- Raw mode: Ollama sends our text as is (no chat template), the same text the model was
  trained on (pitwall.textformat).
- One token, with its top logprobs: P(S), P(M), P(L) are the probabilities of " S",
  " M", " L", renormalised over the three. Ollama only reports the top-k tokens, so a
  tier outside them counts as probability 0; if none of the three is there, the router
  falls back to M and the answer is flagged `no_answer`, so a broken model shows up in
  the numbers instead of hiding behind the fallback.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from pitwall.metrics import TIERS
from pitwall.textformat import ANSWERS

TOP_LOGPROBS = 20  # the most Ollama returns
NUM_CTX = 2048  # longest training example is ~920 tokens
FALLBACK = "M"
_TOKEN_TO_TIER = {text: tier for tier, text in ANSWERS.items()}


class OllamaError(RuntimeError):
    pass


def model_name(bundle_name: str) -> str:
    """Ollama model name for a bundle: lowercase letters, digits and hyphens only."""
    return "pitwall-" + re.sub(r"[^a-z0-9]+", "-", bundle_name.lower()).strip("-")


def modelfile(gguf: Path) -> str:
    return (
        f'FROM "{gguf.resolve().as_posix()}"\n'
        "PARAMETER temperature 0\n"
        f"PARAMETER num_ctx {NUM_CTX}\n"
    )


def create(name: str, gguf: Path) -> None:
    """`ollama create` from a Modelfile in a temp folder (the bundle stays untouched)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "Modelfile"
        path.write_text(modelfile(gguf), encoding="utf-8")
        result = subprocess.run(
            ["ollama", "create", name, "-f", str(path)], capture_output=True, text=True
        )
    if result.returncode != 0:
        raise OllamaError(f"ollama create {name} failed: {result.stderr.strip()[-500:]}")


@dataclass(frozen=True)
class Answer:
    probs: tuple[float, float, float]  # P(S), P(M), P(L); all 0 when no_answer
    no_answer: bool
    latency_ms: float
    prompt_tokens: int


def probs_from_logprobs(top: list[dict]) -> tuple[tuple[float, float, float], bool]:
    """((P(S), P(M), P(L)), no_answer) from Ollama's top_logprobs for one token."""
    found: dict[str, float] = {}
    for item in top:
        tier = _TOKEN_TO_TIER.get(item.get("token", ""))
        if tier is not None and tier not in found:
            found[tier] = float(item["logprob"])
    if not found:
        return (0.0, 0.0, 0.0), True
    peak = max(found.values())
    weights = [math.exp(found[t] - peak) if t in found else 0.0 for t in TIERS]
    total = sum(weights)
    return (weights[0] / total, weights[1] / total, weights[2] / total), False


class OllamaRouter:
    def __init__(
        self, base_url: str, model: str, timeout: float = 60.0, client: httpx.Client | None = None
    ) -> None:
        self.model = model
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)

    def answer(self, text: str) -> Answer:
        body = {
            "model": self.model,
            "prompt": text,
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "keep_alive": "15m",
            "options": {"num_predict": 1, "temperature": 0, "num_gpu": 0, "num_ctx": NUM_CTX},
        }
        start = time.perf_counter()
        try:
            response = self._client.post("/api/generate", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        latency = (time.perf_counter() - start) * 1000
        data = response.json()
        logprobs = data.get("logprobs") or []
        top = logprobs[0].get("top_logprobs", []) if logprobs else []
        probs, missing = probs_from_logprobs(top)
        return Answer(probs, missing, round(latency, 1), int(data.get("prompt_eval_count", 0)))
