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

import hashlib
import math
import re
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


def create(client: httpx.Client, name: str, gguf: Path, sha256: str | None = None) -> None:
    """Create an Ollama model from a GGUF over Ollama's HTTP API.

    Upload the file as a blob (skipped when Ollama already has it), then create the
    model from that blob. Over HTTP, not the `ollama` CLI: the CLI only works when Ollama
    can read the file's path, which isn't true once Pitwall runs in a container.
    """
    digest = "sha256:" + (sha256 or _sha256(gguf))
    try:
        if client.head(f"/api/blobs/{digest}").status_code != 200:
            with gguf.open("rb") as f:
                client.post(f"/api/blobs/{digest}", content=f, timeout=600).raise_for_status()
        response = client.post(
            "/api/create",
            json={
                "model": name,
                "files": {"model.gguf": digest},
                "parameters": {"temperature": 0, "num_ctx": NUM_CTX},
                "stream": False,
            },
            timeout=600,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"creating {name} in Ollama failed: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def choose_tier(answer: Answer, threshold: float) -> tuple[str, str]:
    """(tier, reason) from the model's answer: the cheapest tier whose cumulative
    probability reaches the threshold, or M when the model gave none of the three.
    Used by evaluation and by the service, so both decide the same way."""
    if answer.no_answer:
        return FALLBACK, "ft: no S/M/L in the model's top tokens, fallback M"
    cumulative = 0.0
    tier = TIERS[-1]
    for t, p in zip(TIERS, answer.probs, strict=True):
        cumulative += p
        if cumulative >= threshold - 1e-12:
            tier = t
            break
    probs = " ".join(f"P({t})={v:.2f}" for t, v in zip(TIERS, answer.probs, strict=True))
    return tier, f"ft: {probs}, threshold {threshold}"
