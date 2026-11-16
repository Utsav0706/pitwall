"""Dyno agents for the promotion gate (`dyno run ... --agent pitwall.dyno_agent:Router`).

- Router: asks the running Pitwall service which tier a prompt needs, for the model
  named in PITWALL_DYNO_MODEL (an alias such as production/candidate, or v1). The tier
  is the "answer" Dyno checks against the expected tier.
- AlwaysM: the trivial rule, the baseline a first model must at least match.

Dyno builds agents with no arguments, so the model comes from the environment;
`pitwall promote` sets it for each run.
"""

from __future__ import annotations

import os

import httpx
from dyno.agent import AgentResult


class Router:
    def __init__(self) -> None:
        self.model = os.environ.get("PITWALL_DYNO_MODEL", "production")
        url = os.environ.get("PITWALL_URL", "http://127.0.0.1:8003")
        self._client = httpx.Client(base_url=url, timeout=60.0)

    def run(self, input: str) -> AgentResult:
        try:
            response = self._client.post("/route", json={"prompt": input, "model": self.model})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            detail = getattr(getattr(exc, "response", None), "text", "")[:200]
            return AgentResult(error=f"/route failed: {exc} {detail}".strip())
        body = response.json()
        return AgentResult(
            final_answer=body["tier"],
            metadata={
                "model_version": body["model_version"],
                "probs": body["probs"],
                "latency_ms": body["latency_ms"],
                "fallback": body["fallback"],
            },
        )


class AlwaysM:
    def run(self, input: str) -> AgentResult:
        return AgentResult(final_answer="M", metadata={"model_version": 0})
