"""Prometheus metrics for the routing service (GET /metrics).

- pitwall_route_requests_total{model_ref, version, tier, fallback}: traffic, tier mix
  and fail-safes, per model version (a canary or a promotion shows up as a new version).
- pitwall_route_latency_seconds{version}: the routing time the promotion rule limits
  (p95 <= 600 ms); buckets around that limit.
- pitwall_route_errors_total{reason}: requests answered 503 (no model / Ollama down).
- pitwall_model_info{model_ref, version}: 1 for the version each reference serves now.
- pitwall_prompt_chars: prompt length; it drives CPU latency and is the cheapest early
  sign of drift.

No cost metric: the routing decision runs on the local CPU (~GBP 0). The money is spent
by Gearbox's answers, which belong on Gearbox's own metrics.

Each app gets its own registry, so tests can build many apps in one process.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.025, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.5, 5.0)
PROMPT_BUCKETS = (50, 100, 200, 350, 500, 1000, 2000, 3000, 5000)


class ServiceMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "pitwall_route_requests",
            "Routing decisions",
            ["model_ref", "version", "tier", "fallback"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "pitwall_route_latency_seconds",
            "Time to decide a tier (the model call)",
            ["version"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "pitwall_route_errors",
            "Requests answered with an error",
            ["reason"],
            registry=self.registry,
        )
        self.model_info = Gauge(
            "pitwall_model_info",
            "1 for the version each model reference serves",
            ["model_ref", "version"],
            registry=self.registry,
        )
        self.prompt_chars = Histogram(
            "pitwall_prompt_chars",
            "Prompt length in characters",
            buckets=PROMPT_BUCKETS,
            registry=self.registry,
        )
        self._serving: dict[str, str] = {}

    def record(
        self, ref: str, version: int, tier: str, fallback: bool, seconds: float, chars: int
    ) -> None:
        v = f"v{version}"
        self.requests.labels(ref, v, tier, str(fallback).lower()).inc()
        self.latency.labels(v).observe(seconds)
        self.prompt_chars.observe(chars)
        previous = self._serving.get(ref)
        if previous != v:
            if previous is not None:  # the alias moved: promotion or rollback
                self.model_info.remove(ref, previous)
            self.model_info.labels(ref, v).set(1)
            self._serving[ref] = v
