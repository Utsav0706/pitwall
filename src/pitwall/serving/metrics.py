"""Prometheus metrics for the routing service (GET /metrics).

- pitwall_route_requests_total{model_ref, version, tier, fallback}: traffic, tier mix
  and fail-safes, per model version (a canary or a promotion shows up as a new version).
- pitwall_route_latency_seconds{version}: the routing time the promotion rule limits
  (p95 <= 600 ms); buckets around that limit.
- pitwall_route_errors_total{reason}: requests answered 503 (no model / Ollama down).
- pitwall_model_info{model_ref, version}: 1 for the version each reference serves now.
- pitwall_prompt_chars: prompt length; it drives CPU latency and is the cheapest early
  sign of drift.
- pitwall_drift_share / _detected / _feature{feature} / _check_timestamp_seconds: the
  latest `pitwall drift check` (M9), read from the database at scrape time.

No cost metric: the routing decision runs on the local CPU (~GBP 0). The money is spent
by Gearbox's answers, which belong on Gearbox's own metrics.

Each app gets its own registry, so tests can build many apps in one process.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily

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


class DriftCollector:
    """Reads the latest `pitwall drift check` from the database at scrape time, so the
    drift verdict lives in Prometheus (alert PitwallDrift) and Grafana. Nothing is
    exported before the first check: no data, rather than a made-up 0."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def collect(self) -> Iterator[GaugeMetricFamily]:
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from pitwall.serving.db import DriftCheck

        with Session(self._engine) as session:
            row = session.scalars(
                select(DriftCheck).order_by(DriftCheck.created_at.desc()).limit(1)
            ).first()
        if row is None:
            return
        yield GaugeMetricFamily(
            "pitwall_drift_share",
            "Share of drifted features in the latest drift check",
            value=row.share,
        )
        yield GaugeMetricFamily(
            "pitwall_drift_detected",
            "1 if the latest drift check detected drift",
            value=float(row.detected),
        )
        yield GaugeMetricFamily(
            "pitwall_drift_check_timestamp_seconds",
            "When the latest drift check ran",
            # SQLite gives back naive datetimes; they were stored in UTC.
            value=row.created_at.replace(tzinfo=row.created_at.tzinfo or UTC).timestamp(),
        )
        per_feature = GaugeMetricFamily(
            "pitwall_drift_feature",
            "1 if this feature drifted in the latest check",
            labels=["feature"],
        )
        for f in json.loads(row.details)["features"]:
            per_feature.add_metric([f["name"]], float(f["drifted"]))
        yield per_feature
