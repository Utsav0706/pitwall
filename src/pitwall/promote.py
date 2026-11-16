"""Promotion and rollback of the router model: the rule is code, not a feeling.

    pitwall candidate v2          # label a registered version as the one to try
    pitwall promote               # gate the candidate against production; promote if it passes
    pitwall rollback              # put back the version the last promotion replaced

A candidate reaches production only if ALL of these hold:

1. Dyno gate: on the same 119 test cases, run the same day through the running service,
   its accuracy (Dyno's task_success, exact tier match) is at most `max_drop_pp` points
   below production's.
2. Under-routing (a smaller tier than needed = a bad answer) is no higher than
   production's.
3. p95 routing latency is within `p95_ms_limit` (config.yaml).

Under-routing and p95 come from each version's evaluation (`pitwall evaluate`, where
latency is measured on every test prompt). With no production model yet, the baseline
is the trivial rule "always M": a first model must at least match it.

`decide` is the rule alone, with no MLflow, Dyno or network in it, so it can be read,
tested (tests/test_promote.py is its contract) and rewritten on its own. Everything
else in this file is plumbing: running Dyno, reading metrics, moving aliases, logging.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pitwall.config import PromotionRules, Settings
from pitwall.registry import REGISTERED_MODEL

PRODUCTION = "production"
CANDIDATE = "candidate"
ALWAYS_M = "always-M"
SUITE = Path("data/dyno/router_suite.yaml")
GATE_METRIC = "task_success"


class PromotionError(RuntimeError):
    pass


# --- the rule ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStats:
    """What the rule needs to know about one side (candidate or baseline)."""

    label: str  # "v2", or "always-M"
    under_rate: float
    latency_p95_ms: float


@dataclass(frozen=True)
class GateOutcome:
    """Dyno's verdict on accuracy (from dyno.gate.gate)."""

    passed: bool
    baseline: float | None  # accuracy, 0-1
    candidate: float | None
    reason: str


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Decision:
    promote: bool
    checks: list[Check] = field(default_factory=list)

    @property
    def reason(self) -> str:
        failed = [c for c in self.checks if not c.passed]
        if not failed:
            return "all checks passed"
        return "; ".join(f"{c.name}: {c.detail}" for c in failed)


def decide(
    candidate: ModelStats, baseline: ModelStats, gate: GateOutcome, rules: PromotionRules
) -> Decision:
    """Promote only if accuracy holds (Dyno), under-routing doesn't rise, and p95 fits."""
    checks = [
        Check("dyno_gate", gate.passed, gate.reason),
        Check(
            "under_routing",
            candidate.under_rate <= baseline.under_rate + 1e-9,
            f"{candidate.label} {candidate.under_rate:.1%} vs "
            f"{baseline.label} {baseline.under_rate:.1%}",
        ),
        Check(
            "latency_p95",
            candidate.latency_p95_ms <= rules.p95_ms_limit,
            f"{candidate.latency_p95_ms:.0f} ms (limit {rules.p95_ms_limit:.0f} ms)",
        ),
    ]
    return Decision(all(c.passed for c in checks), checks)


# --- plumbing --------------------------------------------------------------------------------


def _client() -> Any:
    from pitwall.registry import client

    return client()


def alias_version(c: Any, alias: str) -> int | None:
    try:
        return int(c.get_model_version_by_alias(REGISTERED_MODEL, alias).version)
    except Exception:  # MLflow raises when the alias doesn't exist
        return None


def parse_ref(c: Any, ref: str) -> int:
    """'candidate' / 'production' / 'v2' -> version number."""
    match = re.fullmatch(r"v(\d+)", ref)
    if match:
        c.get_model_version(REGISTERED_MODEL, match.group(1))  # raises if it doesn't exist
        return int(match.group(1))
    version = alias_version(c, ref)
    if version is None:
        raise PromotionError(f"no {REGISTERED_MODEL} version has the alias {ref!r}")
    return version


def version_stats(c: Any, version: int) -> ModelStats:
    tags = c.get_model_version(REGISTERED_MODEL, str(version)).tags
    try:
        return ModelStats(f"v{version}", float(tags["under_rate"]), float(tags["latency_p95_ms"]))
    except KeyError as exc:
        raise PromotionError(
            f"v{version} has no evaluation metrics; run `pitwall evaluate`"
        ) from exc


def always_m_stats() -> ModelStats:
    """The trivial rule's numbers, from its reference run (`pitwall reference`)."""
    from pitwall.tracking import list_runs

    for r in list_runs():
        if r["name"] == ALWAYS_M and r["tags"].get("kind") == "reference":
            return ModelStats(ALWAYS_M, r["metrics"]["under_rate"], 0.0)
    raise PromotionError("no always-M reference run; run `pitwall reference` first")


def dyno_executable() -> str:
    name = "dyno.exe" if sys.platform == "win32" else "dyno"
    path = Path(sys.executable).parent / name
    if not path.exists():
        raise PromotionError(f'Dyno not found at {path}; pip install -e ".[gate]"')
    return str(path)


def run_dyno(agent: str, model_ref: str | None, settings: Settings) -> str:
    """One Dyno run of the gate suite; returns Dyno's run id."""
    env = {
        **os.environ,
        "DATABASE_URL": settings.dyno_database_url,
        "PITWALL_URL": settings.pitwall_url,
        "PITWALL_DYNO_MODEL": model_ref or "",
    }
    result = subprocess.run(
        [dyno_executable(), "run", str(SUITE), "--agent", agent],
        capture_output=True,
        text=True,
        env=env,
    )
    match = re.search(r"^run (\w+):", result.stdout, re.MULTILINE)
    if result.returncode != 0 or not match:
        raise PromotionError(f"dyno run failed: {(result.stderr or result.stdout)[-600:]}")
    errors = re.search(r"(\d+) errors", result.stdout)
    if errors and int(errors.group(1)) > 0:
        raise PromotionError(
            f"dyno run {match.group(1)} had {errors.group(1)} agent errors: is the service "
            f"running at {settings.pitwall_url} and serving {model_ref!r}?"
        )
    return match.group(1)


def dyno_gate(
    candidate_run: str, baseline_run: str, rules: PromotionRules, url: str
) -> GateOutcome:
    from dyno.gate import gate
    from dyno.store import Store, open_engine

    r = gate(
        candidate_run,
        baseline_run,
        max_drop=rules.max_drop_pp,
        metric=GATE_METRIC,
        store=Store(open_engine(url)),
    )
    return GateOutcome(r.passed, r.baseline, r.candidate, r.reason)


@dataclass
class Result:
    decision: Decision
    candidate_version: int
    baseline: str  # "v1" or "always-M"
    dyno_runs: dict[str, str]
    mlflow_run: str


def log_decision(kind: str, record: dict[str, Any], metrics: dict[str, float]) -> str:
    import mlflow

    from pitwall import tracking

    name = f"{kind} {record.get('to', '')}".strip()
    with tracking.run(name, kind, tags={k: str(v) for k, v in record.items()}) as active:
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_dict(record, f"{kind}.json")
        return active.info.run_id


def promote(ref: str, settings: Settings, rules: PromotionRules) -> Result:
    c = _client()
    candidate_version = parse_ref(c, ref)
    production_version = alias_version(c, PRODUCTION)
    if production_version == candidate_version:
        raise PromotionError(f"v{candidate_version} is already production")
    candidate = version_stats(c, candidate_version)
    baseline = (
        always_m_stats() if production_version is None else version_stats(c, production_version)
    )

    cand_run = run_dyno("pitwall.dyno_agent:Router", f"v{candidate_version}", settings)
    if production_version is None:
        base_run = run_dyno("pitwall.dyno_agent:AlwaysM", None, settings)
    else:
        base_run = run_dyno("pitwall.dyno_agent:Router", f"v{production_version}", settings)
    gate_outcome = dyno_gate(cand_run, base_run, rules, settings.dyno_database_url)

    decision = decide(candidate, baseline, gate_outcome, rules)
    record = {
        "decision": "promoted" if decision.promote else "rejected",
        "from": baseline.label,
        "to": candidate.label,
        "reason": decision.reason,
        "dyno_candidate_run": cand_run,
        "dyno_baseline_run": base_run,
        "checks": json.dumps([asdict(ch) for ch in decision.checks]),
    }
    metrics = {
        "candidate_accuracy": gate_outcome.candidate or 0.0,
        "baseline_accuracy": gate_outcome.baseline or 0.0,
        "candidate_under_rate": candidate.under_rate,
        "baseline_under_rate": baseline.under_rate,
        "candidate_p95_ms": candidate.latency_p95_ms,
    }
    if decision.promote:
        c.set_registered_model_alias(REGISTERED_MODEL, PRODUCTION, str(candidate_version))
    mlflow_run = log_decision("promotion", record, metrics)
    return Result(
        decision,
        candidate_version,
        baseline.label,
        {"candidate": cand_run, "baseline": base_run},
        mlflow_run,
    )


def rollback() -> tuple[str, str, str]:
    """(from, to, mlflow run): production goes back to the version the last promotion
    replaced. Refuses when that promotion isn't what's live, or it replaced always-M."""
    from pitwall.tracking import list_runs

    c = _client()
    current = alias_version(c, PRODUCTION)
    last = next(
        (
            r
            for r in list_runs()
            if r["tags"].get("kind") == "promotion" and r["tags"].get("decision") == "promoted"
        ),
        None,
    )
    if last is None or current is None:
        raise PromotionError("nothing to roll back: no promotion has happened")
    to_label, from_label = last["tags"]["to"], last["tags"]["from"]
    if to_label != f"v{current}":
        raise PromotionError(
            f"production is v{current}, but the last promotion was to {to_label}; "
            "nothing to roll back (already rolled back?)"
        )
    if from_label == ALWAYS_M:
        raise PromotionError(
            f"{to_label} was the first production model (it replaced the always-M rule); "
            "there is no earlier model to roll back to"
        )
    c.set_registered_model_alias(REGISTERED_MODEL, PRODUCTION, from_label[1:])
    run = log_decision(
        "rollback",
        {"decision": "rolled back", "from": to_label, "to": from_label},
        {},
    )
    return to_label, from_label, run


def set_candidate(ref: str) -> int:
    c = _client()
    version = parse_ref(c, ref)
    c.set_registered_model_alias(REGISTERED_MODEL, CANDIDATE, str(version))
    return version
