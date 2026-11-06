"""Score a router's picks on the fixed test set and log the result to MLflow.

Every evaluated router, whatever produced its picks, goes through `score` and `log`, so
baseline, reference and fine-tuned rows are computed the same way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pitwall import metrics, tracking
from pitwall.data import OUT_DIR, SRC_DIR, Pair, load_pairs

TEST_FILE = OUT_DIR / "test.jsonl"
SCORES_FILE = SRC_DIR / "labels.jsonl"


@dataclass
class Prediction:
    """One routing decision, in Gearbox's router_predictions.jsonl format."""

    id: str
    router: str
    tier: str
    reason: str = ""
    latency_ms: float = 0.0
    cost_gbp: float = 0.0


@dataclass
class Evaluation:
    router: str
    metrics: dict[str, Any]
    predictions: list[Prediction] = field(default_factory=list)


def score(
    router: str,
    predictions: list[Prediction],
    test: list[Pair] | None = None,
    scores: dict[str, dict[str, int]] | None = None,
) -> Evaluation:
    test = test if test is not None else load_pairs(TEST_FILE)
    scores = scores if scores is not None else metrics.load_scores(SCORES_FILE)
    by_id = {p.id: p for p in predictions}
    if missing := [p.id for p in test if p.id not in by_id]:
        raise ValueError(
            f"{router}: no prediction for {len(missing)} test prompts, e.g. {missing[0]}"
        )
    ordered = [by_id[p.id] for p in test]
    result = metrics.evaluate(
        truth=[p.label for p in test],
        picks=[pred.tier for pred in ordered],
        scores=[scores[p.id] for p in test],
        parts=[p.part for p in test],
    )
    return Evaluation(router, result, ordered)


def predictions_jsonl(predictions: list[Prediction]) -> str:
    return "".join(json.dumps(p.__dict__, ensure_ascii=False) + "\n" for p in predictions)


def log(
    evaluation: Evaluation,
    kind: str,
    params: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """One MLflow run for one evaluated router; returns the run id."""
    import mlflow

    with tracking.run(
        evaluation.router, kind, tags={"router": evaluation.router, **(tags or {})}
    ) as r:
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics.flat(evaluation.metrics))
        mlflow.log_dict(evaluation.metrics, "metrics.json")
        mlflow.log_text(predictions_jsonl(evaluation.predictions), "predictions.jsonl")
        for name, obj in (artifacts or {}).items():
            mlflow.log_dict(obj, name)
        return r.info.run_id


def read_predictions(path: Path) -> dict[str, list[Prediction]]:
    """A router_predictions.jsonl file, grouped by router name."""
    grouped: dict[str, list[Prediction]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                grouped.setdefault(row["router"], []).append(Prediction(**row))
    return grouped
