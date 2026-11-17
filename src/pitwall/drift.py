"""Drift: do today's prompts still look like the ones the production model learned from?

Reference: the production bundle's own training file. Current: the last N prompts the
production model routed (Pitwall's route log). Three features, each tested by Evidently:

- prompt length (characters): Kolmogorov-Smirnov test, drift when p < 0.05. Cheapest
  signal, and what drives routing latency on CPU.
- task mix: chi-square test on a *predicted* task type (coding, reasoning, research,
  writing). Live prompts carry no label, so each is given the task of the nearest
  centroid of the training prompts' embeddings (bge-small, the LR baseline's model).
- meaning: Evidently's embeddings drift with its "model" method: a classifier learns to
  tell reference from current embeddings; a ROC AUC above 0.7 means they are easy to
  tell apart. Two sets of prompts can match on length and task and still be about
  different things.

Drift is detected when the share of drifted features reaches `drift.threshold`
(config.yaml, 0.5 = 2 of 3). The test methods are pinned: Evidently would otherwise
switch to distance scores (higher = drift) on large samples and flip the meaning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

P_VALUE = 0.05
AUC_THRESHOLD = 0.7
TASK_TYPES = ("coding", "reasoning", "research", "writing")


class TaskCentroids:
    """Nearest-centroid task type from normalised embeddings (cosine similarity)."""

    def __init__(self, embeddings: np.ndarray, tasks: list[str]) -> None:
        labels = np.asarray(tasks)
        self.tasks = sorted(set(tasks))
        rows = []
        for t in self.tasks:
            c = embeddings[labels == t].mean(axis=0)
            rows.append(c / np.linalg.norm(c))
        self.centroids = np.vstack(rows)

    def predict(self, embeddings: np.ndarray) -> list[str]:
        return [self.tasks[i] for i in np.argmax(embeddings @ self.centroids.T, axis=1)]


@dataclass(frozen=True)
class FeatureDrift:
    name: str
    method: str
    value: float  # p-value (length, task mix) or ROC AUC (embeddings)
    threshold: float
    drifted: bool


@dataclass
class DriftResult:
    features: list[FeatureDrift]
    share: float
    detected: bool
    reference_n: int
    current_n: int
    task_mix: dict[str, dict[str, float]] = field(default_factory=dict)
    snapshot: Any = None  # the Evidently report, for the HTML

    def to_record(self) -> dict[str, Any]:
        return {
            "share": self.share,
            "detected": self.detected,
            "reference_n": self.reference_n,
            "current_n": self.current_n,
            "features": [asdict(f) for f in self.features],
            "task_mix": self.task_mix,
        }


def _frame(texts: list[str], embeddings: np.ndarray, tasks: list[str]) -> Any:
    import pandas as pd

    df = pd.DataFrame(embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])])
    df["chars"] = [len(t) for t in texts]
    df["task"] = tasks
    return df


def _mix(tasks: list[str]) -> dict[str, float]:
    return {t: round(tasks.count(t) / len(tasks), 3) for t in TASK_TYPES}


def check(
    reference_texts: list[str],
    reference_embeddings: np.ndarray,
    reference_tasks: list[str],
    current_texts: list[str],
    current_embeddings: np.ndarray,
    share_threshold: float,
    auc_threshold: float = AUC_THRESHOLD,
) -> DriftResult:
    """Run the three Evidently checks; reference tasks are labels, current are predicted."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import EmbeddingsDrift, ValueDrift
    from evidently.metrics.embeddings import ModelDriftMethod

    if len(current_texts) < 30:
        raise ValueError(f"only {len(current_texts)} current prompts; need at least 30")
    centroids = TaskCentroids(reference_embeddings, reference_tasks)
    current_tasks = centroids.predict(current_embeddings)
    # Reference tasks are predicted too (not the labels), so both sides carry the same
    # classifier's errors and only a real change of mix shows up as drift.
    reference_pred = centroids.predict(reference_embeddings)

    columns = [f"e{i}" for i in range(reference_embeddings.shape[1])]
    definition = DataDefinition(
        numerical_columns=["chars", *columns],
        categorical_columns=["task"],
        embeddings={"prompt": columns},
    )
    ref = Dataset.from_pandas(
        _frame(reference_texts, reference_embeddings, reference_pred), data_definition=definition
    )
    cur = Dataset.from_pandas(
        _frame(current_texts, current_embeddings, current_tasks), data_definition=definition
    )
    report = Report(
        [
            ValueDrift(column="chars", method="ks", threshold=P_VALUE),
            ValueDrift(column="task", method="chisquare", threshold=P_VALUE),
            EmbeddingsDrift(
                embeddings_name="prompt", drift_method=ModelDriftMethod(threshold=auc_threshold)
            ),
        ]
    )
    snapshot = report.run(current_data=cur, reference_data=ref)
    values = [float(m["value"]) for m in snapshot.dict()["metrics"]]
    features = [
        FeatureDrift("prompt_length", "K-S p-value", values[0], P_VALUE, values[0] < P_VALUE),
        FeatureDrift("task_mix", "chi-square p-value", values[1], P_VALUE, values[1] < P_VALUE),
        FeatureDrift(
            "embeddings",
            "domain classifier ROC AUC",
            values[2],
            auc_threshold,
            values[2] > auc_threshold,
        ),
    ]
    share = sum(f.drifted for f in features) / len(features)
    return DriftResult(
        features,
        share,
        share >= share_threshold,
        len(reference_texts),
        len(current_texts),
        {"reference": _mix(reference_pred), "current": _mix(current_tasks)},
        snapshot,
    )


def render(result: DriftResult) -> str:
    lines = [f"{'feature':<14} {'test':<26} {'value':>8} {'threshold':>9}  drifted"]
    for f in result.features:
        mark = "YES" if f.drifted else "no"
        lines.append(f"{f.name:<14} {f.method:<26} {f.value:>8.4f} {f.threshold:>9}  {mark}")
    mix = result.task_mix
    lines.append(
        "task mix  reference "
        + " ".join(f"{t} {mix['reference'][t]:.0%}" for t in TASK_TYPES)
        + "\n          current   "
        + " ".join(f"{t} {mix['current'][t]:.0%}" for t in TASK_TYPES)
    )
    verdict = "DRIFT DETECTED" if result.detected else "no drift"
    lines.append(
        f"{verdict}: {result.share:.0%} of features drifted "
        f"({result.current_n} current vs {result.reference_n} reference prompts)"
    )
    return "\n".join(lines)


def save_record(engine: Any, result: DriftResult, model_ref: str, reference: str) -> int:
    from sqlalchemy.orm import Session

    from pitwall.serving.db import DriftCheck

    with Session(engine) as session:
        row = DriftCheck(
            created_at=datetime.now(UTC),
            model_ref=model_ref,
            reference=reference,
            window_n=result.current_n,
            share=result.share,
            detected=result.detected,
            details=json.dumps(result.to_record()),
        )
        session.add(row)
        session.commit()
        return int(row.id)


def distinct(prompts: list[str]) -> list[str]:
    """Each prompt once, first occurrence kept. Drift is about which kinds of prompts
    arrive, not how often one repeats; and repeats would leak across the embedding
    classifier's own train/test split (it remembers them), inflating its ROC AUC."""
    return list(dict.fromkeys(prompts))


def current_window(engine: Any, model_ref: str, n: int) -> list[str]:
    """The last n prompts the given reference routed (newest first in the log)."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from pitwall.serving.db import RouteLog

    with Session(engine) as session:
        rows = session.scalars(
            select(RouteLog.prompt)
            .where(RouteLog.model_ref == model_ref)
            .order_by(RouteLog.created_at.desc())
            .limit(n)
        ).all()
    return list(rows)


def report_path(directory: Path = Path("runs")) -> Path:
    return directory / f"drift-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.html"
