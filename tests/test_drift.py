from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from prometheus_client import CollectorRegistry, generate_latest
from sqlalchemy.orm import Session

from pitwall import drift
from pitwall.serving.db import RouteLog, open_engine
from pitwall.serving.metrics import DriftCollector

TASKS = list(drift.TASK_TYPES)


def population(n: int, rng: np.random.Generator, shift: float = 0.0, mix=None):
    """Prompts whose embedding direction depends on the task (plus a global shift)."""
    mix = mix or [0.25] * 4
    tasks = list(rng.choice(TASKS, size=n, p=mix))
    centres = np.eye(16)[:4] * 3
    emb = np.vstack([centres[TASKS.index(t)] + rng.normal(0, 1, 16) for t in tasks])
    emb[:, 8] += shift * 4
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    lengths = rng.normal(250 + 400 * shift, 60, n).clip(10).astype(int)
    texts = ["x" * int(n_chars) for n_chars in lengths]
    return texts, emb, tasks


def test_task_centroids_recover_the_task() -> None:
    rng = np.random.default_rng(1)
    texts, emb, tasks = population(400, rng)
    model = drift.TaskCentroids(emb, tasks)
    accuracy = np.mean([p == t for p, t in zip(model.predict(emb), tasks, strict=True)])
    assert accuracy > 0.9


def test_same_population_is_not_drift() -> None:
    rng = np.random.default_rng(2)
    ref = population(365, rng)
    cur = population(200, rng)
    result = drift.check(ref[0], ref[1], ref[2], cur[0], cur[1], share_threshold=0.5)
    assert not result.detected
    by_name = {f.name: f for f in result.features}
    assert by_name["embeddings"].value < drift.AUC_THRESHOLD
    assert result.current_n == 200 and result.reference_n == 365


def test_new_kind_of_prompt_is_drift() -> None:
    rng = np.random.default_rng(3)
    ref = population(365, rng)
    cur = population(200, rng, shift=1.0, mix=[0.05, 0.45, 0.45, 0.05])
    result = drift.check(ref[0], ref[1], ref[2], cur[0], cur[1], share_threshold=0.5)
    assert result.detected and result.share == 1.0
    assert [f.drifted for f in result.features] == [True, True, True]
    assert result.task_mix["current"]["research"] > result.task_mix["reference"]["research"]
    text = drift.render(result)
    assert "DRIFT DETECTED" in text and "embeddings" in text


def test_too_few_current_prompts() -> None:
    rng = np.random.default_rng(4)
    ref = population(100, rng)
    cur = population(10, rng)
    with pytest.raises(ValueError, match="at least 30"):
        drift.check(ref[0], ref[1], ref[2], cur[0], cur[1], share_threshold=0.5)


def log_row(session: Session, i: int, ref: str, prompt: str, when: datetime) -> None:
    session.add(
        RouteLog(
            request_id=f"r{i}",
            created_at=when,
            model_ref=ref,
            model_version=1,
            tier="M",
            p_s=0.3,
            p_m=0.7,
            p_l=0.0,
            fallback=False,
            latency_ms=50,
            prompt_chars=len(prompt),
            prompt_cut=False,
            prompt=prompt,
        )
    )


def test_window_is_the_newest_prompts_of_that_reference(tmp_path: Path) -> None:
    engine = open_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    start = datetime(2026, 11, 17, tzinfo=UTC)
    with Session(engine) as s:
        for i in range(5):
            log_row(s, i, "production", f"prod {i}", start + timedelta(minutes=i))
        log_row(s, 9, "candidate", "cand", start + timedelta(minutes=9))
        s.commit()
    assert drift.current_window(engine, "production", 3) == ["prod 4", "prod 3", "prod 2"]


def value(text: str, name: str) -> float:
    """The sample value of an unlabelled metric (skips the # HELP / # TYPE lines)."""
    line = next(line for line in text.splitlines() if line.startswith(name + " "))
    return float(line.split(" ")[1])


def test_latest_check_becomes_gauges(tmp_path: Path) -> None:
    engine = open_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    registry = CollectorRegistry()
    registry.register(DriftCollector(engine))
    assert "pitwall_drift" not in generate_latest(registry).decode()  # no check yet: no data

    features = [
        drift.FeatureDrift("prompt_length", "K-S p-value", 0.001, 0.05, True),
        drift.FeatureDrift("task_mix", "chi-square p-value", 0.2, 0.05, False),
        drift.FeatureDrift("embeddings", "ROC AUC", 0.9, 0.7, True),
    ]
    result = drift.DriftResult(features, 2 / 3, True, 365, 200)
    drift.save_record(engine, result, "production", "v1 train_old.jsonl")
    text = generate_latest(registry).decode()
    assert "pitwall_drift_detected 1.0" in text
    assert 'pitwall_drift_feature{feature="task_mix"} 0.0' in text
    assert 'pitwall_drift_feature{feature="embeddings"} 1.0' in text
    share = value(text, "pitwall_drift_share")
    assert share == pytest.approx(2 / 3)
    stamp = value(text, "pitwall_drift_check_timestamp_seconds")
    assert abs(stamp - datetime.now(UTC).timestamp()) < 120  # stored as UTC, read as UTC
    record = json.loads(json.dumps(result.to_record()))
    assert record["features"][2]["name"] == "embeddings"
