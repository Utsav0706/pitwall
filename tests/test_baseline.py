from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pitwall import baseline, evaluation, metrics, tracking
from pitwall.data import Pair


def pair(i: int, label: str, part: str = "old", text: str | None = None) -> Pair:
    return Pair(
        id=f"p{i:03d}",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": text or f"prompt {i} {label}"},
            {"role": "assistant", "content": label},
        ],
        label=label,  # type: ignore[arg-type]
        task_type="writing",
        source="dolly",
        part=part,  # type: ignore[arg-type]
    )


class FakeEmbedder:
    """Separable 'embeddings': the label letter in the text decides the direction."""

    name = "fake"

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        rows = []
        for t in texts:
            v = rng.normal(0, 0.05, 8)
            v["SML".index(t.split()[-1])] += 1.0
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows)


# --- metrics -------------------------------------------------------------------------


def test_under_and_over_routing_are_separate() -> None:
    m = metrics.evaluate(truth=["S", "M", "L", "M"], picks=["S", "S", "M", "L"])
    assert m["accuracy"] == 0.25
    assert m["under_rate"] == 0.5  # M->S, L->M
    assert m["over_rate"] == 0.25  # M->L
    assert m["routing_cost"] == (2 * 2 + 1) / 4
    assert m["picks"] == {"S": 2, "M": 1, "L": 1}
    assert m["confusion"] == [[1, 0, 0], [1, 0, 1], [0, 1, 0]]


def test_pass_rate_uses_the_picked_tiers_score() -> None:
    scores = [{"S": 2, "M": 5, "L": 5}, {"S": 4, "M": 4, "L": 3}]
    m = metrics.evaluate(truth=["M", "S"], picks=["S", "L"], scores=scores)
    assert m["pass_rate"] == 0.0  # S scored 2, L scored 3


def test_by_part_and_flat_keys() -> None:
    m = metrics.evaluate(
        truth=["S", "M"], picks=["S", "S"], scores=[{"S": 5}, {"S": 1}], parts=["old", "hard"]
    )
    assert m["by_part"]["old"]["accuracy"] == 1.0
    assert m["by_part"]["hard"]["under_rate"] == 1.0
    keys = metrics.flat(m)
    assert keys["hard.accuracy"] == 0.0
    assert keys["picks.S"] == 2.0
    assert "confusion" not in keys


# --- baseline ------------------------------------------------------------------------


def test_decide_is_cumulative() -> None:
    p = np.array([0.3, 0.35, 0.35])
    assert baseline.decide(p, 0.3) == 0
    assert baseline.decide(p, 0.6) == 1  # 0.3 + 0.35 >= 0.6
    assert baseline.decide(p, 0.7) == 2


def test_proba_matches_sklearn() -> None:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 4))
    y = np.repeat([0, 1, 2], 20)
    model = LogisticRegression(max_iter=1000).fit(x, y)
    w = baseline.Weights(model.coef_, model.intercept_, 0.5)
    np.testing.assert_allclose(w.proba(x), model.predict_proba(x), atol=1e-9)


def test_train_searches_grid_and_predicts(tmp_path: Path) -> None:
    pairs = [pair(i, "SML"[i % 3]) for i in range(30)]
    trained = baseline.train(pairs, FakeEmbedder(), cache_dir=tmp_path)
    assert len(trained.grid) == 5 * 2 * 4
    assert trained.grid[0]["cost"] == min(r["cost"] for r in trained.grid)
    assert trained.params["examples"] == 30
    preds = baseline.predict("lr-test", trained.weights, pairs, FakeEmbedder(), tmp_path)
    assert [p.tier for p in preds] == [p.label for p in pairs]  # separable data
    assert preds[0].reason.startswith("lr: P(S)=")


def test_fixed_settings_skip_the_search(tmp_path: Path) -> None:
    pairs = [pair(i, "SML"[i % 3]) for i in range(12)]
    settings = {"C": 0.1, "class_weight": "balanced", "threshold": 0.6}
    trained = baseline.train(pairs, FakeEmbedder(), settings, cache_dir=tmp_path)
    assert trained.grid == []
    assert trained.weights.threshold == 0.6
    assert trained.params["folds"] == 0


def test_embedding_cache_is_reused(tmp_path: Path) -> None:
    class Counting(FakeEmbedder):
        calls = 0

        def encode(self, texts: list[str]) -> np.ndarray:
            Counting.calls += 1
            return super().encode(texts)

    texts = ["a S", "b M"]
    first = baseline.embed_cached(texts, Counting(), tmp_path)
    second = baseline.embed_cached(texts, Counting(), tmp_path)
    assert Counting.calls == 1
    np.testing.assert_array_equal(first, second)


def test_weights_json_round_trip_shape() -> None:
    w = baseline.Weights(np.zeros((3, 4)), np.zeros(3), 0.6).to_json()
    assert w["classes"] == ["S", "M", "L"]
    assert len(w["coef"]) == 3 and len(w["coef"][0]) == 4


# --- evaluation + tracking -------------------------------------------------------------


def test_score_needs_every_test_prompt() -> None:
    test = [pair(1, "S"), pair(2, "M")]
    preds = [evaluation.Prediction("p001", "r", "S")]
    with pytest.raises(ValueError, match="no prediction for 1"):
        evaluation.score("r", preds, test, {"p001": {"S": 5}, "p002": {"S": 1}})


def test_predictions_use_gearbox_format(tmp_path: Path) -> None:
    preds = [evaluation.Prediction("p001", "lr", "M", "why", 1.5, 0.0)]
    text = evaluation.predictions_jsonl(preds)
    row = json.loads(text)
    assert set(row) == {"id", "router", "tier", "reason", "latency_ms", "cost_gbp"}
    path = tmp_path / "preds.jsonl"
    path.write_text(text, encoding="utf-8")
    assert evaluation.read_predictions(path)["lr"][0].tier == "M"


def test_data_version_reads_dvc_lock(tmp_path: Path) -> None:
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\nstages:\n  pairs:\n    outs:\n    - path: data/derived\n"
        "      md5: abc123.dir\n",
        encoding="utf-8",
    )
    assert tracking.data_version(lock) == "abc123.dir"
    assert tracking.data_version(tmp_path / "missing.lock") == "unknown"


def test_log_writes_tags_metrics_and_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    from pitwall.config import get_settings

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    get_settings.cache_clear()
    monkeypatch.chdir(root)  # dvc.lock and data/derived/manifest.json, if built
    monkeypatch.setattr(mlflow, "set_experiment", _experiment_in(tmp_path, mlflow.set_experiment))
    try:
        test = [pair(1, "S", "old"), pair(2, "L", "hard")]
        result = evaluation.score(
            "always-M",
            [evaluation.Prediction(p.id, "always-M", "M") for p in test],
            test,
            {"p001": {"S": 5, "M": 5}, "p002": {"M": 2, "L": 5}},
        )
        run_id = evaluation.log(result, "reference", params={"x": 1})
        run = mlflow.get_run(run_id)
        assert run.data.tags["kind"] == "reference"
        assert run.data.tags["router"] == "always-M"
        assert {"data_version", "git_commit", "git_dirty"} <= set(run.data.tags)
        assert run.data.metrics["pass_rate"] == 0.5
        assert run.data.metrics["hard.under_rate"] == 1.0
        listed = tracking.list_runs()
        assert listed[0]["id"] == run_id
    finally:
        get_settings.cache_clear()


def _experiment_in(tmp_path: Path, original):
    """Create the experiment with its artifacts in tmp_path, not ./mlruns in the repo."""

    def set_experiment(name: str):
        import mlflow

        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(tmp_path / "art").as_uri())
        return original(name)

    return set_experiment
