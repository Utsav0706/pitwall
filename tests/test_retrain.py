from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from pitwall import promote as p
from pitwall import retrain
from test_promote import RULES, settings, world  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_gates_but_leaves_the_alias(world) -> None:  # noqa: F811
    reg, logged, _, _ = world
    reg.aliases["production"] = 1
    result = p.promote("v2", settings(), RULES, dry_run=True)
    assert result.decision.promote
    assert reg.aliases["production"] == 1  # not moved
    assert logged[-1][1]["decision"] == "would promote"
    with pytest.raises(p.PromotionError, match="no promotion"):
        p.rollback()  # a dry run is not a promotion to roll back


def fake_runner(outputs: dict[str, tuple[int, str]]):
    calls: list[list[str]] = []

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        code, out = outputs.get(" ".join(args[:2]).replace("\\", "/"), (0, ""))
        return subprocess.CompletedProcess(args, code, out, "")

    return run, calls


def test_finish_runs_the_loop_in_order() -> None:
    run, calls = fake_runner(
        {
            "register bundles/ft-train_full": (
                0,
                "registered: pitwall-router version 2 (ft-train_full)",
            ),
            "canary replay": (0, "40 requests, GBP 0.0080 -> runs\\canary-20261118-100000.jsonl"),
        }
    )
    lines: list[str] = []
    version = retrain.finish(Path("bundles/ft-train_full"), 40, 0.1, lines.append, run)
    assert version == "v2"
    assert [c[0] for c in calls] == [
        "bundle",
        "evaluate",
        "register",
        "candidate",
        "promote",
        "canary",
        "canary",
        "promote",
    ]
    assert calls[4] == ["promote", "v2", "--dry-run"]
    assert calls[6] == ["canary", "report", "runs\\canary-20261118-100000.jsonl"]
    assert calls[7] == ["promote", "v2"]


def test_a_rejected_gate_stops_before_canary_and_promotion() -> None:
    run, calls = fake_runner(
        {
            "register bundles/x": (0, "registered: pitwall-router version 3 (x)"),
            "promote v3": (1, "REJECTED: v3 vs v2"),
        }
    )
    with pytest.raises(retrain.RetrainError, match="gate \\(dry run\\) failed"):
        retrain.finish(Path("bundles/x"), 40, 0.1, lambda s: None, run)
    assert calls[-1] == ["promote", "v3", "--dry-run"]  # nothing after the gate


def test_no_canary_when_n_is_zero() -> None:
    steps = retrain.finish_steps(Path("b"), "v2", 0, 0.1)
    assert [s.name for s in steps] == ["candidate", "gate (dry run)", "promote"]


def test_version_parsing() -> None:
    assert retrain.registered_version("already registered: pitwall-router version 4 (b)") == "v4"
    with pytest.raises(retrain.RetrainError):
        retrain.registered_version("error: bundle check failed")


def test_notebook_trains_the_file_named_in_its_parameters_cell() -> None:
    nb = json.loads((ROOT / "notebooks/train_router_colab.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert code[0].startswith("# What to train on") and 'TRAIN = "train_full"' in code[0]
    train_cell = next(c for c in code if "pitwall.finetune.train" in c)
    assert "{TRAIN}.jsonl" in train_cell and "bundles/{NAME}" in train_cell
    assert not any("train_old.jsonl" in c for c in code)
