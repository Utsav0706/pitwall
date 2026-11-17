from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pitwall import loop
from pitwall.config import Settings
from test_service import setup  # noqa: F401  (pytest fixture)

# --- POST /feedback ---------------------------------------------------------------------------


def test_feedback_is_stored_and_exported(setup) -> None:  # noqa: F811
    client, _, _, _, engine = setup
    client.post("/route", json={"prompt": "Prove the theorem"}, headers={"x-request-id": "r1"})
    client.post("/route", json={"prompt": "Write a haiku"}, headers={"x-request-id": "r2"})
    down = client.post(
        "/feedback", json={"request_id": "r1", "rating": "down", "comment": "too weak"}
    )
    up = client.post("/feedback", json={"request_id": "r2", "rating": "up"})
    assert down.status_code == 201 and up.status_code == 201
    rows = loop.feedback_failures(engine)
    assert len(rows) == 1  # only the thumbs-down
    row = rows[0]
    assert row.input == "Prove the theorem" and row.source == "feedback"
    assert row.reason == "thumbs-down on routing to M (v1): too weak"
    assert row.id == loop.prompt_id("Prove the theorem")
    assert 'pitwall_feedback_total{rating="down"} 1.0' in client.get("/metrics/").text


def test_feedback_needs_a_real_decision_and_a_valid_rating(setup) -> None:  # noqa: F811
    client = setup[0]
    assert (
        client.post("/feedback", json={"request_id": "nope", "rating": "down"}).status_code == 404
    )
    client.post("/route", json={"prompt": "x"}, headers={"x-request-id": "r3"})
    assert client.post("/feedback", json={"request_id": "r3", "rating": "meh"}).status_code == 422


# --- ids, suites, failures ----------------------------------------------------------------------


def test_prompt_id_matches_dynos_case_id() -> None:
    from dyno.failures import ProductionFailure, case_id

    text = "  What is   the Capital of FRANCE? "
    assert loop.prompt_id(text) == case_id(ProductionFailure(input=text))


def test_sample_suite_loads_in_dyno(tmp_path: Path) -> None:
    from dyno.cases import load_suite

    cases = loop.write_sample_suite(["a question", "another one"], tmp_path / "s.yaml")
    suite = load_suite(tmp_path / "s.yaml")
    assert [c.id for c in suite.cases] == list(cases)
    assert suite.agent == "gearbox" and suite.scorers == ["quality"]


def test_estimate_and_missing_key(tmp_path: Path) -> None:
    assert loop.estimate_gbp(10) == pytest.approx(0.035)
    with pytest.raises(loop.LoopError, match="ANTHROPIC_API_KEY"):
        loop.run_sample(tmp_path / "s.yaml", Settings(_env_file=None, anthropic_api_key=None), 0.1)


def fake_run(stdout: str, code: int = 0):
    def run(cmd, **kwargs):
        run.calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=code, stdout=stdout, stderr="")

    run.calls = []
    return run


def test_run_sample_passes_budget_and_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = fake_run(
        "run abc123def456: pitwall-production-sample x1 with gearbox: 10 attempts, "
        "8 passed, 0 errors, 1+2 tokens, GBP 0.004100, 30.0s\n"
    )
    monkeypatch.setattr(loop.subprocess, "run", run)
    monkeypatch.setattr("pitwall.promote.dyno_executable", lambda: "dyno")
    settings = Settings(_env_file=None, anthropic_api_key="sk-test")
    run_id, spent = loop.run_sample(tmp_path / "s.yaml", settings, 0.10)
    assert (run_id, spent) == ("abc123def456", 0.0041)
    env = run.calls[0][1]["env"]
    assert env["JUDGE_BUDGET_GBP"] == "0.1000" and env["ANTHROPIC_API_KEY"] == "sk-test"


def test_dyno_failures_keep_quality_fails_only(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {
        "judge_cost_gbp": 0.02,
        "failures": [
            {
                "case_id": "prod-1",
                "scorer": "quality",
                "failed": 1,
                "attempts": 1,
                "explanation": "wrong year",
            },
            {
                "case_id": "prod-2",
                "scorer": "cost_latency",
                "failed": 1,
                "attempts": 1,
                "explanation": "slow",
            },
        ],
    }
    monkeypatch.setattr(loop.subprocess, "run", fake_run(json.dumps(report)))
    monkeypatch.setattr("pitwall.promote.dyno_executable", lambda: "dyno")
    rows, got = loop.dyno_failures(
        "abc", {"prod-1": "Q1", "prod-2": "Q2"}, Settings(_env_file=None)
    )
    assert [(r.id, r.input, r.source) for r in rows] == [("prod-1", "Q1", "dyno-sample")]
    assert rows[0].reason == "Dyno quality judge: wrong year"
    assert got["judge_cost_gbp"] == 0.02


# --- import ---------------------------------------------------------------------------


def rows() -> list[loop.FailureRow]:
    return [
        loop.FailureRow(loop.prompt_id("Q one"), "Q one", "judge: wrong", "dyno-sample"),
        loop.FailureRow(
            loop.prompt_id("Q two"), "Q two", "thumbs-down", "feedback", tags=["tier-M"]
        ),
    ]


def test_failure_rows_follow_dynos_format(tmp_path: Path) -> None:
    from dyno.failures import read_failures

    path = tmp_path / "f.jsonl"
    loop.write_failures(rows(), path)
    parsed = read_failures(path)
    assert [f.input for f in parsed] == ["Q one", "Q two"]
    assert parsed[1].source == "feedback" and parsed[1].tags == ["tier-M"]


def test_queue_skips_prompts_already_queued(tmp_path: Path) -> None:
    queue = tmp_path / "to_label.jsonl"
    assert loop.queue_for_labelling(rows(), queue) == 2
    assert (
        loop.queue_for_labelling(
            rows()[:1] + [loop.FailureRow("x", "Q three", "r", "feedback")], queue
        )
        == 1
    )
    lines = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert [line["prompt"] for line in lines] == ["Q one", "Q two", "Q three"]


def test_import_cases_with_real_dyno_skips_earlier_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    failures = tmp_path / "f.jsonl"
    loop.write_failures(rows(), failures)
    first, message = loop.import_cases(failures, "1")
    assert first is not None and "2 new cases" in message
    cases = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert {c["input"] for c in cases["cases"]} == {"Q one", "Q two"}
    second, message = loop.import_cases(failures, "2")
    assert second is None and message.startswith("0 new cases")
    assert sorted(p.name for p in loop.CASES_DIR.glob("*.yaml")) == ["cases-1.yaml"]


def test_dyno_cli_is_reachable() -> None:
    from pitwall.promote import dyno_executable

    assert subprocess.run([dyno_executable(), "version"], capture_output=True).returncode == 0
