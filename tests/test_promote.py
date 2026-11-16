"""The contract for pitwall.promote. `decide` may be rewritten freely; these tests must
keep passing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from pitwall import dyno_agent, dyno_suite
from pitwall import promote as p
from pitwall.config import PromotionRules

RULES = PromotionRules(p95_ms_limit=600, max_drop_pp=0)
GATE_OK = p.GateOutcome(True, 0.479, 0.479, "task_success 47.9% -> 47.9% (drop 0.0 pp)")


def stats(label: str, under: float = 0.084, p95: float = 488.0) -> p.ModelStats:
    return p.ModelStats(label, under, p95)


# --- decide: the rule ----------------------------------------------------------------------


def test_promotes_when_every_check_passes() -> None:
    d = p.decide(stats("v2"), stats("v1"), GATE_OK, RULES)
    assert d.promote
    assert [c.name for c in d.checks] == ["dyno_gate", "under_routing", "latency_p95"]
    assert d.reason == "all checks passed"


def test_rejects_when_dyno_gate_fails() -> None:
    gate = p.GateOutcome(False, 0.50, 0.45, "task_success dropped 5.0 pp")
    d = p.decide(stats("v2"), stats("v1"), gate, RULES)
    assert not d.promote
    assert "dyno_gate" in d.reason and "dropped" in d.reason


def test_rejects_more_under_routing_even_with_equal_accuracy() -> None:
    d = p.decide(stats("v2", under=0.10), stats("v1", under=0.084), GATE_OK, RULES)
    assert not d.promote
    assert d.reason.startswith("under_routing")


def test_equal_under_routing_is_allowed() -> None:
    assert p.decide(stats("v2", under=0.084), stats("v1", under=0.084), GATE_OK, RULES).promote


def test_rejects_too_slow() -> None:
    d = p.decide(stats("v2", p95=601), stats("v1"), GATE_OK, RULES)
    assert not d.promote
    assert d.reason == "latency_p95: 601 ms (limit 600 ms)"


def test_every_failed_check_is_reported() -> None:
    gate = p.GateOutcome(False, 0.5, 0.4, "dropped")
    d = p.decide(stats("v2", under=0.2, p95=900), stats("v1"), gate, RULES)
    assert [c.name for c in d.checks if not c.passed] == [
        "dyno_gate",
        "under_routing",
        "latency_p95",
    ]


def test_first_model_against_always_m() -> None:
    d = p.decide(stats("v1"), p.ModelStats("always-M", 0.084, 0.0), GATE_OK, RULES)
    assert d.promote


# --- plumbing, with a fake registry and a fake Dyno ------------------------------------------


class FakeRegistry:
    def __init__(self, versions: dict[int, dict[str, str]], aliases: dict[str, int]) -> None:
        self.versions = versions
        self.aliases = aliases

    def get_model_version_by_alias(self, name: str, alias: str):
        if alias not in self.aliases:
            raise RuntimeError(f"alias {alias} not found")
        return SimpleNamespace(version=str(self.aliases[alias]))

    def get_model_version(self, name: str, version: str):
        if int(version) not in self.versions:
            raise RuntimeError("no such version")
        return SimpleNamespace(version=version, tags=self.versions[int(version)])

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = int(version)


TAGS = {"under_rate": "0.0840", "latency_p95_ms": "487.6000"}


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch):
    reg = FakeRegistry({1: dict(TAGS), 2: dict(TAGS)}, {})
    logged: list[tuple[str, dict]] = []
    dyno_calls: list[tuple[str, str | None]] = []
    state = {"gate": GATE_OK, "runs": []}
    monkeypatch.setattr(p, "_client", lambda: reg)
    monkeypatch.setattr(p, "always_m_stats", lambda: p.ModelStats("always-M", 0.084, 0.0))

    def fake_run_dyno(agent: str, ref: str | None, settings) -> str:
        dyno_calls.append((agent.split(":")[1], ref))
        return f"run{len(dyno_calls)}"

    def fake_log(kind: str, record: dict, metrics: dict) -> str:
        logged.append((kind, record))
        state["runs"].insert(0, {"tags": {"kind": kind, **record}})
        return f"mlflow{len(logged)}"

    monkeypatch.setattr(p, "run_dyno", fake_run_dyno)
    monkeypatch.setattr(p, "dyno_gate", lambda c, b, rules, url: state["gate"])
    monkeypatch.setattr(p, "log_decision", fake_log)
    monkeypatch.setattr("pitwall.tracking.list_runs", lambda: state["runs"])
    return reg, logged, dyno_calls, state


def settings():
    from pitwall.config import Settings

    return Settings(_env_file=None)


def test_first_promotion_gates_against_always_m(world) -> None:
    reg, logged, dyno_calls, _ = world
    p.set_candidate("v1")
    result = p.promote("candidate", settings(), RULES)
    assert result.decision.promote and result.baseline == "always-M"
    assert dyno_calls == [("Router", "v1"), ("AlwaysM", None)]
    assert reg.aliases["production"] == 1
    assert logged[-1][0] == "promotion" and logged[-1][1]["decision"] == "promoted"


def test_rejected_candidate_leaves_production_alone(world) -> None:
    reg, logged, _, state = world
    reg.aliases["production"] = 1
    state["gate"] = p.GateOutcome(False, 0.48, 0.40, "dropped 8 pp")
    result = p.promote("v2", settings(), RULES)
    assert not result.decision.promote
    assert reg.aliases["production"] == 1
    assert logged[-1][1]["decision"] == "rejected"


def test_cannot_promote_what_is_already_production(world) -> None:
    reg, _, _, _ = world
    reg.aliases["production"] = 1
    with pytest.raises(p.PromotionError, match="already production"):
        p.promote("v1", settings(), RULES)


def test_missing_alias_and_unevaluated_version(world) -> None:
    reg, _, _, _ = world
    with pytest.raises(p.PromotionError, match="alias 'candidate'"):
        p.promote("candidate", settings(), RULES)
    reg.versions[3] = {}
    with pytest.raises(p.PromotionError, match="pitwall evaluate"):
        p.promote("v3", settings(), RULES)


def test_rollback_restores_the_replaced_version_once(world) -> None:
    reg, logged, _, _ = world
    reg.aliases["production"] = 1
    p.promote("v2", settings(), RULES)
    assert reg.aliases["production"] == 2
    old, new, _ = p.rollback()
    assert (old, new) == ("v2", "v1") and reg.aliases["production"] == 1
    assert logged[-1][0] == "rollback"
    with pytest.raises(p.PromotionError, match="already rolled back"):
        p.rollback()


def test_rollback_refuses_without_history_or_past_the_first_model(world) -> None:
    reg, _, _, _ = world
    with pytest.raises(p.PromotionError, match="no promotion"):
        p.rollback()
    p.promote("v1", settings(), RULES)  # first model, replaced always-M
    with pytest.raises(p.PromotionError, match="first production model"):
        p.rollback()


# --- Dyno agent and suite ---------------------------------------------------------------------


def test_router_agent_returns_the_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json={"tier": "S", "model_version": 2, "probs": {}, "latency_ms": 5, "fallback": False},
        )

    monkeypatch.setenv("PITWALL_DYNO_MODEL", "v2")
    agent = dyno_agent.Router()
    agent._client = httpx.Client(base_url="http://p", transport=httpx.MockTransport(handler))
    result = agent.run("Write a haiku")
    assert result.final_answer == "S" and result.metadata["model_version"] == 2
    assert seen == {"prompt": "Write a haiku", "model": "v2"}


def test_router_agent_reports_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = dyno_agent.Router()
    agent._client = httpx.Client(
        base_url="http://p",
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="no model")),
    )
    result = agent.run("x")
    assert result.error and "no model" in result.error


def test_suite_has_one_exact_case_per_test_prompt(tmp_path: Path) -> None:
    from pitwall.data import write_jsonl
    from test_baseline import pair

    test = tmp_path / "test.jsonl"
    write_jsonl(test, [pair(1, "S", "old"), pair(2, "L", "hard")])
    out = tmp_path / "suite.yaml"
    assert dyno_suite.write_suite(out, test) == 2
    suite = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert suite["scorers"] == [{"task_success": {"mode": "exact"}}]
    assert suite["cases"][1] == {
        "id": "p002",
        "input": "prompt 2 L",
        "expected": "L",
        "tags": ["hard", "writing"],
    }

    from dyno.cases import load_suite

    assert len(load_suite(out).cases) == 2  # Dyno accepts it


def test_decision_is_recorded_before_the_alias_moves(world, monkeypatch) -> None:
    reg, _, _, _ = world

    def broken_log(kind, record, metrics):
        raise RuntimeError("tracking server down")

    monkeypatch.setattr(p, "log_decision", broken_log)
    with pytest.raises(RuntimeError):
        p.promote("v1", settings(), RULES)
    assert "production" not in reg.aliases  # no record -> no promotion
