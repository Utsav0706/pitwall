from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from prometheus_client import generate_latest

from pitwall.serving.metrics import ServiceMetrics
from test_service import setup  # noqa: F401  (pytest fixture)

ROOT = Path(__file__).resolve().parents[1]
METRIC = re.compile(r"\bpitwall_[a-z_]+")


def exported_names() -> set[str]:
    m = ServiceMetrics()
    m.record("production", 1, "M", False, 0.1, 120)
    m.errors.labels("ollama").inc()
    text = generate_latest(m.registry).decode()
    return {
        line.split("{")[0].split(" ")[0]
        for line in text.splitlines()
        if line.startswith("pitwall_")
    }


def dashboard() -> dict:
    return json.loads(
        (ROOT / "monitoring/grafana/dashboards/pitwall.json").read_text(encoding="utf-8")
    )


def test_every_dashboard_and_alert_query_uses_real_metrics() -> None:
    names = exported_names()
    exprs = [t["expr"] for p in dashboard()["panels"] for t in p["targets"]]
    rules = yaml.safe_load((ROOT / "monitoring/prometheus/rules.yml").read_text(encoding="utf-8"))
    exprs += [r["expr"] for g in rules["groups"] for r in g["rules"]]
    used = {name for e in exprs for name in METRIC.findall(e)}
    assert used, "no pitwall metrics referenced"
    assert used <= names, f"queries use metrics the service does not export: {used - names}"


def test_dashboard_uses_the_provisioned_datasource() -> None:
    ds = yaml.safe_load(
        (ROOT / "monitoring/grafana/provisioning/datasources/prometheus.yml").read_text(
            encoding="utf-8"
        )
    )["datasources"][0]
    for panel in dashboard()["panels"]:
        assert panel["datasource"]["uid"] == ds["uid"]
        assert all(t["datasource"]["uid"] == ds["uid"] for t in panel["targets"])
    ids = [p["id"] for p in dashboard()["panels"]]
    assert len(ids) == len(set(ids))


def test_latency_limit_matches_the_promotion_rule() -> None:
    from pitwall.config import load_config

    limit_s = load_config(ROOT / "config.yaml").promotion.p95_ms_limit / 1000
    rules = (ROOT / "monitoring/prometheus/rules.yml").read_text(encoding="utf-8")
    assert f"> {limit_s:g}" in rules  # PitwallSlowRouting
    lat = next(p for p in dashboard()["panels"] if p["id"] == 6)
    assert lat["fieldConfig"]["defaults"]["thresholds"]["steps"][1]["value"] == limit_s


def test_model_info_follows_an_alias_move() -> None:
    m = ServiceMetrics()
    m.record("production", 1, "M", False, 0.1, 10)
    m.record("production", 2, "S", False, 0.1, 10)
    text = generate_latest(m.registry).decode()
    assert 'pitwall_model_info{model_ref="production",version="v2"} 1.0' in text
    assert 'version="v1"' not in text.split("pitwall_model_info")[-1].split("\n#")[0]


def test_route_requests_show_up_on_metrics(setup) -> None:  # noqa: F811
    client = setup[0]
    client.post("/route", json={"prompt": "Write a haiku"})
    client.post("/route", json={"prompt": "hi", "model": "candidate"})  # no such alias -> 503
    text = client.get("/metrics/").text
    labels = 'fallback="false",model_ref="production",tier="M",version="v1"'
    assert f"pitwall_route_requests_total{{{labels}}} 1.0" in text
    assert 'pitwall_route_errors_total{reason="model_unavailable"} 1.0' in text
    assert "pitwall_route_latency_seconds_bucket" in text


def test_fail_safe_share_is_zero_not_empty_when_nothing_failed() -> None:
    # The fallback="true" series only exists after a first fail-safe; without the
    # `or vector(0)` the panel shows "No data" instead of 0%.
    for panel in dashboard()["panels"]:
        for t in panel["targets"]:
            if 'fallback="true"' in t["expr"]:
                assert "or vector(0)" in t["expr"], panel["title"]
