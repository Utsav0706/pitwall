from __future__ import annotations

import json
from pathlib import Path

import httpx
import yaml

from pitwall import canary


def test_config_routes_through_pitwall_without_touching_the_rest(tmp_path: Path) -> None:
    gearbox = tmp_path / "gearbox"
    gearbox.mkdir()
    base = {
        "routing": {"router": "v3-embed", "canary": None, "v3_embed": {"artifact": "m.json"}},
        "cache": {"enabled": True},
    }
    (gearbox / "config.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    out = canary.write_config(gearbox, 10, "http://127.0.0.1:8003", tmp_path / "c.yaml")
    config = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert config["routing"]["router"] == "v4-finetuned"
    assert config["routing"]["canary"] == {"router": "v4-candidate", "percent": 10}
    assert config["routing"]["pitwall"]["url"] == "http://127.0.0.1:8003"
    assert config["routing"]["v3_embed"] == {"artifact": "m.json"} and config["cache"] == {
        "enabled": True
    }
    assert base["routing"]["router"] == "v3-embed"  # the original dict is untouched


def gearbox(cost: float = 0.001, routers: list[str] | None = None):
    routers = routers or ["v4-finetuned"]
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        router = routers[(len(sent) - 1) % len(routers)]
        return httpx.Response(
            200,
            json={"choices": []},
            headers={
                "x-gearbox-router": router,
                "x-gearbox-tier": "M",
                "x-gearbox-cost": f"{cost:.6f}",
                "x-gearbox-reason": "pitwall v1: ft: P(S)=0.27; pitwall request pw",
                "x-request-id": f"gbx-{len(sent)}",
            },
        )

    return httpx.Client(base_url="http://gb", transport=httpx.MockTransport(handler)), sent


def test_replay_records_rows_and_stops_at_the_cap(tmp_path: Path) -> None:
    client, sent = gearbox(cost=0.004)
    prompts = [(f"p{i}", f"prompt {i}") for i in range(10)]
    out = tmp_path / "rows.jsonl"
    n, spent, capped = canary.replay(client, prompts, cap_gbp=0.01, out=out)
    assert capped and n == 3  # 0.004, 0.008, 0.012 -> stop before a 4th request
    assert sent[0] == {
        "model": "auto",
        "messages": [{"role": "user", "content": "prompt 0"}],
        "max_tokens": canary.MAX_TOKENS,
    }
    rows = canary.read_rows(out)
    assert [r["request_id"] for r in rows] == ["gbx-1", "gbx-2", "gbx-3"]
    assert rows[0]["router"] == "v4-finetuned" and rows[0]["cost_gbp"] == 0.004


def test_summary_compares_both_sides() -> None:
    rows = [
        {
            "id": "a",
            "status": 200,
            "router": "v4-finetuned",
            "tier": "M",
            "cost_gbp": 0.001,
            "latency_ms": 900,
            "reason": "pitwall v1: ft",
        },
        {
            "id": "b",
            "status": 200,
            "router": "v4-finetuned",
            "tier": "M",
            "cost_gbp": 0.003,
            "latency_ms": 1100,
            "reason": "pitwall unavailable (ConnectError), using M",
        },
        {
            "id": "c",
            "status": 200,
            "router": "v4-candidate",
            "tier": "S",
            "cost_gbp": 0.0,
            "latency_ms": 500,
            "reason": "pitwall v2: ft",
        },
        {
            "id": "d",
            "status": 500,
            "router": "",
            "tier": "",
            "cost_gbp": 0.0,
            "latency_ms": 10,
            "reason": "",
        },
    ]
    scores = {"a": {"M": 5}, "b": {"M": 2}, "c": {"S": 4}}
    s = canary.summarise(rows, scores)
    prod, cand = s["v4-finetuned"], s["v4-candidate"]
    assert prod["requests"] == 2 and cand["requests"] == 1
    assert cand["share"] == 1 / 3
    assert prod["pass_rate"] == 0.5 and cand["pass_rate"] == 1.0
    assert prod["fail_safe"] == 1
    assert prod["cost_per_request_gbp"] == 0.002
    assert prod["versions"] == ["v1"] and cand["versions"] == ["v2"]
    text = canary.render(s, errors=1)
    assert "v4-candidate" in text and "1 requests failed" in text


def test_prompts_are_a_seeded_sample() -> None:
    from test_baseline import pair

    pairs = [pair(i, "S") for i in range(20)]
    a = canary.pick_prompts(pairs, 5, seed=1)
    assert a == canary.pick_prompts(pairs, 5, seed=1)
    assert a != canary.pick_prompts(pairs, 5, seed=2)
    assert a[0][1].startswith("prompt ")
