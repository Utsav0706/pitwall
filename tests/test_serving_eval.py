from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from pitwall import bundle_eval, compare, ollama, registry
from pitwall.data import Pair


def pair(i: int, label: str) -> Pair:
    return Pair(
        id=f"p{i:03d}",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": label},
        ],
        label=label,  # type: ignore[arg-type]
        task_type="writing",
        source="dolly",
        part="old",
    )


def fake_ollama(top: list[dict] | None, seen: list[dict] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        logprobs = [{"token": top[0]["token"], "top_logprobs": top}] if top else []
        return httpx.Response(
            200, json={"response": "x", "logprobs": logprobs, "prompt_eval_count": 12}
        )

    return httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))


# --- reading the model's answer --------------------------------------------------------


def test_probs_renormalise_over_the_three_answers() -> None:
    top = [
        {"token": " M", "logprob": -0.1},
        {"token": " the", "logprob": -2.0},  # other tokens are ignored
        {"token": " S", "logprob": -1.0},
        {"token": " L", "logprob": -3.0},
    ]
    (s, m, l_), missing = ollama.probs_from_logprobs(top)
    assert not missing
    assert s + m + l_ == pytest.approx(1.0)
    assert m > s > l_
    assert m / s == pytest.approx(np.exp(0.9))


def test_tier_outside_top_k_counts_as_zero() -> None:
    (s, m, l_), missing = ollama.probs_from_logprobs([{"token": " S", "logprob": -0.5}])
    assert (s, m, l_) == (1.0, 0.0, 0.0)
    assert not missing


def test_no_answer_tokens_is_flagged() -> None:
    probs, missing = ollama.probs_from_logprobs([{"token": "hello", "logprob": -0.1}])
    assert missing and probs == (0.0, 0.0, 0.0)


def test_request_is_raw_cpu_one_token() -> None:
    seen: list[dict] = []
    router = ollama.OllamaRouter("http://ollama", "m", client=fake_ollama([], seen))
    answer = router.answer("text\n\nTier:")
    body = seen[0]
    assert body["raw"] is True and body["logprobs"] is True
    assert body["options"]["num_gpu"] == 0
    assert body["options"]["num_predict"] == 1
    assert answer.prompt_tokens == 12
    assert answer.no_answer  # empty top list


def test_http_error_is_an_ollama_error() -> None:
    client = httpx.Client(
        base_url="http://ollama",
        transport=httpx.MockTransport(lambda r: httpx.Response(404, json={"error": "no model"})),
    )
    with pytest.raises(ollama.OllamaError):
        ollama.OllamaRouter("http://ollama", "m", client=client).answer("x")


def test_model_name_and_modelfile(tmp_path: Path) -> None:
    assert ollama.model_name("ft-train_old") == "pitwall-ft-train-old"
    text = ollama.modelfile(tmp_path / "model.gguf")
    assert text.startswith('FROM "') and "PARAMETER temperature 0" in text


# --- routing a set of prompts -------------------------------------------------------------


def test_route_uses_threshold_and_falls_back_on_no_answer() -> None:
    answers = iter(
        [
            [{"token": " S", "logprob": -0.2}, {"token": " M", "logprob": -1.8}],  # P(S)~0.83
            [{"token": " S", "logprob": -0.9}, {"token": " M", "logprob": -0.5}],  # P(S)~0.4
            [{"token": "?", "logprob": -0.1}],  # no answer
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        top = next(answers)
        return httpx.Response(200, json={"logprobs": [{"token": "x", "top_logprobs": top}]})

    client = httpx.Client(base_url="http://o", transport=httpx.MockTransport(handler))
    router = ollama.OllamaRouter("http://o", "m", client=client)
    scored = bundle_eval.route(router, "ft", [pair(1, "S"), pair(2, "M"), pair(3, "L")], 0.6)
    assert [p.tier for p in scored.predictions] == ["S", "M", "M"]
    assert scored.no_answer == 1
    assert "fallback" in scored.predictions[2].reason


def test_latency_stats() -> None:
    stats = bundle_eval.latency_stats([float(x) for x in range(1, 101)])
    assert stats["latency_p50_ms"] == pytest.approx(50.5)
    assert stats["latency_max_ms"] == 100.0


# --- compare --------------------------------------------------------------------------------


def test_replay_command_runs_gearbox_in_its_own_venv(tmp_path: Path) -> None:
    py = (
        tmp_path
        / ".venv"
        / ("Scripts/python.exe" if compare.platform.system() == "Windows" else "bin/python")
    )
    py.parent.mkdir(parents=True)
    py.write_text("")
    cmd = compare.replay_command(tmp_path, Path("p.jsonl"), Path("out"))
    assert cmd[0] == str(py)
    assert cmd[1:4] == ["-m", "gearbox.eval", "offline"]


def test_missing_gearbox_venv_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="GEARBOX_DIR"):
        compare.gearbox_python(tmp_path)


def test_render_has_both_tables() -> None:
    runs = [
        {
            "name": "ft-x",
            "tags": {"kind": "finetune"},
            "metrics": {
                "accuracy": 0.5,
                "under_rate": 0.1,
                "over_rate": 0.4,
                "pass_rate": 0.78,
                "picks.S": 0,
                "picks.M": 119,
                "picks.L": 0,
                "hard.accuracy": 0.4,
                "latency_p50_ms": 120.0,
                "latency_p95_ms": 200.0,
            },
        }
    ]
    replay = {
        "always-M": {
            "quality_pct": 97.0,
            "pass_rate": 0.78,
            "cost_pct": 7.1,
            "saved_pct": 92.9,
            "cost_per_1k_requests_gbp": 0.5,
            "picks": {"S": 0, "M": 119, "L": 0},
        }
    }
    text = compare.render(runs, replay)
    assert (
        "| ft-x | finetune | 50.0% | 10.0% | 40.0% | 78.0% | 0/119/0 | 40.0% | 120 / 200 ms |"
        in text
    )
    assert "| always-M | 97.0% | 78.0% | 7.1% | 92.9% | GBP 0.500 | 0/119/0 |" in text


# --- registry ---------------------------------------------------------------------------------


def test_register_needs_an_evaluation_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    from pitwall.config import get_settings
    from test_finetune import make_bundle

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    get_settings.cache_clear()
    try:
        directory = make_bundle(tmp_path)
        mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
        with pytest.raises(registry.RegistryError, match="pitwall evaluate"):
            registry.register(directory)

        exp = mlflow.create_experiment("router", artifact_location=(tmp_path / "art").as_uri())
        sha = json.loads((directory / "bundle.json").read_text())["gguf"]["sha256"]
        with mlflow.start_run(experiment_id=exp, run_name="ft-test"):
            mlflow.set_tags({"kind": "finetune", "gguf_sha256": sha})
            mlflow.log_metrics({"accuracy": 0.5, "latency_p95_ms": 210.0})
        version, created = registry.register(directory)
        assert created and str(version.version) == "1"
        assert version.tags["gguf_sha256"] == sha
        assert version.tags["accuracy"] == "0.5000"
        again, created_again = registry.register(directory)
        assert not created_again and str(again.version) == "1"  # same GGUF -> same version
        assert (tmp_path / "art").exists()
    finally:
        get_settings.cache_clear()
