from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from pitwall.config import Settings
from pitwall.serving.app import create_app
from pitwall.serving.db import RouteLog, open_engine
from pitwall.serving.models import ModelManager, ModelUnavailable, Resolved
from test_finetune import make_bundle

ROOT = Path(__file__).resolve().parents[1]


class FakeOllama:
    """Ollama's HTTP API: blob upload, create, generate with logprobs, version."""

    def __init__(self, top: list[dict] | None = None, down: bool = False) -> None:
        self.top = (
            top
            if top is not None
            else [
                {"token": " M", "logprob": -0.2},
                {"token": " S", "logprob": -1.8},
                {"token": " L", "logprob": -3.0},
            ]
        )
        self.down = down
        self.created: list[str] = []
        self.prompts: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        path = request.url.path
        if path.startswith("/api/blobs"):
            return httpx.Response(200 if request.method == "HEAD" else 201)
        if path == "/api/create":
            self.created.append(json.loads(request.content)["model"])
            return httpx.Response(200, json={"status": "success"})
        if path == "/api/generate":
            self.prompts.append(json.loads(request.content)["prompt"])
            return httpx.Response(
                200, json={"logprobs": [{"token": "x", "top_logprobs": self.top}]}
            )
        if path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        return httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(self.handler))


class FakeResolver:
    """The registry: aliases -> versions, versions -> bundle folders."""

    def __init__(self, bundles: dict[int, Path], aliases: dict[str, int]) -> None:
        self.bundles = bundles
        self.aliases = aliases
        self.lookups = 0

    def resolve(self, ref: str) -> Resolved:
        self.lookups += 1
        version = int(ref[1:]) if ref.startswith("v") else self.aliases.get(ref)
        if version is None or version not in self.bundles:
            raise ModelUnavailable(f"no model for {ref!r}")
        sha = json.loads((self.bundles[version] / "bundle.json").read_text())["gguf"]["sha256"]
        return Resolved(version, str(self.bundles[version]), sha)

    def fetch(self, resolved: Resolved, dest: Path) -> Path:
        target = dest / "bundle"
        shutil.copytree(resolved.source, target)
        return target


def real_template_bundle(tmp_path: Path, name: str = "ft-a") -> Path:
    """A test bundle that names the repo's real router:1 template and its hash."""
    from pitwall.prompts import get_template

    directory = make_bundle(tmp_path / name)
    data = json.loads((directory / "bundle.json").read_text())
    data["prompt_template"] = "router:1"
    data["prompt_sha256"] = get_template("router:1", ROOT / "prompts").sha256
    data["chosen"]["threshold"] = 0.6
    (directory / "bundle.json").write_text(json.dumps(data))
    return directory


@pytest.fixture
def setup(tmp_path: Path):
    ollama = FakeOllama()
    bundles = {1: real_template_bundle(tmp_path, "b1"), 2: real_template_bundle(tmp_path, "b2")}
    resolver = FakeResolver(bundles, {"production": 1})
    clock = {"t": 0.0}
    manager = ModelManager(
        resolver,
        ollama.client(),
        tmp_path / "cache",
        refresh_s=30,
        prompts_dir=ROOT / "prompts",
        clock=lambda: clock["t"],
    )
    engine = open_engine(f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    settings = Settings(_env_file=None, pitwall_model="production")
    app = create_app(settings, manager, engine, ollama.client())
    return TestClient(app), ollama, resolver, clock, engine


def test_route_answers_and_logs(setup) -> None:
    client, ollama, _, _, engine = setup
    r = client.post("/route", json={"prompt": "Write a haiku"}, headers={"x-request-id": "gb-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tier"] == "M"  # P(S) ~0.17 < 0.6, P(S)+P(M) >= 0.6
    assert body["model_version"] == 1
    assert body["request_id"] == "gb-1"
    assert sum(body["probs"].values()) == pytest.approx(1.0)
    assert not body["fallback"]
    # the model saw exactly the training layout
    assert ollama.prompts[0].endswith("Request:\nWrite a haiku\n\nTier:")
    assert ollama.created == ["pitwall-router-v1"]
    with Session(engine) as s:
        row = s.scalars(select(RouteLog)).one()
    assert (row.request_id, row.tier, row.model_ref, row.prompt) == (
        "gb-1",
        "M",
        "production",
        "Write a haiku",
    )


def test_long_prompt_is_cut_and_flagged(setup) -> None:
    client, ollama, _, _, engine = setup
    client.post("/route", json={"prompt": "x" * 3500})
    with Session(engine) as s:
        row = s.scalars(select(RouteLog)).one()
    assert row.prompt_cut and row.prompt_chars == 3500 and len(row.prompt) == 3000


def test_fixed_version_and_missing_alias(setup) -> None:
    client, _, _, _, _ = setup
    assert client.post("/route", json={"prompt": "hi", "model": "v2"}).json()["model_version"] == 2
    r = client.post("/route", json={"prompt": "hi", "model": "candidate"})
    assert r.status_code == 503
    assert "candidate" in r.json()["detail"]


def test_alias_move_is_picked_up_after_refresh(setup) -> None:
    client, _, resolver, clock, _ = setup
    assert client.post("/route", json={"prompt": "a"}).json()["model_version"] == 1
    resolver.aliases["production"] = 2  # a promotion
    clock["t"] = 10
    assert client.post("/route", json={"prompt": "a"}).json()["model_version"] == 1  # cached
    clock["t"] = 31
    assert client.post("/route", json={"prompt": "a"}).json()["model_version"] == 2
    assert client.get("/models").json()["production"]["version"] == 2


def test_no_answer_falls_back_to_m(setup) -> None:
    client, ollama, _, _, _ = setup
    ollama.top = [{"token": " the", "logprob": -0.1}]
    body = client.post("/route", json={"prompt": "?"}).json()
    assert body["tier"] == "M" and body["fallback"]


def test_ollama_down_is_503(setup) -> None:
    client, ollama, _, _, _ = setup
    client.post("/route", json={"prompt": "warm"})  # load while up
    ollama.down = True
    assert client.post("/route", json={"prompt": "hi"}).status_code == 503
    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["detail"]["ollama"].startswith("down")


def test_health_ok_and_empty_prompt(setup) -> None:
    client, _, _, _, _ = setup
    assert client.get("/health").json()["model"] == "v1"
    assert client.post("/route", json={"prompt": ""}).status_code == 422


def test_template_hash_mismatch_is_refused(tmp_path: Path) -> None:
    directory = real_template_bundle(tmp_path)
    data = json.loads((directory / "bundle.json").read_text())
    data["prompt_sha256"] = "0" * 64
    (directory / "bundle.json").write_text(json.dumps(data))
    manager = ModelManager(
        FakeResolver({1: directory}, {"production": 1}),
        FakeOllama().client(),
        tmp_path / "cache",
        prompts_dir=ROOT / "prompts",
    )
    with pytest.raises(ModelUnavailable, match="trained with prompt"):
        manager.get("production")


def test_registry_hash_mismatch_is_refused(tmp_path: Path) -> None:
    directory = real_template_bundle(tmp_path)

    class WrongHash(FakeResolver):
        def resolve(self, ref: str) -> Resolved:
            r = super().resolve(ref)
            return Resolved(r.version, r.source, "f" * 64)

    manager = ModelManager(
        WrongHash({1: directory}, {"production": 1}),
        FakeOllama().client(),
        tmp_path / "cache",
        prompts_dir=ROOT / "prompts",
    )
    with pytest.raises(ModelUnavailable, match="registry's hash"):
        manager.get("production")


def test_migrations_match_the_models(tmp_path: Path) -> None:
    from alembic import command

    from pitwall.serving.db import alembic_config

    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    open_engine(url)
    command.check(alembic_config(url))  # raises if models and migrations differ
