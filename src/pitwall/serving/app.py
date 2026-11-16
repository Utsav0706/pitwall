"""The routing service: Gearbox asks "which tier for this prompt?", Pitwall answers.

    POST /route   {"prompt": "...", "model": "production"}  -> tier, reason, version, probs
    GET  /health  ready when Ollama answers and the default model is loaded
    GET  /models  which version each reference is serving
    GET  /metrics Prometheus metrics (pitwall.serving.metrics)

Run: `pitwall serve` (port 8003). Endpoints are plain `def`: the Ollama call blocks, so
FastAPI runs each request in its worker threads instead of stalling the event loop.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated

import httpx
from fastapi import FastAPI, Header, HTTPException
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from pitwall import __version__
from pitwall.config import Settings, get_settings
from pitwall.ollama import OllamaError, OllamaRouter, choose_tier
from pitwall.serving.db import RouteLog, open_engine
from pitwall.serving.metrics import ServiceMetrics
from pitwall.serving.models import MlflowResolver, ModelManager, ModelUnavailable
from pitwall.textformat import render

log = logging.getLogger("pitwall.serving")


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=200_000)
    model: str | None = Field(
        default=None, description="production, candidate, or a version such as v1"
    )


class RouteResponse(BaseModel):
    tier: str
    reason: str
    model_version: int
    request_id: str
    probs: dict[str, float]
    latency_ms: float
    fallback: bool


def create_app(
    settings: Settings | None = None,
    manager: ModelManager | None = None,
    engine: Engine | None = None,
    ollama_client: httpx.Client | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    client = ollama_client or httpx.Client(base_url=settings.ollama_base_url, timeout=60.0)
    manager = manager or ModelManager(
        MlflowResolver(), client, settings.model_cache_dir, settings.alias_refresh_s
    )
    engine = engine or open_engine(settings.database_url)
    app = FastAPI(title="Pitwall router", version=__version__)
    metrics = ServiceMetrics()
    app.mount("/metrics", make_asgi_app(registry=metrics.registry))
    app.state.metrics = metrics

    @app.post("/route", response_model=RouteResponse)
    def route(
        body: RouteRequest,
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> RouteResponse:
        request_id = (x_request_id or uuid.uuid4().hex)[:64]
        ref = body.model or settings.pitwall_model
        try:
            model = manager.get(ref)
        except ModelUnavailable as exc:
            metrics.errors.labels("model_unavailable").inc()
            raise HTTPException(503, str(exc)) from exc
        template = model.template
        seen = body.prompt[: template.max_prompt_chars]
        text = render(template.system, template.render_user(body.prompt))
        start = time.perf_counter()
        try:
            answer = OllamaRouter("", model.ollama_name, client=client).answer(text)
        except OllamaError as exc:
            metrics.errors.labels("ollama").inc()
            raise HTTPException(503, str(exc)) from exc
        seconds = time.perf_counter() - start
        latency = round(seconds * 1000, 1)
        tier, reason = choose_tier(answer, model.threshold)
        metrics.record(ref, model.version, tier, answer.no_answer, seconds, len(body.prompt))
        with Session(engine) as session:
            session.add(
                RouteLog(
                    request_id=request_id,
                    created_at=datetime.now(UTC),
                    model_ref=ref[:32],
                    model_version=model.version,
                    tier=tier,
                    p_s=answer.probs[0],
                    p_m=answer.probs[1],
                    p_l=answer.probs[2],
                    fallback=answer.no_answer,
                    latency_ms=latency,
                    prompt_chars=len(body.prompt),
                    prompt_cut=len(body.prompt) > template.max_prompt_chars,
                    prompt=seen,
                )
            )
            session.commit()
        log.info("route", extra={"request_id": request_id, "tier": tier, "version": model.version})
        return RouteResponse(
            tier=tier,
            reason=reason,
            model_version=model.version,
            request_id=request_id,
            probs=dict(zip("SML", answer.probs, strict=True)),
            latency_ms=latency,
            fallback=answer.no_answer,
        )

    @app.get("/health")
    def health() -> dict:
        checks: dict[str, str] = {}
        try:
            client.get("/api/version", timeout=3).raise_for_status()
            checks["ollama"] = "ok"
        except httpx.HTTPError as exc:
            checks["ollama"] = f"down: {type(exc).__name__}"
        try:
            checks["model"] = f"v{manager.get(settings.pitwall_model).version}"
        except ModelUnavailable as exc:
            checks["model"] = f"unavailable: {exc}"[:300]
        ready = checks["ollama"] == "ok" and checks["model"].startswith("v")
        if not ready:
            raise HTTPException(503, {"status": "not ready", **checks})
        return {"status": "ok", "default_model": settings.pitwall_model, **checks}

    @app.get("/models")
    def models() -> dict:
        return {
            ref: {
                "version": m.version,
                "threshold": m.threshold,
                "bundle": m.manifest.name,
                "base_model": m.manifest.base_model,
                "data_version": m.manifest.data_version,
                "prompt_template": m.manifest.prompt_template,
            }
            for ref, m in manager.loaded().items()
        }

    return app
