"""Which model answers a request, and getting it ready in Ollama.

A request names a model by reference: an MLflow alias of `pitwall-router` (production,
candidate) or a fixed version (v1, v2...). The manager

1. resolves the reference in the registry (re-checked every `refresh_s`, so promoting
   or rolling back = moving an alias, no restart);
2. downloads that version's bundle once into the cache and checks it (bundle check,
   GGUF hash = the registry's tag);
3. refuses a bundle whose prompt template hash doesn't match its training run, so the
   model never sees different instructions from the ones it learned;
4. creates the model in Ollama (`pitwall-router-v<N>`) over the HTTP API.

Loaded versions are kept, so an alias that moves back (rollback) is instant.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from pitwall import ollama
from pitwall.finetune import bundle as bundle_mod
from pitwall.prompts import PROMPTS_DIR, PromptTemplate, get_template

VERSION_REF = re.compile(r"^v(\d+)$")


class ModelUnavailable(RuntimeError):
    """No usable model for this reference (503)."""


@dataclass(frozen=True)
class Resolved:
    version: int
    source: str  # artifact URI of the bundle
    gguf_sha256: str | None


class Resolver(Protocol):
    def resolve(self, ref: str) -> Resolved: ...

    def fetch(self, resolved: Resolved, dest: Path) -> Path: ...


class MlflowResolver:
    """The real registry. Imported lazily: mlflow is slow to import."""

    def resolve(self, ref: str) -> Resolved:
        from pitwall.registry import REGISTERED_MODEL, client

        c = client()
        match = VERSION_REF.match(ref)
        try:
            if match:
                mv = c.get_model_version(REGISTERED_MODEL, match.group(1))
            else:
                mv = c.get_model_version_by_alias(REGISTERED_MODEL, ref)
        except Exception as exc:  # MlflowException for unknown alias/version, or no server
            raise ModelUnavailable(
                f"no {REGISTERED_MODEL} model for {ref!r} in MLflow ({type(exc).__name__}: "
                f"{str(exc)[:150]})"
            ) from exc
        return Resolved(int(mv.version), mv.source, mv.tags.get("gguf_sha256"))

    def fetch(self, resolved: Resolved, dest: Path) -> Path:
        import mlflow

        dest.mkdir(parents=True, exist_ok=True)
        return Path(
            mlflow.artifacts.download_artifacts(artifact_uri=resolved.source, dst_path=str(dest))
        )


@dataclass(frozen=True)
class LoadedModel:
    version: int
    ollama_name: str
    threshold: float
    template: PromptTemplate
    manifest: bundle_mod.Manifest


class ModelManager:
    def __init__(
        self,
        resolver: Resolver,
        ollama_client: httpx.Client,
        cache_dir: Path,
        refresh_s: float = 30.0,
        prompts_dir: Path = PROMPTS_DIR,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._resolver = resolver
        self._ollama = ollama_client
        self._cache = cache_dir
        self._refresh_s = refresh_s
        self._prompts = prompts_dir
        self._clock = clock
        self._by_version: dict[int, LoadedModel] = {}
        self._by_ref: dict[str, tuple[int, float]] = {}  # ref -> (version, looked up at)
        self._lock = threading.Lock()

    def get(self, ref: str) -> LoadedModel:
        with self._lock:
            now = self._clock()
            cached = self._by_ref.get(ref)
            if cached and now - cached[1] < self._refresh_s:
                return self._by_version[cached[0]]
            resolved = self._resolver.resolve(ref)
            model = self._by_version.get(resolved.version) or self._load(resolved)
            self._by_version[resolved.version] = model
            self._by_ref[ref] = (resolved.version, now)
            return model

    def loaded(self) -> dict[str, LoadedModel]:
        with self._lock:
            return {ref: self._by_version[v] for ref, (v, _) in self._by_ref.items()}

    def _load(self, resolved: Resolved) -> LoadedModel:
        directory = self._cache / f"v{resolved.version}" / "bundle"
        if not (directory / bundle_mod.MANIFEST).exists():
            fetched = self._resolver.fetch(resolved, self._cache / f"v{resolved.version}")
            directory = fetched
        problems = bundle_mod.check(directory)
        if problems:
            raise ModelUnavailable(f"v{resolved.version} bundle failed its check: {problems}")
        manifest = bundle_mod.load_manifest(directory)
        assert manifest.gguf is not None  # check() requires it
        if resolved.gguf_sha256 and manifest.gguf.sha256 != resolved.gguf_sha256:
            raise ModelUnavailable(f"v{resolved.version}: GGUF differs from the registry's hash")
        try:
            template = get_template(manifest.prompt_template, self._prompts)
        except KeyError as exc:
            raise ModelUnavailable(f"v{resolved.version}: {exc}") from exc
        if template.sha256 != manifest.prompt_sha256:
            raise ModelUnavailable(
                f"v{resolved.version} was trained with prompt {manifest.prompt_template} "
                f"{manifest.prompt_sha256[:12]}, but prompts/ has {template.sha256[:12]}"
            )
        name = f"pitwall-router-v{resolved.version}"
        try:
            ollama.create(self._ollama, name, directory / manifest.gguf.file, manifest.gguf.sha256)
        except ollama.OllamaError as exc:
            raise ModelUnavailable(str(exc)) from exc
        return LoadedModel(
            version=resolved.version,
            ollama_name=name,
            threshold=float(manifest.chosen["threshold"]),
            template=template,
            manifest=manifest,
        )
