"""The MLflow model registry for the router: one registered model, a version per bundle.

A version points at a copy of the bundle inside its evaluation run's artifacts, not at
bundles/ (a scratch folder that can be deleted or overwritten): the registry must own
the exact file it may later promote. Aliases (production, candidate) are set by the
promotion step (M6), never here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pitwall.config import get_settings
from pitwall.finetune import bundle as bundle_mod
from pitwall.tracking import EXPERIMENT

REGISTERED_MODEL = "pitwall-router"
VERSION_METRICS = ("accuracy", "under_rate", "pass_rate", "latency_p95_ms")


class RegistryError(RuntimeError):
    pass


def client() -> Any:
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    return mlflow.MlflowClient()


def evaluation_run(c: Any, gguf_sha256: str) -> Any:
    """The newest finetune run that evaluated this exact GGUF."""
    experiment = c.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RegistryError("no MLflow experiment yet; run `pitwall evaluate` first")
    runs = c.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.kind = 'finetune' and tags.gguf_sha256 = '{gguf_sha256}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RegistryError("this bundle has no evaluation run; run `pitwall evaluate` first")
    return runs[0]


def existing_version(c: Any, gguf_sha256: str) -> Any | None:
    for v in c.search_model_versions(f"name = '{REGISTERED_MODEL}'"):
        if v.tags.get("gguf_sha256") == gguf_sha256:
            return v
    return None


def register(directory: Path) -> tuple[Any, bool]:
    """(model version, created). Registering the same GGUF twice returns the first one."""
    manifest = bundle_mod.load_manifest(directory)
    if manifest.gguf is None:
        raise RegistryError("bundle has no GGUF")
    c = client()
    found = existing_version(c, manifest.gguf.sha256)
    if found is not None:
        return found, False
    run = evaluation_run(c, manifest.gguf.sha256)
    c.log_artifacts(run.info.run_id, str(directory), "bundle")
    try:
        c.get_registered_model(REGISTERED_MODEL)
    except Exception:  # MLflow raises RestException / MlflowException when missing
        c.create_registered_model(
            REGISTERED_MODEL, description="Fine-tuned router: prompt -> tier S/M/L (Pitwall)."
        )
    tags = {
        "bundle": manifest.name,
        "gguf_sha256": manifest.gguf.sha256,
        "base_model": manifest.base_model,
        "data_version": manifest.data_version,
        **{m: f"{run.data.metrics[m]:.4f}" for m in VERSION_METRICS if m in run.data.metrics},
    }
    version = c.create_model_version(
        REGISTERED_MODEL,
        source=f"{run.info.artifact_uri}/bundle",
        run_id=run.info.run_id,
        tags=tags,
        description=f"{manifest.name}: {manifest.base_model} + LoRA, {manifest.gguf.quant}",
    )
    return version, True
