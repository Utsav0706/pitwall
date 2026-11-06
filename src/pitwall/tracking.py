"""Every MLflow run Pitwall makes carries the same tags, so "which data, which prompt,
which code?" is answered from MLflow alone, for any run, months later.

- data_version: the hash DVC recorded for data/derived in dvc.lock.
- prompt_template / prompt_sha256: from data/derived/manifest.json.
- git_commit / git_dirty: dirty = uncommitted changes, so the commit alone may not
  reproduce the run.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from pitwall.config import get_settings

EXPERIMENT = "router"
DVC_LOCK = Path("dvc.lock")
DERIVED = Path("data/derived")


def data_version(lock: Path = DVC_LOCK, out: str = "data/derived") -> str:
    """DVC's hash of the built data, or 'unknown' when the lock is missing."""
    if not lock.exists():
        return "unknown"
    stages = (yaml.safe_load(lock.read_text(encoding="utf-8")) or {}).get("stages", {})
    for stage in stages.values():
        for o in stage.get("outs", []):
            if o.get("path") == out:
                return str(o.get("md5", "unknown"))
    return "unknown"


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, dirty


def standard_tags(derived: Path = DERIVED, lock: Path = DVC_LOCK) -> dict[str, str]:
    commit, dirty = git_state()
    tags = {"data_version": data_version(lock), "git_commit": commit, "git_dirty": str(dirty)}
    manifest_path = derived / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tags["prompt_template"] = manifest["prompt_template"]
        tags["prompt_sha256"] = manifest["prompt_template_sha256"]
    return tags


@contextmanager
def run(name: str, kind: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """An MLflow run in the 'router' experiment with Pitwall's standard tags.

    kind: baseline | reference | finetune (M4), so the UI can filter by it.
    """
    import mlflow  # heavy import, only when a run is actually logged

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as active:
        mlflow.set_tags({**standard_tags(), "kind": kind, **(tags or {})})
        yield active


def list_runs() -> list[dict[str, Any]]:
    """Every run in the experiment, newest first, as plain dicts."""
    import mlflow

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        return []
    found = mlflow.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], output_format="list"
    )
    return [
        {
            "name": r.info.run_name,
            "id": r.info.run_id,
            "tags": r.data.tags,
            "metrics": r.data.metrics,
        }
        for r in found
    ]
