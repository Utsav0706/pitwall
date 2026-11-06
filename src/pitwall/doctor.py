"""`pitwall doctor`: is everything Pitwall depends on installed and running?

Later milestones need heavy tools (MLflow, DVC, Evidently) and two local servers
(MLflow, Ollama). A missing one should give a clear line, not a stack trace halfway
through a training or promotion run. Only the standard library is used here, so the
doctor works even when the tools it checks are missing.

Each package is imported, not just looked up: Evidently 0.7.23 installs fine next to
pandas 3 but fails on import, and only an import shows that.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import metadata

from pitwall.config import Settings

# Python packages from the [ops] extra, and the milestone that first needs each.
PACKAGES: dict[str, str] = {"mlflow": "M2", "dvc": "M1", "evidently": "M9"}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def check_python() -> Check:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < (3, 12):
        return Check("python", False, f"{version}; Pitwall needs 3.12 or newer")
    return Check("python", True, version)


def check_package(name: str, needed_from: str) -> Check:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        return Check(
            name, False, f'not installed (needed from {needed_from}: pip install -e ".[ops]")'
        )
    # MLflow prints an INFO hint for coding agents on import; it is noise here.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        importlib.import_module(name)
    except Exception as exc:  # any import failure is the finding, whatever its type
        return Check(name, False, f"{version} installed but import fails: {exc}"[:200])
    return Check(name, True, version)


def check_url(name: str, url: str, timeout: float = 2.0) -> Check:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(500).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return Check(name, False, f"not reachable at {url} ({reason})")
    return Check(name, True, f"{url} {_short(body)}".strip())


def _short(body: str) -> str:
    # Ollama's /api/version answers {"version": "..."}; MLflow's /health answers "OK".
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:40]
    if isinstance(data, dict) and "version" in data:
        return f"version {data['version']}"
    return ""


def run_checks(settings: Settings) -> list[Check]:
    checks = [check_python()]
    checks += [check_package(name, needed) for name, needed in PACKAGES.items()]
    checks.append(check_url("mlflow server", settings.mlflow_tracking_uri.rstrip("/") + "/health"))
    checks.append(check_url("ollama", settings.ollama_base_url.rstrip("/") + "/api/version"))
    return checks
