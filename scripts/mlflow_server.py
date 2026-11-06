"""Start the local MLflow server Pitwall logs to (no Docker needed).

    python scripts/mlflow_server.py          # http://127.0.0.1:5000

Runs and the model registry live in mlflow.db (SQLite: the registry needs a database,
the plain file store can't hold it); files such as predictions and weights go to
mlartifacts/. Both are git-ignored. Port 5000, as MLFLOW_TRACKING_URI in .env.example.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}",
        "--artifacts-destination",
        (ROOT / "mlartifacts").as_uri(),
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--workers",
        "1",  # one process is plenty locally and keeps RAM low
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
