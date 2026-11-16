"""Run the routing service locally: http://127.0.0.1:8003

    python scripts/dev_api.py                 # serves the 'production' alias
    python scripts/dev_api.py --model v1      # a fixed registry version (before M6)

Runs from the repo folder, so prompts/, .env and pitwall.db resolve wherever it is
started from. Needs the MLflow server (scripts/mlflow_server.py) and Ollama running.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="alias or version to serve by default, e.g. v1")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.model:
        os.environ["PITWALL_MODEL"] = args.model
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from pitwall.cli import app

    sys.argv = ["pitwall", "serve"]
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
