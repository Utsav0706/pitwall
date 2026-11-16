"""Start Gearbox (port 8002) in canary mode: router v4-finetuned (Pitwall's production
model) with v4-candidate (Pitwall's candidate) on a share of requests.

Run it with GEARBOX's Python, from anywhere:

    ..\\gearbox\\.venv\\Scripts\\python.exe scripts\\gearbox_canary.py

It points Gearbox at runs/gearbox-canary.yaml (write it first: `pitwall canary config`)
and starts Gearbox's own dev server from Gearbox's repo. Gearbox's files are not touched;
it imports nothing from Pitwall (the repos stay separate).
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

PITWALL = Path(__file__).resolve().parents[1]
CONFIG = PITWALL / "runs" / "gearbox-canary.yaml"
GEARBOX = Path(os.environ.get("GEARBOX_DIR", PITWALL.parent / "gearbox")).resolve()


def main() -> int:
    if not CONFIG.exists():
        print(f"{CONFIG} not found: run `pitwall canary config` first", file=sys.stderr)
        return 1
    os.environ["CONFIG_PATH"] = str(CONFIG)
    runpy.run_path(str(GEARBOX / "scripts" / "dev_api.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
