"""Run the whole loop in order, one command after another:

  1. normal traffic          -> drift check: no drift
  2. a new kind of traffic   -> drift check: DRIFT (alert PitwallDrift fires)
  3. retrain finish          -> evaluate, register, candidate, gate, canary, promote
  4. the new traffic again   -> drift check against the new model's data: no drift
  5. rollback                -> production back on the previous version
  6. promote again           -> the gate passes again; the new model is back

    python scripts/run_loop.py --bundle bundles/ft-train_full --yes

Needs: Pitwall's service, MLflow and Ollama; Gearbox in canary mode for step 3
(scripts/gearbox_canary.py); Prometheus + Grafana to watch it. Step 3's canary is PAID
(~GBP 0.01, capped); everything else is free. Steps run in order and stop on the first
unexpected result. The training itself (Colab) happens before this script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PITWALL = str(
    Path(sys.executable).parent / ("pitwall.exe" if sys.platform == "win32" else "pitwall")
)


def run(title: str, args: list[str], expect: int | tuple[int, ...] = 0) -> None:
    print(f"\n{'=' * 78}\n{title}\n$ pitwall {' '.join(args)}\n{'=' * 78}", flush=True)
    code = subprocess.call([PITWALL, *args])
    allowed = expect if isinstance(expect, tuple) else (expect,)
    if code not in allowed:
        sys.exit(f"stopped: exit code {code}, expected {allowed}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--yes", action="store_true", help="allow the paid canary (step 3)")
    parser.add_argument("--n", type=int, default=200, help="requests per traffic step")
    args = parser.parse_args()

    run(
        "1. Normal traffic: the general test prompts",
        ["traffic", "--source", "test", "--part", "old", "--n", str(args.n), "--seed", "31"],
    )
    run("   Drift check (expect: no drift)", ["drift", "check", "--window", str(args.n)])
    run(
        "2. New traffic: SimpleQA / AIME / MMLU-Pro prompts",
        ["traffic", "--source", "pool_new", "--n", str(args.n), "--seed", "32"],
    )
    run(
        "   Drift check (expect: DRIFT, exit 3)",
        ["drift", "check", "--window", str(args.n)],
        expect=3,
    )
    finish = ["retrain", "finish", str(args.bundle)] + (
        ["--yes"] if args.yes else ["--canary-n", "0"]
    )
    run("3. Retrain on old + new prompts: evaluate, register, gate, canary, promote", finish)
    run(
        "4. The new traffic again, now served by the retrained model",
        ["traffic", "--source", "pool_new", "--n", str(args.n), "--seed", "33"],
    )
    run(
        "   Drift check against the new model's training data (expect: no drift)",
        ["drift", "check", "--window", str(args.n)],
    )
    run("5. Rollback: production back on the previous version", ["rollback"])
    run("6. Promote the retrained model again (the gate runs again)", ["promote", "candidate"])
    print("\nLoop complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
