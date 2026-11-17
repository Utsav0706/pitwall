"""`pitwall retrain`: the loop from new data to a promoted model, as one command per half.

    pitwall retrain prepare --train train_full    # check the data, print the Colab steps
    (train on Colab: notebooks/train_router_colab.ipynb with TRAIN = "train_full")
    pitwall retrain finish bundles/ft-train_full  # evaluate -> register -> candidate ->
                                                  # gate (dry run) -> canary -> promote

The middle step is the one thing that can't run here: training needs a GPU, and the free
one is on Colab. Everything around it is the same commands a person would type, in the
order the spec asks for ("Dyno gate passes -> canary -> promoted"), stopping at the
first failure so a rejected model never reaches the canary or production.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

COLAB_URL = (
    "https://colab.research.google.com/github/Utsav0706/pitwall/blob/main/"
    "notebooks/train_router_colab.ipynb"
)


class RetrainError(RuntimeError):
    pass


@dataclass
class Step:
    name: str
    args: list[str]
    paid: bool = False


def pitwall_executable() -> str:
    name = "pitwall.exe" if sys.platform == "win32" else "pitwall"
    return str(Path(sys.executable).parent / name)


def finish_steps(bundle: Path, version_ref: str, canary_n: int, cap_gbp: float) -> list[Step]:
    """The steps after `register` (which decides the version number)."""
    steps = [
        Step("candidate", ["candidate", version_ref]),
        Step("gate (dry run)", ["promote", version_ref, "--dry-run"]),
    ]
    if canary_n > 0:
        steps.append(
            Step(
                "canary",
                ["canary", "replay", "--n", str(canary_n), "--cap-gbp", str(cap_gbp)],
                paid=True,
            )
        )
    steps.append(Step("promote", ["promote", version_ref]))
    return steps


def registered_version(output: str) -> str:
    """'registered: pitwall-router version 2 (...)' -> 'v2'."""
    match = re.search(r"pitwall-router version (\d+)", output)
    if not match:
        raise RetrainError(f"could not read the registered version from: {output[-300:]}")
    return f"v{match.group(1)}"


def canary_file(output: str) -> Path | None:
    match = re.search(r"-> (\S+canary-\S+\.jsonl)", output)
    return Path(match.group(1)) if match else None


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def default_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([pitwall_executable(), *args], capture_output=True, text=True)


def finish(
    bundle: Path,
    canary_n: int,
    cap_gbp: float,
    echo: Callable[[str], None],
    run: Runner = default_runner,
) -> str:
    """Run the second half of the loop; returns the promoted version (or raises)."""

    def step(name: str, args: list[str]) -> str:
        echo(f"== {name}: pitwall {' '.join(args)}")
        result = run(args)
        out = (result.stdout or "") + (result.stderr or "")
        echo(out.rstrip())
        if result.returncode != 0:
            raise RetrainError(f"{name} failed (exit {result.returncode}); stopping here")
        return out

    step("bundle check", ["bundle", "check", str(bundle)])
    step("evaluate", ["evaluate", str(bundle)])
    version = registered_version(step("register", ["register", str(bundle)]))
    for s in finish_steps(bundle, version, canary_n, cap_gbp):
        out = step(s.name, s.args)
        if s.name == "canary" and (rows := canary_file(out)) is not None:
            step("canary report", ["canary", "report", str(rows)])
    return version
