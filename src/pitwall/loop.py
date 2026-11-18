"""Closing the loop: bad answers from production become test cases and training data.

Two sources of failures:
- feedback: thumbs-down on a routing decision (POST /feedback);
- sample-eval: a sample of recent production prompts is answered through Gearbox and
  judged by Dyno's calibrated `quality` judge; answers that fail are failures. (The
  spec's "nightly" sample, as a command; scheduling it is left to cron or a CI job.)

`pitwall failures import` writes both to one JSONL file in Dyno's documented format
(docs/integration.md: {id, input, output, reason, tags, source}), then
- `dyno import-failures` turns it into new Dyno cases, deduplicated against the earlier
  imports, in data/production/cases/;
- the prompts go into the training queue, data/production/to_label.jsonl: they still
  need a label (which tier answers well) before a retrain can learn from them.

data/production/ is versioned with DVC, not git: these are production prompts, and they
don't belong in a public repo.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pitwall.config import Settings

PRODUCTION_DIR = Path("data/production")
CASES_DIR = PRODUCTION_DIR / "cases"
QUEUE = PRODUCTION_DIR / "to_label.jsonl"
# Per prompt: a Gearbox answer (Haiku, ~600 tokens out) + one quality judgement.
EST_ANSWER_GBP = 0.0005
EST_JUDGE_GBP = 0.003


class LoopError(RuntimeError):
    pass


@dataclass
class FailureRow:
    """One production failure, in Dyno's import-failures format."""

    id: str
    input: str
    reason: str
    source: str  # feedback | dyno-sample
    output: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v not in (None, [])})


def prompt_id(prompt: str) -> str:
    """The id Dyno gives a case without one (dyno.failures.case_id): prod- + 8 hex of
    the sha256 of the case-folded, whitespace-collapsed input."""
    normal = " ".join(prompt.casefold().split())
    return "prod-" + hashlib.sha256(normal.encode("utf-8")).hexdigest()[:8]


# --- feedback -----------------------------------------------------------------------------


def feedback_failures(engine: Any) -> list[FailureRow]:
    """Every thumbs-down, with the prompt the decision was made on."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from pitwall.serving.db import Feedback, RouteLog

    with Session(engine) as session:
        rows = session.execute(
            select(Feedback, RouteLog)
            .join(RouteLog, RouteLog.request_id == Feedback.request_id)
            .where(Feedback.rating == "down")
            .order_by(Feedback.created_at)
        ).all()
    return [
        FailureRow(
            id=prompt_id(route.prompt),
            input=route.prompt,
            reason=f"thumbs-down on routing to {route.tier} (v{route.model_version})"
            + (f": {fb.comment}" if fb.comment else ""),
            source="feedback",
            tags=[f"tier-{route.tier}"],
        )
        for fb, route in rows
    ]


# --- sample-eval -----------------------------------------------------------------------------


def estimate_gbp(n: int) -> float:
    return n * (EST_ANSWER_GBP + EST_JUDGE_GBP)


def sample_prompts(engine: Any, n: int, window: int = 500, seed: int = 7) -> list[str]:
    """n distinct prompts drawn from the last `window` production decisions."""
    from pitwall.drift import current_window, distinct

    pool = distinct(current_window(engine, "production", window))
    random.Random(seed).shuffle(pool)
    return pool[:n]


def write_sample_suite(prompts: list[str], path: Path) -> dict[str, str]:
    """A Dyno suite: these prompts, judged by `quality`. Returns {case id: prompt}."""
    cases = {prompt_id(p): p for p in prompts}
    suite = {
        "name": "pitwall-production-sample",
        "agent": "gearbox",
        "k": 1,
        "scorers": ["quality"],
        "cases": [{"id": cid, "input": p, "tags": ["production"]} for cid, p in cases.items()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(suite, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cases


def run_sample(suite: Path, settings: Settings, cap_gbp: float) -> tuple[str, float]:
    """`dyno run` the sample through Gearbox; (dyno run id, GBP Gearbox spent)."""
    from pitwall.promote import dyno_executable

    key = settings.anthropic_api_key
    if key is None:
        raise LoopError("sample-eval needs ANTHROPIC_API_KEY in Pitwall's .env (for the judge)")
    env = {
        **os.environ,
        "DATABASE_URL": settings.dyno_database_url,
        "GEARBOX_URL": settings.gearbox_url,
        "ANTHROPIC_API_KEY": key.get_secret_value(),
        "JUDGE_BUDGET_GBP": f"{cap_gbp:.4f}",  # Dyno stops judging past this
    }
    result = subprocess.run(
        [dyno_executable(), "run", str(suite)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    match = re.search(r"^run (\w+):.*?GBP ([0-9.]+)", result.stdout, re.MULTILINE)
    if result.returncode != 0 or not match:
        raise LoopError(f"dyno run failed: {(result.stderr or result.stdout)[-600:]}")
    return match.group(1), float(match.group(2))


def dyno_failures(
    run_id: str, cases: dict[str, str], settings: Settings
) -> tuple[list[FailureRow], dict]:
    """Failures of the `quality` scorer in a Dyno run, as failure rows (+ the report)."""
    from pitwall.promote import dyno_executable

    result = subprocess.run(
        [dyno_executable(), "report", run_id, "--format", "json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "DATABASE_URL": settings.dyno_database_url},
    )
    if result.returncode != 0:
        raise LoopError(f"dyno report failed: {result.stderr[-400:]}")
    report = json.loads(result.stdout)
    rows = [
        FailureRow(
            id=f["case_id"],
            input=cases[f["case_id"]],
            reason=f"Dyno quality judge: {f['explanation']}"[:500],
            source="dyno-sample",
        )
        for f in report["failures"]
        if f["scorer"] == "quality" and f["case_id"] in cases
    ]
    return rows, report


# --- import: Dyno cases + training queue --------------------------------------------------------


def write_failures(rows: list[FailureRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in rows), encoding="utf-8", newline="\n")


def import_cases(failures_file: Path, stamp: str) -> tuple[Path | None, str]:
    """`dyno import-failures` into data/production/cases/<stamp>.yaml, skipping prompts
    already in earlier imports. (None, message) when nothing new was found."""
    from pitwall.promote import dyno_executable

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    earlier = sorted(CASES_DIR.glob("*.yaml"))
    out = CASES_DIR / f"cases-{stamp}.yaml"
    cmd = [dyno_executable(), "import-failures", str(failures_file), "--out", str(out)]
    for path in earlier:
        cmd += ["--existing", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise LoopError(f"dyno import-failures failed: {(result.stderr or result.stdout)[-600:]}")
    message = result.stdout.strip()
    if message.startswith("0 new cases"):
        out.unlink(missing_ok=True)
        return None, message
    return out, message


def queue_for_labelling(rows: list[FailureRow], queue: Path = QUEUE) -> int:
    """Append prompts not already queued; returns how many were added."""
    queue.parent.mkdir(parents=True, exist_ok=True)
    known: set[str] = set()
    if queue.exists():
        known = {
            json.loads(line)["id"]
            for line in queue.read_text(encoding="utf-8").splitlines()
            if line
        }
    added = 0
    with queue.open("a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            pid = prompt_id(r.input)
            if pid in known:
                continue
            known.add(pid)
            f.write(
                json.dumps(
                    {
                        "id": pid,
                        "prompt": r.input,
                        "source": r.source,
                        "reason": r.reason,
                        "queued_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    }
                )
                + "\n"
            )
            added += 1
    return added


def dvc_add(path: Path = PRODUCTION_DIR) -> str:
    """Version data/production/ with DVC; returns the new hash from its .dvc file."""
    result = subprocess.run(
        ["dvc", "add", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise LoopError(f"dvc add failed: {(result.stderr or result.stdout)[-400:]}")
    meta = yaml.safe_load(Path(f"{path}.dvc").read_text(encoding="utf-8"))
    return str(meta["outs"][0]["md5"])
