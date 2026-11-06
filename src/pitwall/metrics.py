"""How good is a router's choice of tier? The same numbers for every model Pitwall
evaluates (baseline, references, fine-tunes), so they can be compared row by row.

- accuracy: picked the label (the cheapest tier that answered well).
- under-routing: picked a smaller tier than the label, so the answer is likely bad.
  Reported apart from over-routing (a bigger tier than needed: only wasted money),
  because the two mistakes don't cost the same.
- routing cost: Gearbox's training objective. Right = 0, over-route = 1, under-route = 2.
- pass rate: the share of prompts whose PICKED tier actually scored >= 4 with Gearbox's
  judge. Gearbox stored every tier's score for every prompt, so this is free, and it
  measures the thing that matters (a good answer), not agreement with a label.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

Tier = Literal["S", "M", "L"]
TIERS: tuple[Tier, ...] = ("S", "M", "L")
UNDER_ROUTE_COST = 2.0  # as in Gearbox's v3 training
PASS_SCORE = 4  # Gearbox's judge: a score of 4 or 5 is an answer good enough to send


def rank(tier: str) -> int:
    return TIERS.index(tier)  # type: ignore[arg-type]


def confusion(truth: Sequence[str], picks: Sequence[str]) -> list[list[int]]:
    """Rows = label, columns = pick, both in S, M, L order."""
    matrix = [[0, 0, 0] for _ in TIERS]
    for t, p in zip(truth, picks, strict=True):
        matrix[rank(t)][rank(p)] += 1
    return matrix


def _core(
    truth: Sequence[str], picks: Sequence[str], scores: Sequence[dict[str, int]] | None
) -> dict[str, Any]:
    n = len(truth)
    if n == 0:
        raise ValueError("no predictions to evaluate")
    under = sum(rank(p) < rank(t) for t, p in zip(truth, picks, strict=True))
    over = sum(rank(p) > rank(t) for t, p in zip(truth, picks, strict=True))
    result: dict[str, Any] = {
        "n": n,
        "accuracy": (n - under - over) / n,
        "under_rate": under / n,
        "over_rate": over / n,
        "routing_cost": (UNDER_ROUTE_COST * under + over) / n,
        "picks": {t: sum(p == t for p in picks) for t in TIERS},
    }
    if scores is not None:
        passed = sum(s.get(p, 0) >= PASS_SCORE for s, p in zip(scores, picks, strict=True))
        result["pass_rate"] = passed / n
    return result


def evaluate(
    truth: Sequence[str],
    picks: Sequence[str],
    scores: Sequence[dict[str, int]] | None = None,
    parts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """All metrics, plus the same per part (old / hard) when parts are given."""
    result = _core(truth, picks, scores)
    result["confusion"] = confusion(truth, picks)
    if parts is not None:
        result["by_part"] = {}
        for part in sorted(set(parts)):
            idx = [i for i, p in enumerate(parts) if p == part]
            result["by_part"][part] = _core(
                [truth[i] for i in idx],
                [picks[i] for i in idx],
                [scores[i] for i in idx] if scores is not None else None,
            )
    return result


def flat(metrics: dict[str, Any]) -> dict[str, float]:
    """The numeric metrics as MLflow keys: accuracy, hard.accuracy, picks.S, ..."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "confusion":
            continue
        if key == "by_part":
            for part, sub in value.items():
                out.update({f"{part}.{k}": v for k, v in flat(sub).items()})
        elif key == "picks":
            out.update({f"picks.{t}": float(v) for t, v in value.items()})
        else:
            out[key] = float(value)
    return out


def load_scores(path: Path) -> dict[str, dict[str, int]]:
    """Gearbox's labels.jsonl: each prompt's judge score (1-5) for every tier."""
    with path.open(encoding="utf-8") as f:
        return {row["id"]: row["scores"] for row in map(json.loads, f) if row}


def table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    """Plain-ASCII comparison table (cp1252 consoles)."""

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{100 * v:.1f}%"

    head = (
        f"{'run':<26} {'acc':>6} {'under':>6} {'over':>6} {'cost':>5} {'pass':>6} "
        f"{'S/M/L picks':>12} {'hard acc':>8}"
    )
    lines = [head, "-" * len(head)]
    for name, m in rows:
        picks = "/".join(str(m["picks"][t]) for t in TIERS)
        hard = m.get("by_part", {}).get("hard", {}).get("accuracy")
        lines.append(
            f"{name:<26} {pct(m['accuracy']):>6} {pct(m['under_rate']):>6} "
            f"{pct(m['over_rate']):>6} {m['routing_cost']:>5.2f} {pct(m.get('pass_rate')):>6} "
            f"{picks:>12} {pct(hard):>8}"
        )
    return "\n".join(lines)
