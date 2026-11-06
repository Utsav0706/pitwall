"""Pure pieces of fine-tuning: no torch here, so CI tests them in seconds.

- validation_split: stratified by label and seeded, so every run holds out the same
  prompts and the rare L examples are on both sides.
- make_example: (prompt ids, answer id). Only the answer token is learned, at the one
  position after "Tier:"; the prompt is context. A request longer than max_len is an
  error, not silently cut: the cut already happened, visibly, in the data build.
- choose: epochs and threshold picked on validation by routing cost, the same objective
  and threshold grid as the LR baseline, so the two are tuned the same way.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from pitwall.baseline import GRID, decide, routing_cost
from pitwall.data import Pair
from pitwall.metrics import TIERS


def validation_split(
    pairs: list[Pair], fraction: float = 0.2, seed: int = 20261029
) -> tuple[list[Pair], list[Pair]]:
    """(train, validation), stratified by label; same input and seed -> same split."""
    by_label: dict[str, list[Pair]] = defaultdict(list)
    for p in sorted(pairs, key=lambda p: p.id):
        by_label[p.label].append(p)
    rng = random.Random(seed)
    train: list[Pair] = []
    val: list[Pair] = []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_val = round(len(group) * fraction)
        if len(group) >= 2:
            n_val = max(1, n_val)  # every tier appears in validation, however rare
        val += group[:n_val]
        train += group[n_val:]
    return sorted(train, key=lambda p: p.id), sorted(val, key=lambda p: p.id)


def make_example(
    prompt_ids: list[int], answer_ids: list[int], max_len: int
) -> tuple[list[int], int]:
    """(prompt ids, answer id); the answer must be exactly one token."""
    if len(answer_ids) != 1:
        raise ValueError(f"answer is {len(answer_ids)} tokens; it must be one")
    if len(prompt_ids) + 1 > max_len:
        raise ValueError(f"example is {len(prompt_ids) + 1} tokens, over max_len {max_len}")
    return prompt_ids, answer_ids[0]


@dataclass
class EpochResult:
    epoch: int
    probs: np.ndarray  # (n_val, 3): P(S), P(M), P(L) renormalised over the three answers
    labels: np.ndarray  # (n_val,) as 0/1/2


def score_thresholds(result: EpochResult) -> list[dict[str, Any]]:
    rows = []
    for t in GRID["threshold"]:
        pred = np.asarray([decide(p, t) for p in result.probs])
        rows.append(
            {
                "epoch": result.epoch,
                "threshold": t,
                "cost": round(routing_cost(result.labels, pred), 4),
                "accuracy": round(float((pred == result.labels).mean()), 4),
                "under_rate": round(float((pred < result.labels).mean()), 4),
                "picks": {t_: int((pred == i).sum()) for i, t_ in enumerate(TIERS)},
            }
        )
    return rows


def choose(results: list[EpochResult]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(best row, all rows). Lowest validation routing cost; ties go to fewer epochs
    (less overfitting), then the lower threshold (cheaper)."""
    if not results:
        raise ValueError("no validation results to choose from")
    rows = [row for r in results for row in score_thresholds(r)]
    best = min(rows, key=lambda r: (r["cost"], r["epoch"], r["threshold"]))
    return best, rows


def renormalise(logits: np.ndarray) -> np.ndarray:
    """Softmax over the three answer logits only (the rest of the vocabulary ignored)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
