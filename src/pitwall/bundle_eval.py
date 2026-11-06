"""Evaluate a fine-tuned bundle where it will run: the quantised GGUF, in Ollama, on CPU.

1. Quantisation check: score the bundle's own validation prompts (the same stratified
   split training used) and compare with the numbers training recorded in fp16 on the
   GPU. A big gap means the Q4 export broke the model; then no test number is trusted.
2. Test: route the 119 test prompts with the bundle's threshold; score them exactly as
   the LR and reference rows are scored (pitwall.evaluation).
3. Latency: wall time of every request after one warm-up, as Gearbox would wait for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pitwall import evaluation
from pitwall.baseline import decide
from pitwall.data import Pair, load_pairs, sha256_file
from pitwall.finetune import bundle as bundle_mod
from pitwall.finetune import core
from pitwall.metrics import TIERS
from pitwall.ollama import FALLBACK, OllamaRouter
from pitwall.textformat import render_pair


@dataclass
class Scored:
    predictions: list[evaluation.Prediction]
    probs: np.ndarray  # (n, 3)
    no_answer: int
    latencies: list[float]


def route(
    router: OllamaRouter, name: str, pairs: list[Pair], threshold: float, progress: Any = None
) -> Scored:
    preds, rows, latencies, missing = [], [], [], 0
    for i, pair in enumerate(pairs, 1):
        answer = router.answer(render_pair(pair)[0])
        p = np.asarray(answer.probs)
        if answer.no_answer:
            missing += 1
            tier, reason = FALLBACK, "ft: no S/M/L in the model's top tokens, fallback M"
        else:
            tier = TIERS[decide(p, threshold)]
            probs = " ".join(f"P({t})={v:.2f}" for t, v in zip(TIERS, p, strict=True))
            reason = f"ft: {probs}, threshold {threshold}"
        preds.append(evaluation.Prediction(pair.id, name, tier, reason, answer.latency_ms, 0.0))
        rows.append(p)
        latencies.append(answer.latency_ms)
        if progress:
            progress(i, len(pairs))
    return Scored(preds, np.asarray(rows), missing, latencies)


def latency_stats(latencies: list[float]) -> dict[str, float]:
    a = np.asarray(latencies)
    return {
        "latency_p50_ms": round(float(np.percentile(a, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(a, 95)), 1),
        "latency_max_ms": round(float(a.max()), 1),
    }


def validation_check(
    router: OllamaRouter, manifest: bundle_mod.Manifest, train_file: Path, progress: Any = None
) -> dict[str, Any]:
    """The GGUF's numbers on the training run's validation prompts vs the recorded ones."""
    if sha256_file(train_file) != manifest.train_sha256:
        raise ValueError(f"{train_file} is not the file this bundle was trained on (sha256)")
    _, val = core.validation_split(load_pairs(train_file), seed=manifest.training["seed"])
    if len(val) != manifest.validation_examples:
        raise ValueError(
            f"validation has {len(val)} prompts; training used {manifest.validation_examples}"
        )
    threshold = float(manifest.chosen["threshold"])
    scored = route(router, "validation", val, threshold, progress)
    labels = np.asarray([TIERS.index(p.label) for p in val])
    gguf = core.score_thresholds(core.EpochResult(manifest.chosen["epoch"], scored.probs, labels))
    gguf_row = next(r for r in gguf if r["threshold"] == threshold)
    return {
        "threshold": threshold,
        "trained": {k: manifest.chosen[k] for k in ("cost", "accuracy", "under_rate", "picks")},
        "gguf": {k: gguf_row[k] for k in ("cost", "accuracy", "under_rate", "picks")},
        "no_answer": scored.no_answer,
    }


def manifest_params(manifest: bundle_mod.Manifest) -> dict[str, Any]:
    """Bundle settings as flat MLflow params."""
    return {
        "base_model": manifest.base_model,
        "base_revision": manifest.base_revision[:12],
        "train_file": Path(manifest.train_file).stem,
        "examples": manifest.examples,
        "epochs": manifest.chosen["epoch"],
        "threshold": manifest.chosen["threshold"],
        **{f"lora_{k}": v for k, v in manifest.lora.items() if k != "target_modules"},
        **{k: manifest.training[k] for k in ("lr", "batch", "max_len", "seed")},
        "quant": manifest.gguf.quant if manifest.gguf else "none",
        "text_format": manifest.text_format,
    }


def manifest_tags(manifest: bundle_mod.Manifest) -> dict[str, str]:
    return {
        "bundle": manifest.name,
        "bundle_git_commit": manifest.git_commit,
        "bundle_data_version": manifest.data_version,
        "gguf_sha256": manifest.gguf.sha256 if manifest.gguf else "",
        "trained_on": manifest.training.get("device", ""),
    }


def load_checked(directory: Path) -> bundle_mod.Manifest:
    problems = bundle_mod.check(directory)
    if problems:
        raise ValueError("bundle check failed: " + "; ".join(problems))
    return bundle_mod.load_manifest(directory)
