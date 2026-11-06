"""The baseline the fine-tune must beat: sentence embedding + logistic regression,
trained exactly as Gearbox's router v3 is, on Pitwall's splits.

Reimplemented here (not imported from Gearbox) so the repos stay independent; the M2
sanity check proves the two give the same predictions with the same settings.

1. Embed each prompt with bge-small-en-v1.5 (384 numbers, normalised; local, free).
2. Multinomial logistic regression gives P(S), P(M), P(L).
3. Pick the cheapest tier whose cumulative probability reaches a threshold t: S if
   P(S) >= t, else M if P(S) + P(M) >= t, else L. Higher t = fewer under-routes.

C, class weighting and t are chosen by 5-fold stratified CV on the training file only,
scored by routing cost (under-route = 2, over-route = 1).
"""

from __future__ import annotations

import hashlib
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pitwall.data import Pair, prompt_text
from pitwall.evaluation import Prediction
from pitwall.metrics import TIERS, UNDER_ROUTE_COST

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GRID: dict[str, list[Any]] = {
    "C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "class_weight": [None, "balanced"],
    "threshold": [0.4, 0.5, 0.6, 0.7],
}
FOLDS = 5
SEED = 20261029
CACHE_DIR = Path(".cache/embeddings")


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str = EMBEDDING_MODEL) -> None:
        self.name = name
        self._model: Any = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device="cpu")
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )


def embed_cached(texts: list[str], embedder: Embedder, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Embeddings cached on disk by content hash, so reruns take seconds."""
    key = hashlib.sha256((embedder.name + "\x00" + "\x00".join(texts)).encode()).hexdigest()
    path = cache_dir / f"emb-{key[:16]}.npy"
    if path.exists():
        return np.load(path)
    x = embedder.encode(texts)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


@dataclass
class Weights:
    coef: np.ndarray  # (3, dim), rows in S, M, L order
    intercept: np.ndarray  # (3,)
    threshold: float

    def proba(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.coef.T + self.intercept
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": "pitwall-lr/1",
            "embedding_model": EMBEDDING_MODEL,
            "classes": list(TIERS),
            "threshold": self.threshold,
            "intercept": [round(float(v), 6) for v in self.intercept],
            "coef": [[round(float(v), 6) for v in row] for row in self.coef],
        }


def decide(p: np.ndarray, threshold: float) -> int:
    """Index of the cheapest tier whose cumulative probability reaches the threshold."""
    return int(np.argmax(np.cumsum(p) >= threshold - 1e-12))


def routing_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    under = (y_pred < y_true).sum()
    over = (y_pred > y_true).sum()
    return float((UNDER_ROUTE_COST * under + over) / len(y_true))


def fit(x: np.ndarray, y: np.ndarray, c: float, class_weight: str | None) -> Weights:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=5000)
    model.fit(x, y)
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("training data must contain all three tiers")
    return Weights(model.coef_.copy(), model.intercept_.copy(), threshold=0.5)


def cross_validate(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Every grid setting scored by mean CV routing cost; best first."""
    from sklearn.model_selection import StratifiedKFold

    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED % 2**32).split(x, y))
    results = []
    for c, cw in itertools.product(GRID["C"], GRID["class_weight"]):
        fold_probs = [(val, fit(x[tr], y[tr], c, cw).proba(x[val])) for tr, val in folds]
        for t in GRID["threshold"]:
            pred = np.empty_like(y)
            for val, p in fold_probs:
                pred[val] = [decide(row, t) for row in p]
            results.append(
                {
                    "C": c,
                    "class_weight": cw,
                    "threshold": t,
                    "cost": round(routing_cost(y, pred), 4),
                    "accuracy": round(float((pred == y).mean()), 4),
                    "under_rate": round(float((pred < y).mean()), 4),
                }
            )
    # Lowest cost wins; ties go to the more regularised model (smaller C).
    return sorted(results, key=lambda r: (r["cost"], r["C"]))


@dataclass
class Trained:
    weights: Weights
    params: dict[str, Any]
    grid: list[dict[str, Any]]  # empty when the settings were given, not searched


def train(
    pairs: list[Pair],
    embedder: Embedder,
    settings: dict[str, Any] | None = None,
    cache_dir: Path = CACHE_DIR,
) -> Trained:
    """Fit on `pairs`. settings = {C, class_weight, threshold} skips the CV search."""
    x = embed_cached([prompt_text(p) for p in pairs], embedder, cache_dir)
    y = np.asarray([TIERS.index(p.label) for p in pairs])
    grid: list[dict[str, Any]] = []
    if settings is None:
        grid = cross_validate(x, y)
        settings = {k: grid[0][k] for k in ("C", "class_weight", "threshold")}
    weights = fit(x, y, settings["C"], settings["class_weight"])
    weights.threshold = float(settings["threshold"])
    params = {
        **settings,
        "embedding_model": embedder.name,
        "examples": len(pairs),
        "folds": FOLDS if grid else 0,
        "seed": SEED,
    }
    return Trained(weights, params, grid)


def predict(
    router: str,
    weights: Weights,
    pairs: list[Pair],
    embedder: Embedder,
    cache_dir: Path = CACHE_DIR,
) -> list[Prediction]:
    """Test-set predictions in Gearbox's format. latency_ms = LR only (embeddings cached);
    the real per-request latency is measured when the model is served (M4)."""
    x = embed_cached([prompt_text(p) for p in pairs], embedder, cache_dir)
    out = []
    for pair, row in zip(pairs, x, strict=True):
        start = time.perf_counter()
        p = weights.proba(row)
        tier = TIERS[decide(p, weights.threshold)]
        ms = (time.perf_counter() - start) * 1000
        probs = " ".join(f"P({t})={v:.2f}" for t, v in zip(TIERS, p, strict=True))
        reason = f"lr: {probs}, threshold {weights.threshold}"
        out.append(Prediction(pair.id, router, tier, reason, round(ms, 3), 0.0))
    return out
