"""Evaluation metrics for imbalanced screening problems."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Score above which a share `fpr` of legitimate applications would be flagged."""
    return float(np.quantile(score[y == 0], 1 - fpr))


def rates(y: np.ndarray, score: np.ndarray, threshold: float) -> tuple[float, float]:
    """(recall, false positive rate) when flagging every score above the threshold."""
    flagged = score > threshold
    return float(flagged[y == 1].mean()), float(flagged[y == 0].mean())


def recall_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float = 0.05) -> float:
    return rates(y, score, threshold_at_fpr(y, score, fpr))[0]


def fpr_ratio(y: np.ndarray, score: np.ndarray, group: np.ndarray, threshold: float) -> float:
    """Predictive equality as in the BAF paper: lower group FPR / higher group FPR (1 = equal)."""
    a = rates(y[group], score[group], threshold)[1]
    b = rates(y[~group], score[~group], threshold)[1]
    return float(min(a, b) / max(a, b)) if max(a, b) > 0 else 1.0


def summary(y, score, group=None, fpr: float = 0.05) -> dict:
    y, score = np.asarray(y), np.asarray(score)
    threshold = threshold_at_fpr(y, score, fpr)
    out = {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_5fpr": rates(y, score, threshold)[0],
    }
    if group is not None:
        out["fpr_ratio_age"] = fpr_ratio(y, score, np.asarray(group), threshold)
    return out


def bootstrap(y, scores: dict, reference: str, reps: int = 200, seed: int = 42) -> dict:
    """Paired bootstrap of recall at 5% FPR and PR-AUC, each model against `reference`."""
    y = np.asarray(y)
    scores = {name: np.asarray(s) for name, s in scores.items()}
    rng = np.random.default_rng(seed)
    draws = {name: {"recall_at_5fpr": [], "pr_auc": []} for name in scores}
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        for name, s in scores.items():
            draws[name]["recall_at_5fpr"].append(recall_at_fpr(y[idx], s[idx]))
            draws[name]["pr_auc"].append(average_precision_score(y[idx], s[idx]))
    out = {}
    for name, d in draws.items():
        out[name] = {}
        for metric, values in d.items():
            values = np.array(values)
            diff = values - np.array(draws[reference][metric])
            out[name][metric] = {
                "low": float(np.quantile(values, 0.025)),
                "high": float(np.quantile(values, 0.975)),
                "diff": float(diff.mean()),
                "diff_low": float(np.quantile(diff, 0.025)),
                "diff_high": float(np.quantile(diff, 0.975)),
            }
    return out
