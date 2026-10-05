"""Classification metrics for neural distinguisher evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

_LOG_10 = float(np.log(10.0))
CONDITIONAL_NULL_TEST = "conditional_hypergeometric_fixed_margins_v1"


def _binary_labels(values: np.ndarray, *, name: str) -> np.ndarray:
    labels = np.asarray(values).ravel()
    if labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} must be a non-empty binary array")
    return labels.astype(np.int64)


def _probability_scores(values: np.ndarray, *, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64).ravel()
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError(f"{name} must be a non-empty finite array")
    if not ((0.0 <= scores) & (scores <= 1.0)).all():
        raise ValueError(f"{name} must be probabilities between 0 and 1")
    return scores


def select_validation_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Choose a threshold on validation data by maximizing Youden's J."""
    y_true = _binary_labels(y_true, name="validation labels")
    y_score = _probability_scores(y_score, name="validation scores")
    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("validation labels and scores must be non-empty and aligned")
    if len(np.unique(y_true)) < 2:
        return 0.5
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    candidates = np.where(finite)[0]
    best = candidates[int(np.argmax((tpr - fpr)[candidates]))]
    return float(np.clip(thresholds[best], 0.0, 1.0))


def accuracy_null_log10_p_value(correct: int, n_samples: int) -> float:
    """Log10 exact two-sided Binomial(n, 0.5) random-guessing p-value."""
    if n_samples < 1 or not 0 <= correct <= n_samples:
        raise ValueError("correct and n_samples must describe a non-empty test set")
    from scipy.special import logsumexp
    from scipy.stats import binom

    tail = min(correct, n_samples - correct)
    log_tail = float(
        logsumexp(
            binom.logpmf(
                np.arange(tail + 1, dtype=np.int64),
                n_samples,
                0.5,
            )
        )
    )
    log_two_sided = min(0.0, float(np.log(2.0)) + log_tail)
    return log_two_sided / _LOG_10


def accuracy_null_p_value(correct: int, n_samples: int) -> float:
    """Exact p-value, possibly zero when its magnitude is below float range."""
    return float(10.0 ** accuracy_null_log10_p_value(correct, n_samples))


def conditional_accuracy_null_log10_p_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Exact two-sided accuracy p-value conditional on fixed margins."""
    labels = _binary_labels(y_true, name="labels")
    predictions = _binary_labels(y_pred, name="predictions")
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must be aligned")

    n_samples = len(labels)
    n_positive = int(labels.sum())
    n_predicted_positive = int(predictions.sum())
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))

    from scipy.stats import hypergeom

    log_lower = float(
        hypergeom.logcdf(
            true_positive,
            n_samples,
            n_positive,
            n_predicted_positive,
        )
    )
    log_upper = float(
        hypergeom.logsf(
            true_positive - 1,
            n_samples,
            n_positive,
            n_predicted_positive,
        )
    )
    log_two_sided = min(0.0, float(np.log(2.0)) + min(log_lower, log_upper))
    return log_two_sided / _LOG_10


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute accuracy, AUC-ROC, TPR, TNR, and Gohr-style advantage."""
    y_true = _binary_labels(y_true, name="labels")
    y_score = _probability_scores(y_score, name="scores")
    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("labels and scores must be non-empty and aligned")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    y_pred = (y_score >= threshold).astype(np.int64)

    acc = float(np.mean(y_pred == y_true))

    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y_true, y_score))

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy_advantage = acc - 0.5
    advantage_edge = 2.0 * accuracy_advantage
    auc_advantage = 2.0 * (auc - 0.5)
    youden_j = tpr + tnr - 1.0
    null_log10_p = conditional_accuracy_null_log10_p_value(y_true, y_pred)

    return {
        "accuracy": acc,
        "auc_roc": auc,
        "tpr": float(tpr),
        "tnr": float(tnr),
        "accuracy_advantage": float(accuracy_advantage),
        "advantage_edge": float(advantage_edge),
        "auc_advantage": float(auc_advantage),
        "accuracy_null_p_value": float(10.0**null_log10_p),
        "accuracy_null_log10_p_value": null_log10_p,
        "accuracy_null_test": CONDITIONAL_NULL_TEST,
        "youden_j": float(youden_j),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def metrics_row(
    *,
    cipher: str,
    rounds: int,
    split: str,
    n_samples: int,
    metrics: dict[str, Any],
    seed: int | None = None,
    input_delta: tuple[int, int] | None = None,
) -> dict[str, Any]:
    row = {
        "cipher": cipher,
        "rounds": rounds,
        "split": split,
        "n_samples": n_samples,
        **{
            k: metrics[k]
            for k in (
                "accuracy",
                "auc_roc",
                "tpr",
                "tnr",
                "accuracy_advantage",
                "advantage_edge",
                "auc_advantage",
                "accuracy_null_p_value",
                "accuracy_null_log10_p_value",
                "accuracy_null_test",
                "youden_j",
            )
        },
    }
    if seed is not None:
        row["seed"] = seed
    if input_delta is not None:
        row["input_delta_left"] = int(input_delta[0])
        row["input_delta_right"] = int(input_delta[1])
    return row
