"""Tests for signed distinguisher metrics and validation-only calibration."""

from __future__ import annotations

import numpy as np
import pytest

from thesis.eval.metrics import (
    CONDITIONAL_NULL_TEST,
    accuracy_null_log10_p_value,
    accuracy_null_p_value,
    classification_metrics,
    conditional_accuracy_null_log10_p_value,
    select_validation_threshold,
)


def test_random_accuracy_has_zero_signed_edge() -> None:
    metrics = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.9, 0.2, 0.8]),
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_advantage"] == 0.0
    assert metrics["advantage_edge"] == 0.0
    assert metrics["accuracy_null_p_value"] == 1.0


def test_validation_threshold_improves_shifted_scores() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    threshold = select_validation_threshold(labels, scores)
    assert 0.2 < threshold <= 0.3
    assert classification_metrics(labels, scores, threshold=threshold)["accuracy"] == 1.0


def test_null_p_value_detects_clear_edge() -> None:
    assert accuracy_null_p_value(50, 100) == 1.0
    assert accuracy_null_p_value(80, 100) < 1e-8


def test_extreme_null_p_value_retains_finite_log_magnitude() -> None:
    assert accuracy_null_p_value(14_991, 15_000) == 0.0
    log10_p = accuracy_null_log10_p_value(14_991, 15_000)
    assert np.isfinite(log10_p)
    assert log10_p < -4_000


def test_conditional_null_accounts_for_fixed_balanced_labels() -> None:
    labels = np.array([0, 0, 1, 1])
    all_positive = np.ones(4, dtype=np.int64)

    assert conditional_accuracy_null_log10_p_value(labels, all_positive) == 0.0
    metrics = classification_metrics(labels, all_positive.astype(np.float64))
    assert metrics["accuracy"] == 0.5
    assert metrics["accuracy_null_p_value"] == 1.0
    assert metrics["accuracy_null_test"] == CONDITIONAL_NULL_TEST


def test_classification_metrics_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned"):
        classification_metrics(np.array([0, 1]), np.array([0.2]))


def test_classification_metrics_rejects_fractional_labels() -> None:
    with pytest.raises(ValueError, match="binary"):
        classification_metrics(np.array([0.5, 1.0]), np.array([0.2, 0.8]))
