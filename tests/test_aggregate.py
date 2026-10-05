"""Tests for deterministic aggregation across experiment seeds."""

from __future__ import annotations

import warnings

import pytest

from thesis.eval.aggregate import (
    aggregate_rows,
    fisher_combined_log10_p_value,
    student_t_critical_95,
)


def test_aggregate_rows_reports_sample_uncertainty() -> None:
    base = {
        "cipher": "simon",
        "rounds": 3,
        "split": "test",
        "n_samples": 100,
        "auc_roc": 0.7,
        "tpr": 0.7,
        "tnr": 0.7,
        "accuracy_advantage": 0.1,
        "advantage_edge": 0.4,
        "auc_advantage": 0.4,
        "accuracy_null_p_value": 0.01,
        "youden_j": 0.4,
    }
    rows = [
        {**base, "seed": 1, "accuracy": 0.6},
        {**base, "seed": 2, "accuracy": 0.8},
        {**base, "seed": 99, "split": "val", "accuracy": 1.0},
    ]

    aggregate = aggregate_rows(rows)

    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["n_seeds"] == 2
    assert row["seeds"] == "1;2"
    assert row["n_samples_total"] == 200
    assert row["accuracy_mean"] == pytest.approx(0.7)
    assert row["accuracy_std"] == pytest.approx(2**0.5 / 10)
    assert row["accuracy_ci95_low"] < row["accuracy_mean"] < row["accuracy_ci95_high"]
    assert row["accuracy_ci95_low"] == 0.0
    assert row["accuracy_ci95_high"] == 1.0
    assert row["accuracy_null_p_value_fisher"] == pytest.approx(
        0.0010210340371976192
    )
    assert row["accuracy_null_log10_p_value_fisher"] == pytest.approx(
        -2.990959780027964
    )
    assert row["accuracy_null_test"] == "legacy_unlabeled_v1"
    assert "accuracy_null_p_value_mean" not in row


def test_ten_seed_interval_uses_student_t() -> None:
    assert student_t_critical_95(10) == pytest.approx(2.262)


def test_aggregate_rejects_non_finite_metrics() -> None:
    row = {
        "cipher": "simon",
        "rounds": 3,
        "split": "test",
        "seed": 1,
        "n_samples": 100,
        "accuracy": float("nan"),
        "auc_roc": 0.5,
        "tpr": 0.5,
        "tnr": 0.5,
        "accuracy_advantage": 0.0,
        "advantage_edge": 0.0,
        "auc_advantage": 0.0,
        "accuracy_null_p_value": 1.0,
        "youden_j": 0.0,
    }
    with pytest.raises(ValueError, match="finite"):
        aggregate_rows([row])


def test_aggregate_rejects_partial_seed_metric_rows() -> None:
    row = {
        "cipher": "simon",
        "rounds": 3,
        "split": "test",
        "seed": 1,
        "n_samples": 100,
        "accuracy": 0.5,
        "auc_roc": 0.5,
        "tpr": 0.5,
        "tnr": 0.5,
        "accuracy_advantage": 0.0,
        "advantage_edge": 0.0,
        "auc_advantage": 0.0,
        "accuracy_null_p_value": 1.0,
    }
    with pytest.raises(ValueError, match="Missing metrics.*youden_j"):
        aggregate_rows([row])


def test_fisher_combination_handles_underflowed_p_values_without_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        combined = fisher_combined_log10_p_value([-4_483.0, -4_255.0])
    assert combined < -8_730


def test_aggregate_rejects_mixed_input_differences() -> None:
    base = {
        "cipher": "simon",
        "rounds": 3,
        "split": "test",
        "n_samples": 100,
        "accuracy": 0.5,
        "auc_roc": 0.5,
        "tpr": 0.5,
        "tnr": 0.5,
        "accuracy_advantage": 0.0,
        "advantage_edge": 0.0,
        "auc_advantage": 0.0,
        "accuracy_null_p_value": 1.0,
        "youden_j": 0.0,
        "input_delta_right": 0,
    }
    rows = [
        {**base, "seed": 1, "input_delta_left": 1},
        {**base, "seed": 2, "input_delta_left": 0x40},
    ]
    with pytest.raises(ValueError, match="Inconsistent input differences"):
        aggregate_rows(rows)


def test_aggregate_rejects_mixed_null_test_methods() -> None:
    base = {
        "cipher": "simon",
        "rounds": 3,
        "split": "test",
        "n_samples": 100,
        "accuracy": 0.5,
        "auc_roc": 0.5,
        "tpr": 0.5,
        "tnr": 0.5,
        "accuracy_advantage": 0.0,
        "advantage_edge": 0.0,
        "auc_advantage": 0.0,
        "accuracy_null_p_value": 1.0,
        "youden_j": 0.0,
    }
    rows = [
        {**base, "seed": 1, "accuracy_null_test": "method_a"},
        {**base, "seed": 2, "accuracy_null_test": "method_b"},
    ]

    with pytest.raises(ValueError, match="Inconsistent null-significance"):
        aggregate_rows(rows)
