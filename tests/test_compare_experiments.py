"""Tests for controlled input-difference experiment comparison."""

from __future__ import annotations

import pytest

from thesis.config.loader import config_path_for_profile, load_config
from thesis.eval.compare import require_neural_overlap, run_compare
from thesis.eval.compare_experiments import build_comparison_rows


def _aggregate(mean: float, low: float, high: float) -> dict[str, float]:
    return {
        "auc_roc_mean": mean,
        "auc_roc_ci95_low": low,
        "auc_roc_ci95_high": high,
        "advantage_edge_mean": 2.0 * (mean - 0.5),
        "advantage_edge_ci95_low": 2.0 * (low - 0.5),
        "advantage_edge_ci95_high": 2.0 * (high - 0.5),
    }


def test_build_comparison_rows_detects_interval_overlap() -> None:
    baseline = {"simon": {3: _aggregate(0.8, 0.75, 0.85)}}
    candidate = {"simon": {3: _aggregate(0.9, 0.86, 0.94)}}
    baseline_raw = {
        "simon": {
            3: {
                1: {"auc_roc": 0.78},
                2: {"auc_roc": 0.82},
                3: {"auc_roc": 0.80},
            }
        }
    }
    candidate_raw = {
        "simon": {
            3: {
                1: {"auc_roc": 0.88},
                2: {"auc_roc": 0.92},
                3: {"auc_roc": 0.90},
            }
        }
    }

    rows = build_comparison_rows(
        baseline,
        candidate,
        metrics=("auc_roc",),
        baseline_raw=baseline_raw,
        candidate_raw=candidate_raw,
    )

    assert len(rows) == 1
    assert rows[0]["mean_difference"] == pytest.approx(0.1)
    assert rows[0]["ci95_overlap"] is False
    assert rows[0]["paired_n"] == 3
    assert rows[0]["paired_difference_mean"] == pytest.approx(0.1)
    assert rows[0]["paired_difference_excludes_zero"] is True


def test_paired_test_does_not_treat_nearly_constant_differences_as_exact() -> None:
    baseline = {"simon": {3: _aggregate(0.8, 0.75, 0.85)}}
    candidate = {"simon": {3: _aggregate(0.800000002, 0.75, 0.85)}}
    baseline_raw = {
        "simon": {
            3: {
                1: {"auc_roc": 0.8},
                2: {"auc_roc": 0.8},
                3: {"auc_roc": 0.8},
            }
        }
    }
    candidate_raw = {
        "simon": {
            3: {
                1: {"auc_roc": 0.800000001},
                2: {"auc_roc": 0.800000002},
                3: {"auc_roc": 0.800000003},
            }
        }
    }

    row = build_comparison_rows(
        baseline,
        candidate,
        metrics=("auc_roc",),
        baseline_raw=baseline_raw,
        candidate_raw=candidate_raw,
    )[0]

    assert 0.0 < row["paired_p_value"] < 1.0


def test_delta_profile_only_changes_controlled_fields() -> None:
    baseline = load_config(config_path_for_profile("full"))
    candidate = load_config("thesis/config/thesis_delta_0040.yaml")
    changed = {
        key
        for key in set(baseline) | set(candidate)
        if baseline.get(key) != candidate.get(key)
    }

    assert changed == {"input_delta", "model_dir", "results_dir"}
    assert candidate["input_delta"] == [0x0040, 0]


def test_compare_rejects_duplicate_round_override(tmp_path) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        run_compare(
            config_path_for_profile("quick"),
            rounds_list=[3, 3],
            results_dir=tmp_path,
        )


def test_compare_requires_every_requested_neural_round(tmp_path) -> None:
    path = tmp_path / "simon_multi_seed_raw.csv"
    path.touch()
    with pytest.raises(ValueError, match=r"rounds \[4\]"):
        require_neural_overlap(
            "simon",
            [3, 4],
            {3: {"mean": 0.1, "ci95_low": 0.0, "ci95_high": 0.2}},
            path,
        )
