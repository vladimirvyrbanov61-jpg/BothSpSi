"""Tests for classical DDT and characteristic composition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thesis.classical.ddt_core import (
    Delta32,
    build_transition_from_pairs,
    max_trail_probability,
    normalize_counts,
    prune_top_k,
    validate_probabilities,
)
from thesis.classical.ddt_simon import (
    analytical_round_distribution_from_f,
    compute_f_ddt_exact,
    compute_simon_round_ddt,
    f_ddt_matrix_shape,
    f_ddt_max_probability_per_input,
)
from thesis.classical.ddt_speck import compute_speck_round_ddt, highest_output_probability
from thesis.classical.characteristic import (
    estimate_max_characteristic_from_fixed_transition,
    load_classical_bounds_csv,
    save_classical_bounds_csv,
    track_characteristic_over_rounds,
)
from thesis.data.generator import DEFAULT_INPUT_DELTA


def test_normalize_probabilities_in_unit_interval():
    probs = normalize_counts({(0, 0): 3, (1, 0): 1})
    validate_probabilities(probs)
    assert all(0.0 < p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_f_ddt_shape_and_probabilities():
    f_ddt = compute_f_ddt_exact()
    n_rows, n_cols = f_ddt_matrix_shape()
    assert n_rows == 65536 and n_cols == 65536
    assert len(f_ddt) == 65536
    row0 = f_ddt[0]
    assert abs(sum(row0.values()) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in row0.values())
    assert f_ddt_max_probability_per_input(f_ddt) <= 1.0


def test_simon_analytical_round_from_f():
    f_ddt = compute_f_ddt_exact()
    row = analytical_round_distribution_from_f(DEFAULT_INPUT_DELTA, f_ddt)
    validate_probabilities(row)
    p_max, _ = highest_output_probability(row)
    assert 0.0 < p_max <= 1.0


@pytest.mark.parametrize("cipher", ["simon", "speck"])
def test_empirical_round_ddt_small(cipher):
    n = 8_000
    if cipher == "simon":
        probs = compute_simon_round_ddt(DEFAULT_INPUT_DELTA, n_samples=n, seed=1)
    else:
        probs = compute_speck_round_ddt(DEFAULT_INPUT_DELTA, n_samples=n, seed=2)
    validate_probabilities(probs)
    assert len(probs) >= 1
    assert all(0.0 < p <= 1.0 for p in probs.values())


def test_max_trail_composition_toy():
    transition = {
        (0, 1): {(0, 1): 0.5, (1, 0): 0.5},
        (1, 0): {(0, 0): 1.0},
        (0, 0): {(0, 1): 1.0},
    }
    p, trail = estimate_max_characteristic_from_fixed_transition(
        (0, 1), transition, 3, top_k=8
    )
    assert 0.0 <= p <= 1.0
    assert trail[0] == (0, 1)
    assert len(trail) == 4
    assert all(
        trail[index + 1] in transition[trail[index]]
        for index in range(len(trail) - 1)
    )


def test_max_trail_reconstructs_connected_parent_path():
    transition = {
        (0, 0): {(1, 0): 0.6, (2, 0): 0.4},
        (1, 0): {(3, 0): 0.5, (5, 0): 0.5},
        (2, 0): {(4, 0): 1.0},
    }
    probability, trail = max_trail_probability((0, 0), transition, 2, top_k=8)
    assert probability == pytest.approx(0.4)
    assert trail == [(0, 0), (2, 0), (4, 0)]


def test_build_transition_matrix_keys_are_delta_pairs():
    pairs = [((0, 1), (1, 0)), ((0, 1), (1, 0)), ((0, 1), (0, 1))]
    t = build_transition_from_pairs(pairs)
    assert (0, 1) in t
    validate_probabilities(t[(0, 1)])


def test_track_characteristic_reuses_one_round_traversal(monkeypatch):
    calls = []

    def fake_speck_row(delta_in, n_samples, *, seed):
        calls.append((delta_in, seed))
        return {delta_in: 0.5, (delta_in[0] ^ 1, delta_in[1]): 0.5}

    monkeypatch.setattr(
        "thesis.classical.characteristic.compute_speck_round_ddt",
        fake_speck_row,
    )
    rows = track_characteristic_over_rounds(
        "speck",
        [2, 3],
        DEFAULT_INPUT_DELTA,
        n_samples_row=10,
        top_k=1,
        seed=7,
    )

    assert [row["rounds"] for row in rows] == [2, 3]
    assert [row["max_characteristic_prob"] for row in rows] == [0.25, 0.125]
    assert all(row["is_formal_bound"] is False for row in rows)
    assert all("beam_search" in row["method"] for row in rows)
    assert len(calls) == 1


def test_classical_csv_declares_non_formal_estimate(tmp_path):
    rows = track_characteristic_over_rounds(
        "speck",
        [1],
        DEFAULT_INPUT_DELTA,
        n_samples_row=100,
        top_k=2,
        seed=3,
    )
    rows[0].update(
        {
            "repetitions": 1,
            "probability_std": 0.0,
            "probability_ci95_low": rows[0]["max_characteristic_prob"],
            "probability_ci95_high": rows[0]["max_characteristic_prob"],
        }
    )
    path = tmp_path / "estimate.csv"
    save_classical_bounds_csv(path, rows)

    loaded = load_classical_bounds_csv(
        path,
        expected={
            "cipher": "speck",
            "delta": DEFAULT_INPUT_DELTA,
            "n_samples_row": 100,
            "top_k": 2,
            "seed": 3,
            "repetitions": 1,
        },
    )
    assert 1 in loaded


def test_classical_csv_rejects_wrong_method_and_duplicate_round(tmp_path):
    rows = track_characteristic_over_rounds(
        "speck",
        [1],
        DEFAULT_INPUT_DELTA,
        n_samples_row=100,
        top_k=2,
        seed=3,
    )
    rows[0].update(
        {
            "repetitions": 1,
            "probability_std": 0.0,
            "probability_ci95_low": rows[0]["max_characteristic_prob"],
            "probability_ci95_high": rows[0]["max_characteristic_prob"],
        }
    )
    path = tmp_path / "estimate.csv"
    save_classical_bounds_csv(path, [rows[0], rows[0]])
    expected = {
        "cipher": "speck",
        "delta": DEFAULT_INPUT_DELTA,
        "n_samples_row": 100,
        "top_k": 2,
        "seed": 3,
        "repetitions": 1,
    }
    assert load_classical_bounds_csv(path, expected=expected) == {}

    rows[0]["method"] = "incorrect"
    save_classical_bounds_csv(path, rows)
    assert load_classical_bounds_csv(path, expected=expected) == {}


def test_classical_search_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="unsupported cipher"):
        track_characteristic_over_rounds(
            "invalid",
            [1],
            DEFAULT_INPUT_DELTA,
            n_samples_row=10,
        )
    with pytest.raises(ValueError, match="positive"):
        prune_top_k({DEFAULT_INPUT_DELTA: 1.0}, 0)
    with pytest.raises(ValueError, match="top_k"):
        max_trail_probability(DEFAULT_INPUT_DELTA, {}, 0, top_k=0)
    with pytest.raises(ValueError, match="non-negative"):
        normalize_counts({DEFAULT_INPUT_DELTA: 2, (0, 0): -1})
