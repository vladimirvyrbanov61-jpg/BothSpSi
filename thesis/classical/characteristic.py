"""R-round beam-search estimate of maximum characteristic probability."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np

from thesis.classical.ddt_core import (
    Delta32,
    WORD_MASK,
    max_trail_probability,
    prune_top_k,
    row_from_conditional,
)
from thesis.classical.ddt_simon import (
    analytical_round_distribution_from_f,
    compute_f_ddt_exact,
    compute_simon_round_ddt,
)
from thesis.classical.ddt_speck import compute_speck_round_ddt

CipherName = Literal["simon", "speck"]
CLASSICAL_SCHEMA_VERSION = 3

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _validate_search_inputs(
    cipher: str,
    delta_in: Delta32,
    *,
    top_k: int,
    n_samples_row: int | None,
) -> None:
    if cipher not in {"simon", "speck"}:
        raise ValueError(f"unsupported cipher {cipher!r}")
    if (
        not isinstance(delta_in, tuple)
        or len(delta_in) != 2
        or any(
            isinstance(word, bool)
            or not isinstance(word, (int, np.integer))
            or not 0 <= int(word) <= WORD_MASK
            for word in delta_in
        )
    ):
        raise ValueError("delta_in must contain exactly two 16-bit integer words")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if n_samples_row is not None and n_samples_row < 1:
        raise ValueError("n_samples_row must be positive")


def _expand_transition_rows(
    cipher: CipherName,
    delta_in: Delta32,
    transition_cache: dict[Delta32, dict[Delta32, float]],
    *,
    n_samples_row: int,
    seed: int,
    f_ddt_simon: dict[int, dict[int, float]] | None,
) -> None:
    if delta_in in transition_cache:
        return
    if cipher == "simon" and f_ddt_simon is not None:
        transition_cache[delta_in] = analytical_round_distribution_from_f(
            delta_in,
            f_ddt_simon,
        )
    elif cipher == "simon":
        transition_cache[delta_in] = compute_simon_round_ddt(
            delta_in, n_samples=n_samples_row, seed=seed
        )
    else:
        transition_cache[delta_in] = compute_speck_round_ddt(
            delta_in, n_samples=n_samples_row, seed=seed
        )


def estimate_max_characteristic_probability(
    cipher: CipherName,
    rounds: int,
    delta_in: Delta32,
    *,
    n_samples_row: int | None = None,
    top_k: int = 32,
    seed: int = 0,
    use_simon_f_exact: bool = True,
) -> tuple[float, list[Delta32]]:
    """Estimate the best retained R-round differential trail.

    SIMON round transitions are exact and SPECK rows are sampled, but top-k
    pruning makes the multi-round search non-exhaustive. The returned value is
    a beam-search estimate, not a formal probability bound.
    """
    _validate_search_inputs(
        cipher,
        delta_in,
        top_k=top_k,
        n_samples_row=n_samples_row,
    )
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if rounds == 0:
        return 1.0, [delta_in]

    if n_samples_row is None:
        n_samples_row = 250_000 if cipher == "simon" else 1_000_000

    f_ddt = compute_f_ddt_exact() if cipher == "simon" and use_simon_f_exact else None
    transition_cache: dict[Delta32, dict[Delta32, float]] = {}

    states: dict[Delta32, float] = {delta_in: 1.0}
    parent_layers: list[dict[Delta32, Delta32]] = []

    for r in range(rounds):
        nxt: dict[Delta32, float] = {}
        parents: dict[Delta32, Delta32] = {}
        for d_in, p_in in states.items():
            _expand_transition_rows(
                cipher,
                d_in,
                transition_cache,
                n_samples_row=n_samples_row,
                seed=seed + r,
                f_ddt_simon=f_ddt,
            )
            row = row_from_conditional(transition_cache, d_in)
            for d_out, p_cond in row.items():
                cand = p_in * p_cond
                if cand > nxt.get(d_out, 0.0):
                    nxt[d_out] = cand
                    parents[d_out] = d_in
        if not nxt:
            return 0.0, [delta_in]
        states = prune_top_k(nxt, top_k)
        parent_layers.append({state: parents[state] for state in states})

    best = max(states, key=states.get)
    trail = [best]
    for parents in reversed(parent_layers):
        trail.append(parents[trail[-1]])
    trail.reverse()
    return states[best], trail


def estimate_max_characteristic_from_fixed_transition(
    delta_in: Delta32,
    transition: dict[Delta32, dict[Delta32, float]],
    rounds: int,
    *,
    top_k: int = 32,
) -> tuple[float, list[Delta32]]:
    """Estimate the best retained trail from a fixed sparse transition matrix."""
    return max_trail_probability(delta_in, transition, rounds, top_k=top_k)


# Compatibility aliases for callers using the pre-schema-3 API. New code
# should use the estimate_* names because top-k pruning is not a formal bound.
max_characteristic_probability = estimate_max_characteristic_probability
max_characteristic_from_fixed_transition = (
    estimate_max_characteristic_from_fixed_transition
)


def track_characteristic_over_rounds(
    cipher: CipherName,
    round_list: list[int],
    delta_in: Delta32,
    *,
    n_samples_row: int | None = None,
    top_k: int = 32,
    seed: int = 0,
) -> list[dict[str, Any]]:
    _validate_search_inputs(
        cipher,
        delta_in,
        top_k=top_k,
        n_samples_row=n_samples_row,
    )
    if len(set(round_list)) != len(round_list):
        raise ValueError("round counts must not contain duplicates")
    requested_rounds = sorted(set(round_list))
    if not requested_rounds:
        return []
    if requested_rounds[0] < 1:
        raise ValueError("round counts must be positive")
    if n_samples_row is None:
        n_samples_row = 250_000 if cipher == "simon" else 1_000_000

    f_ddt = compute_f_ddt_exact() if cipher == "simon" else None
    transition_cache: dict[Delta32, dict[Delta32, float]] = {}
    states: dict[Delta32, float] = {delta_in: 1.0}
    rows_by_round: dict[int, dict[str, Any]] = {}

    for round_index in range(1, requested_rounds[-1] + 1):
        nxt: dict[Delta32, float] = {}
        for current_delta, current_probability in states.items():
            _expand_transition_rows(
                cipher,
                current_delta,
                transition_cache,
                n_samples_row=n_samples_row,
                seed=seed + round_index - 1,
                f_ddt_simon=f_ddt,
            )
            for output_delta, conditional_probability in transition_cache[current_delta].items():
                candidate = current_probability * conditional_probability
                if candidate > nxt.get(output_delta, 0.0):
                    nxt[output_delta] = candidate

        states = prune_top_k(nxt, top_k) if nxt else {}
        if round_index in requested_rounds:
            rows_by_round[round_index] = {
                "schema_version": CLASSICAL_SCHEMA_VERSION,
                "cipher": cipher,
                "rounds": round_index,
                "delta_in": delta_in,
                "max_characteristic_prob": max(states.values()) if states else 0.0,
                "trail_len": round_index + 1,
                "method": (
                    "exact_f_ddt_beam_search"
                    if cipher == "simon"
                    else "monte_carlo_beam_search"
                ),
                "is_formal_bound": False,
                "n_samples_row": n_samples_row,
                "top_k": top_k,
                "seed": seed,
            }

    return [rows_by_round[rounds] for rounds in round_list]


def save_classical_bounds_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "schema_version",
        "cipher",
        "rounds",
        "max_characteristic_prob",
        "delta_left",
        "delta_right",
        "method",
        "is_formal_bound",
        "n_samples_row",
        "top_k",
        "seed",
        "repetitions",
        "probability_std",
        "probability_ci95_low",
        "probability_ci95_high",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                d = row["delta_in"]
                w.writerow(
                    {
                        "schema_version": row["schema_version"],
                        "cipher": row["cipher"],
                        "rounds": row["rounds"],
                        "max_characteristic_prob": row["max_characteristic_prob"],
                        "delta_left": d[0],
                        "delta_right": d[1],
                        "method": row["method"],
                        "is_formal_bound": str(
                            bool(row.get("is_formal_bound", False))
                        ).lower(),
                        "n_samples_row": row["n_samples_row"],
                        "top_k": row["top_k"],
                        "seed": row["seed"],
                        "repetitions": row.get("repetitions", 1),
                        "probability_std": row.get("probability_std", 0.0),
                        "probability_ci95_low": row.get(
                            "probability_ci95_low",
                            row["max_characteristic_prob"],
                        ),
                        "probability_ci95_high": row.get(
                            "probability_ci95_high",
                            row["max_characteristic_prob"],
                        ),
                    }
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_classical_bounds_csv(
    path: Path,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[int, float]:
    """Map rounds to best-retained characteristic-probability estimates."""
    out: dict[int, float] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        if expected:
            expected_method = (
                "exact_f_ddt_beam_search"
                if expected["cipher"] == "simon"
                else "monte_carlo_beam_search"
            )
            required = {
                "schema_version": str(CLASSICAL_SCHEMA_VERSION),
                "cipher": str(expected["cipher"]),
                "delta_left": str(int(expected["delta"][0])),
                "delta_right": str(int(expected["delta"][1])),
                "n_samples_row": str(int(expected["n_samples_row"])),
                "top_k": str(int(expected["top_k"])),
                "seed": str(int(expected["seed"])),
                "repetitions": str(int(expected["repetitions"])),
                "is_formal_bound": "false",
                "method": expected_method,
            }
            for row in rows:
                mismatches = {
                    key: (row.get(key), value)
                    for key, value in required.items()
                    if row.get(key) != value
                }
                if mismatches:
                    return {}
        for row in rows:
            rounds = int(row["rounds"])
            probability = float(row["max_characteristic_prob"])
            low = float(row.get("probability_ci95_low") or probability)
            high = float(row.get("probability_ci95_high") or probability)
            if (
                rounds < 1
                or rounds in out
                or not 0.0 <= low <= probability <= high <= 1.0
            ):
                return {}
            out[rounds] = probability
    return out


def load_classical_uncertainty_csv(path: Path) -> dict[int, dict[str, float]]:
    """Load mean and repeated-Monte-Carlo confidence limits by round."""
    if not path.exists():
        return {}
    output: dict[int, dict[str, float]] = {}
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mean = float(row["max_characteristic_prob"])
            rounds = int(row["rounds"])
            low = float(row.get("probability_ci95_low") or mean)
            high = float(row.get("probability_ci95_high") or mean)
            if (
                rounds < 1
                or rounds in output
                or not 0.0 <= low <= mean <= high <= 1.0
            ):
                raise ValueError(f"invalid classical uncertainty row in {path}")
            output[rounds] = {
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
            }
    return output
