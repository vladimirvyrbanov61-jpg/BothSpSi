"""Shared DDT structures, normalization, and max-trail composition."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Hashable, Iterable, TypeVar

import numpy as np

Delta32 = tuple[int, int]
WORD_MASK = 0xFFFF

T = TypeVar("T", bound=Hashable)


def delta32_to_key(delta: Delta32) -> int:
    return ((int(delta[0]) & WORD_MASK) << 16) | (int(delta[1]) & WORD_MASK)


def key_to_delta32(key: int) -> Delta32:
    key &= 0xFFFFFFFF
    return (key >> 16, key & WORD_MASK)


def normalize_counts(counts: dict[T, int | float]) -> dict[T, float]:
    """Convert count histogram to probabilities in (0, 1]."""
    if not counts:
        return {}
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("counts must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("counts must contain positive total mass")
    out = {key: float(value) / total for key, value in counts.items() if value > 0}
    return out


def validate_probabilities(probs: dict[T, float], *, tol: float = 1e-9) -> None:
    for k, p in probs.items():
        if not (0.0 <= p <= 1.0 + tol):
            raise ValueError(f"invalid probability {p} for key {k}")
    s = sum(probs.values())
    if probs and abs(s - 1.0) > 1e-3:
        raise ValueError(f"probabilities sum to {s}, expected ~1")


def prune_top_k(states: dict[T, float], k: int) -> dict[T, float]:
    if k < 1:
        raise ValueError("k must be positive")
    if len(states) <= k:
        return states
    items = sorted(states.items(), key=lambda x: x[1], reverse=True)[:k]
    return dict(items)


def max_trail_probability(
    initial_delta: Delta32,
    transition: Dict[Delta32, Dict[Delta32, float]],
    rounds: int,
    *,
    top_k: int = 32,
) -> tuple[float, list[Delta32]]:
    """Best trail retained by top-k max-product beam search."""
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    for row in transition.values():
        validate_probabilities(row)
    if rounds == 0:
        return 1.0, [initial_delta]

    states: dict[Delta32, float] = {initial_delta: 1.0}
    parent_layers: list[dict[Delta32, Delta32]] = []

    for _ in range(rounds):
        nxt: dict[Delta32, float] = {}
        parents: dict[Delta32, Delta32] = {}
        for d_in, p_in in states.items():
            row = transition.get(d_in)
            if not row:
                continue
            for d_out, p_cond in row.items():
                cand = p_in * p_cond
                if cand > nxt.get(d_out, 0.0):
                    nxt[d_out] = cand
                    parents[d_out] = d_in
        if not nxt:
            return 0.0, [initial_delta]
        states = prune_top_k(nxt, top_k)
        parent_layers.append({state: parents[state] for state in states})

    best = max(states, key=states.get)
    trail = [best]
    for parents in reversed(parent_layers):
        trail.append(parents[trail[-1]])
    trail.reverse()
    return states[best], trail


def build_transition_from_pairs(
    pairs: Iterable[tuple[Delta32, Delta32]],
) -> Dict[Delta32, Dict[Delta32, float]]:
    """P(out | in) from observed (delta_in, delta_out) samples."""
    counts: dict[Delta32, dict[Delta32, int]] = defaultdict(lambda: defaultdict(int))
    for d_in, d_out in pairs:
        counts[d_in][d_out] += 1
    return {d_in: normalize_counts(row) for d_in, row in counts.items()}


def row_from_conditional(
    transition: Dict[Delta32, Dict[Delta32, float]],
    delta_in: Delta32,
) -> dict[Delta32, float]:
    return transition.get(delta_in, {})


def highest_output_probability(
    probs: dict[Delta32, float],
) -> tuple[float, Delta32 | None]:
    if not probs:
        return 0.0, None
    d_best = max(probs, key=probs.get)
    return float(probs[d_best]), d_best


def transition_row_monte_carlo(
    sample_round_pairs: Callable[[Delta32, int], Iterable[tuple[Delta32, Delta32]]],
    delta_in: Delta32,
    n_samples: int,
) -> dict[Delta32, float]:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    pairs = list(sample_round_pairs(delta_in, n_samples))
    counts: dict[Delta32, int] = defaultdict(int)
    for _, d_out in pairs:
        counts[d_out] += 1
    return normalize_counts(counts)
