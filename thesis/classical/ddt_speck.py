"""Speck32/64 empirical DDT for ARX rounds (Monte Carlo)."""

from __future__ import annotations

from typing import Dict, Iterator

import numpy as np

from thesis.classical.ddt_core import (
    WORD_MASK,
    Delta32,
    build_transition_from_pairs,
    normalize_counts,
    validate_probabilities,
)

from Speck.speck import _enc_round

N_BITS = 16
ALPHA = 7
BETA = 2


def speck_enc_round_pair(
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    *,
    n: int = N_BITS,
    alpha: int = ALPHA,
    beta: int = BETA,
) -> tuple[np.ndarray, np.ndarray]:
    """One Speck32/64 encryption round on batched (x, y) with shared subkey."""
    return _enc_round(x, y, k, n, alpha, beta)


def _sample_pairs(
    delta_in: Delta32,
    n_samples: int,
    rng: np.random.Generator,
) -> Iterator[tuple[Delta32, Delta32]]:
    """Monte Carlo: random (x,y,k), fixed input difference Δ_in."""
    dx, dy = int(delta_in[0]) & WORD_MASK, int(delta_in[1]) & WORD_MASK
    for _ in range(n_samples):
        x = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        y = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        k = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        x2 = np.uint16(x ^ dx)
        y2 = np.uint16(y ^ dy)
        xo, yo = speck_enc_round_pair(
            np.array([x], dtype=np.uint16),
            np.array([y], dtype=np.uint16),
            np.array([k], dtype=np.uint16),
        )
        xo2, yo2 = speck_enc_round_pair(
            np.array([x2], dtype=np.uint16),
            np.array([y2], dtype=np.uint16),
            np.array([k], dtype=np.uint16),
        )
        d_out: Delta32 = (int(xo[0] ^ xo2[0]), int(yo[0] ^ yo2[0]))
        yield delta_in, d_out


def compute_speck_round_ddt(
    delta_in: Delta32,
    n_samples: int = 1_000_000,
    *,
    seed: int = 0,
    batch_size: int = 262_144,
) -> dict[Delta32, float]:
    """Empirical 1-round output distribution P(Δ_out | Δ_in) for Speck32/64 ARX round."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    dx, dy = int(delta_in[0]) & WORD_MASK, int(delta_in[1]) & WORD_MASK
    packed_counts: dict[int, int] = {}
    for start in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - start)
        x = rng.integers(0, WORD_MASK + 1, size=size, dtype=np.uint16)
        y = rng.integers(0, WORD_MASK + 1, size=size, dtype=np.uint16)
        k = rng.integers(0, WORD_MASK + 1, size=size, dtype=np.uint16)
        x2 = x ^ np.uint16(dx)
        y2 = y ^ np.uint16(dy)

        xo, yo = speck_enc_round_pair(x, y, k)
        xo2, yo2 = speck_enc_round_pair(x2, y2, k)
        packed = (
            np.left_shift(np.bitwise_xor(xo, xo2).astype(np.uint32), 16)
            | np.bitwise_xor(yo, yo2).astype(np.uint32)
        )
        values, frequencies = np.unique(packed, return_counts=True)
        for value, frequency in zip(values.tolist(), frequencies.tolist()):
            packed_counts[value] = packed_counts.get(value, 0) + frequency

    counts = {
        ((packed >> 16) & WORD_MASK, packed & WORD_MASK): count
        for packed, count in packed_counts.items()
    }
    probs = normalize_counts(counts)
    validate_probabilities(probs)
    return probs


def speck_round_transition_monte_carlo(
    n_samples: int = 1_000_000,
    *,
    seed: int = 0,
) -> Dict[Delta32, Dict[Delta32, float]]:
    """Sparse round transition from random plaintexts, keys, and input differences."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    pairs: list[tuple[Delta32, Delta32]] = []
    for _ in range(n_samples):
        x = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        y = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        k = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        dx = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        dy = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        x2 = np.uint16(x ^ dx)
        y2 = np.uint16(y ^ dy)
        xo, yo = speck_enc_round_pair(
            np.array([x], dtype=np.uint16),
            np.array([y], dtype=np.uint16),
            np.array([k], dtype=np.uint16),
        )
        xo2, yo2 = speck_enc_round_pair(
            np.array([x2], dtype=np.uint16),
            np.array([y2], dtype=np.uint16),
            np.array([k], dtype=np.uint16),
        )
        d_in: Delta32 = (int(dx), int(dy))
        d_out: Delta32 = (int(xo[0] ^ xo2[0]), int(yo[0] ^ yo2[0]))
        pairs.append((d_in, d_out))
    return build_transition_from_pairs(pairs)


def highest_output_probability(
    probs: dict[Delta32, float],
) -> tuple[float, Delta32 | None]:
    if not probs:
        return 0.0, None
    d_best = max(probs, key=probs.get)
    return float(probs[d_best]), d_best
