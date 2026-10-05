"""Simon32/64 differential analysis: exact f-DDT and empirical round DDT."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterator

import numpy as np

from thesis.classical.ddt_core import (
    WORD_MASK,
    Delta32,
    build_transition_from_pairs,
    normalize_counts,
    validate_probabilities,
)

from Simon.simon import f_round

N_BITS = 16


def _mask16() -> int:
    return WORD_MASK


@lru_cache(maxsize=2)
def compute_f_ddt_exact(n: int = N_BITS) -> Dict[int, Dict[int, float]]:
    """Exact DDT of Simon round function f(x) = (S¹x & S⁸x) ⊕ S²x.

    For each input difference α, returns P(β) = Pr_x[f(x) ⊕ f(x⊕α) = β].
    Keys α, β are n-bit integers. Vectorized over x for each α (feasible at n=16).
    """
    size = 1 << n
    rows: Dict[int, Dict[int, float]] = {}
    x_all = np.arange(size, dtype=np.uint16)
    fx = f_round(x_all, n).astype(np.uint32)

    for alpha in range(size):
        beta = fx ^ f_round(x_all ^ np.uint16(alpha), n).astype(np.uint32)
        counts = np.bincount(beta, minlength=size)
        nz = np.nonzero(counts)[0]
        rows[alpha] = {int(b): float(counts[b]) / size for b in nz}

    return rows


def f_ddt_matrix_shape(n: int = N_BITS) -> tuple[int, int]:
    """Return (n_rows, n_cols) for the complete f-DDT as a dense matrix view."""
    size = 1 << n
    return size, size


def f_ddt_max_probability_per_input(f_ddt: Dict[int, Dict[int, float]]) -> float:
    """Maximum entry across all rows (largest one-round f differential probability)."""
    best = 0.0
    for row in f_ddt.values():
        if row:
            best = max(best, max(row.values()))
    return best


def simon_enc_round_pair(
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    *,
    n: int = N_BITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Simon encryption round (same subkey on both members of a pair)."""
    mask = np.uint16(_mask16())
    x_new = (y ^ f_round(x, n) ^ k) & mask
    y_new = x & mask
    return x_new, y_new


def _sample_pairs(
    delta_in: Delta32,
    n_samples: int,
    rng: np.random.Generator,
    *,
    random_key: bool,
) -> Iterator[tuple[Delta32, Delta32]]:
    dx, dy = int(delta_in[0]) & WORD_MASK, int(delta_in[1]) & WORD_MASK
    for _ in range(n_samples):
        x = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        y = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        k = (
            rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
            if random_key
            else np.uint16(0)
        )
        x2 = np.uint16(x ^ dx)
        y2 = np.uint16(y ^ dy)
        xo, yo = simon_enc_round_pair(
            np.array([x]), np.array([y]), np.array([k])
        )
        xo2, yo2 = simon_enc_round_pair(
            np.array([x2]), np.array([y2]), np.array([k])
        )
        d_out: Delta32 = (int(xo[0] ^ xo2[0]), int(yo[0] ^ yo2[0]))
        yield delta_in, d_out


def compute_simon_round_ddt(
    delta_in: Delta32,
    n_samples: int = 250_000,
    *,
    seed: int = 0,
    random_key: bool = True,
) -> dict[Delta32, float]:
    """Empirical 1-round output distribution P(Δ_out | Δ_in) for Simon32/64."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    counts: dict[Delta32, int] = {}
    for _, d_out in _sample_pairs(delta_in, n_samples, rng, random_key=random_key):
        counts[d_out] = counts.get(d_out, 0) + 1
    probs = normalize_counts(counts)
    validate_probabilities(probs)
    return probs


def simon_round_transition_monte_carlo(
    n_samples: int = 250_000,
    *,
    seed: int = 0,
    random_key: bool = True,
    uniform_input_delta: bool = False,
) -> Dict[Delta32, Dict[Delta32, float]]:
    """Estimate sparse Simon round transition matrix via Monte Carlo.

    If uniform_input_delta is False, samples only the fixed caller-provided
    workflow uses compute_simon_round_ddt; when True, also random Δ_in per sample.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    pairs: list[tuple[Delta32, Delta32]] = []

    for _ in range(n_samples):
        x = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        y = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        k = (
            rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
            if random_key
            else np.uint16(0)
        )
        if uniform_input_delta:
            dx = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
            dy = rng.integers(0, WORD_MASK + 1, dtype=np.uint16)
        else:
            dx = dy = np.uint16(0)
        x2 = np.uint16(x ^ dx)
        y2 = np.uint16(y ^ dy)
        xo, yo = simon_enc_round_pair(
            np.array([x]), np.array([y]), np.array([k])
        )
        xo2, yo2 = simon_enc_round_pair(
            np.array([x2]), np.array([y2]), np.array([k])
        )
        d_in: Delta32 = (int(dx), int(dy))
        d_out: Delta32 = (int(xo[0] ^ xo2[0]), int(yo[0] ^ yo2[0]))
        pairs.append((d_in, d_out))

    return build_transition_from_pairs(pairs)


def analytical_round_distribution_from_f(
    delta_in: Delta32,
    f_ddt: Dict[int, Dict[int, float]],
) -> dict[Delta32, float]:
    """Exact key-free SIMON round distribution from the f-DDT and Feistel swap.

    For fixed (dx, dy): dx_out = dy ⊕ β, dy_out = dx where β = f(x)⊕f(x⊕dx).
    Averages uniformly over β according to f-DDT[dx].
    """
    dx, dy = int(delta_in[0]) & WORD_MASK, int(delta_in[1]) & WORD_MASK
    out: dict[Delta32, float] = {}
    row = f_ddt.get(dx, {})
    for beta, p_beta in row.items():
        d_out: Delta32 = (dy ^ beta, dx)
        out[d_out] = out.get(d_out, 0.0) + p_beta
    validate_probabilities(out)
    return out


# Compatibility alias for the pre-schema-3 name.
analytical_round_bound_from_f = analytical_round_distribution_from_f
