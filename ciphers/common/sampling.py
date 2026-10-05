"""Uniform random sampling for 32/64 cipher profiles (no fault injection)."""

from __future__ import annotations

import numpy as np

WORD_MASK = np.uint16(0xFFFF)


def sample_keys(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random 64-bit keys, shape (n, 4) uint16, big-endian word order for schedule."""
    return rng.integers(0, int(WORD_MASK) + 1, size=(n, 4), dtype=np.uint16)


def sample_plaintexts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random 32-bit plaintext blocks, shape (n, 2) uint16 [left, right]."""
    return rng.integers(0, int(WORD_MASK) + 1, size=(n, 2), dtype=np.uint16)


def apply_delta(
    plaintexts: np.ndarray,
    delta: tuple[int, int],
) -> np.ndarray:
    """Word-wise XOR: P1 = P0 ⊕ Δ."""
    d = np.array(delta, dtype=np.uint16)
    out = np.asarray(plaintexts, dtype=np.uint16).copy()
    out ^= d
    return out


def random_blocks(n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent uniform 32-bit blocks, shape (n, 2) uint16."""
    return sample_plaintexts(n, rng)
