"""Ciphertext bit serialization for neural distinguishers (32/64 profiles).

Convention (matches Simon/Speck `*3264/encoding.py`):
  - Block = [left_word, right_word], each uint16.
  - Bits are MSB-first within each 16-bit word; left word occupies indices 0..15,
    right word 16..31.
  - Pair feature = bits(C0) || bits(C1) → length 64.
"""

from __future__ import annotations

import numpy as np

WORD_MASK = np.uint16(0xFFFF)
WORD_BITS = 16
BLOCK_BITS = 32
PAIR_BITS = 64


def block_to_bits(words: np.ndarray, n: int = WORD_BITS) -> np.ndarray:
    """One block (2,) uint16 → (32,) uint8 bits in {0, 1}."""
    if n != WORD_BITS:
        raise ValueError(f"32/64 profiles require n={WORD_BITS}, got {n}")
    w = np.asarray(words, dtype=np.uint16).reshape(2)
    bits = np.empty(BLOCK_BITS, dtype=np.uint8)
    for i, word in enumerate(w):
        wi = int(word) & int(WORD_MASK)
        for b in range(n):
            bits[i * n + (n - 1 - b)] = (wi >> b) & 1
    return bits


def blocks_to_bits(blocks: np.ndarray, n: int = WORD_BITS) -> np.ndarray:
    """(N, 2) uint16 → (N, 32) uint8 bits."""
    if n != WORD_BITS:
        raise ValueError(f"32/64 profiles require n={WORD_BITS}, got {n}")
    b = np.asarray(blocks, dtype=np.uint16)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"blocks must be (N, 2), got {b.shape}")
    out = np.empty((b.shape[0], BLOCK_BITS), dtype=np.uint8)
    for i in range(b.shape[0]):
        out[i] = block_to_bits(b[i], n)
    return out


def concat_pair_bits(
    c0: np.ndarray,
    c1: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Serialize one ciphertext pair to a length-64 float bit vector.

    Parameters
    ----------
    c0, c1 : array-like, shape (2,) uint16
        Left/right words of each 32-bit block.

    Returns
    -------
    ndarray, shape (64,), dtype float32 by default, values in {0.0, 1.0}.
    """
    b0 = block_to_bits(c0)
    b1 = block_to_bits(c1)
    return np.concatenate([b0, b1]).astype(dtype, copy=False)


def concat_pairs_batch(
    c0_blocks: np.ndarray,
    c1_blocks: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Vectorized concat_pair_bits for (N, 2) block arrays → (N, 64)."""
    bits0 = blocks_to_bits(c0_blocks)
    bits1 = blocks_to_bits(c1_blocks)
    return np.hstack([bits0, bits1]).astype(dtype, copy=False)


def reshape_for_cnn(x: np.ndarray, layout: str = "1x64") -> np.ndarray:
    """Optional layouts for Phase-2 CNN (not used in generator output).

    layout:
      - ``1x64``: (N, 1, 64)
      - ``4x16``: (N, 4, 16) — two words × two halves (ablation)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2 or x.shape[1] != PAIR_BITS:
        raise ValueError(f"expected (N, {PAIR_BITS}) features, got {x.shape}")
    if layout == "1x64":
        return x[:, np.newaxis, :]
    if layout == "4x16":
        return x.reshape(-1, 4, 16)
    raise ValueError(f"unknown layout {layout!r}")
