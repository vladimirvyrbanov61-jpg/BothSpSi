"""Blind differential distinguisher dataset generator (Gohr-style).

Class 1 (y=1): (C0, C1) = (Enc_K^R(P0), Enc_K^R(P1)), P1 = P0 ⊕ Δ_P, random K.
Class 0 (y=0): independent uniform random 32-bit block pairs.

Features X = concat_pair_bits(C0, C1) ∈ {0,1}^64 — no plaintext, key, or decrypt data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from ciphers.common.encoding import PAIR_BITS, concat_pairs_batch
from ciphers.common.sampling import (
    apply_delta,
    random_blocks,
    sample_keys,
    sample_plaintexts,
)
from ciphers.registry import CipherName, get_cipher, max_rounds
from thesis.data.cache import (
    cache_path,
    config_fingerprint,
    load_blind_npz,
    save_blind_npz,
)

# Word-level difference: flip LSB of left 16-bit word (common thesis choice)
DEFAULT_INPUT_DELTA: tuple[int, int] = (0x0001, 0x0000)


def _validate_rounds(cipher: CipherName, rounds: int) -> None:
    cap = max_rounds(cipher)
    if not (1 <= rounds <= cap):
        raise ValueError(f"rounds must be in 1..{cap} for {cipher}, got {rounds}")


def generate_distinguisher_dataset(
    cipher_name: CipherName,
    rounds: int,
    n_samples: int,
    rng: np.random.Generator,
    *,
    input_delta: tuple[int, int] = DEFAULT_INPUT_DELTA,
) -> tuple[np.ndarray, np.ndarray]:
    """Build balanced (X, y) with X shape (N, 64) float32, y in {0, 1}."""
    if n_samples < 2 or n_samples % 2 != 0:
        raise ValueError("n_samples must be a positive even number for 50/50 balance")
    if len(input_delta) != 2 or any(
        isinstance(word, bool)
        or not isinstance(word, (int, np.integer))
        or not 0 <= int(word) <= 0xFFFF
        for word in input_delta
    ):
        raise ValueError("input_delta must contain exactly two 16-bit integer words")
    if not any(int(word) for word in input_delta):
        raise ValueError("input_delta must be nonzero")
    _validate_rounds(cipher_name, rounds)

    n_each = n_samples // 2
    cipher = get_cipher(cipher_name)

    keys = sample_keys(n_each, rng)
    p0 = sample_plaintexts(n_each, rng)
    p1 = apply_delta(p0, input_delta)

    # Cipher profiles support one independent key per block. Expanding the
    # complete key batch once is substantially faster than scalar encryption
    # and avoids retaining one cache entry per sample.
    real0 = cipher.encrypt(p0, keys, rounds=rounds)
    real1 = cipher.encrypt(p1, keys, rounds=rounds)
    cipher.clear_subkey_cache()

    rand0 = random_blocks(n_each, rng)
    rand1 = random_blocks(n_each, rng)

    X_real = concat_pairs_batch(real0, real1)
    X_rand = concat_pairs_batch(rand0, rand1)
    X = np.vstack([X_real, X_rand])
    y = np.array([1] * n_each + [0] * n_each, dtype=np.int8)

    perm = rng.permutation(len(y))
    X = X[perm]
    y = y[perm]

    assert X.shape == (n_samples, PAIR_BITS)
    assert set(np.unique(y)) <= {0, 1}
    assert int(y.sum()) == n_each
    return X, y


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    y = np.asarray(y)
    if y.ndim != 1 or y.size == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError("y must be a non-empty binary vector")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in (0, 1):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        if min(n_tr, n_va, n - n_tr - n_va) < 1:
            raise ValueError(
                "each class must contribute at least one train, validation, "
                "and test sample"
            )
        train_idx.extend(idx[:n_tr].tolist())
        val_idx.extend(idx[n_tr : n_tr + n_va].tolist())
        test_idx.extend(idx[n_tr + n_va :].tolist())
    return {
        "train": np.array(train_idx, dtype=np.int64),
        "val": np.array(val_idx, dtype=np.int64),
        "test": np.array(test_idx, dtype=np.int64),
    }


def generate_or_load(
    cipher_name: CipherName,
    rounds: int,
    n_samples: int,
    *,
    input_delta: tuple[int, int] = DEFAULT_INPUT_DELTA,
    seed: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    data_dir: Path | str = "thesis/data/cache",
    force_regen: bool = False,
) -> dict[str, Any]:
    """Generate or load cached blind dataset; returns X, y, splits, rounds, cache_path."""
    data_dir = Path(data_dir)
    tag = config_fingerprint(
        cipher=cipher_name,
        rounds=rounds,
        n_samples=n_samples,
        delta=input_delta,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    path = cache_path(data_dir, cipher_name, rounds, tag)

    if not force_regen and path.exists():
        loaded = load_blind_npz(path)
        y = loaded["y"]
        cached_rounds = int(loaded["rounds"][0])
        if cached_rounds != rounds:
            raise ValueError(
                f"cache round mismatch in {path}: expected {rounds}, found {cached_rounds}"
            )
        if len(y) != n_samples:
            raise ValueError(
                f"cache sample-count mismatch in {path}: "
                f"expected {n_samples}, found {len(y)}"
            )
        if int(y.sum()) != n_samples // 2:
            raise ValueError(f"cache is not class-balanced in {path}")
        return {
            **loaded,
            "rounds": cached_rounds,
            "splits": stratified_split_indices(
                y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
            ),
            "cache_path": path,
        }

    # Reuse the same underlying random pairs across round counts so round curves
    # isolate the effect of additional cipher rounds.
    rng = np.random.default_rng(seed)
    X, y = generate_distinguisher_dataset(
        cipher_name,
        rounds,
        n_samples,
        rng,
        input_delta=input_delta,
    )
    save_blind_npz(path, X, y, rounds)
    return {
        "X": X,
        "y": y,
        "rounds": rounds,
        "splits": stratified_split_indices(
            y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        ),
        "cache_path": path,
    }
