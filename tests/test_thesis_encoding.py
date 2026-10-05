"""Smoke tests for thesis encoding, sampling, and blind data generator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ciphers.common.encoding import (
    PAIR_BITS,
    block_to_bits,
    concat_pair_bits,
    concat_pairs_batch,
    reshape_for_cnn,
)
from ciphers.common.sampling import apply_delta, sample_keys
from ciphers.registry import get_cipher
from thesis.data.generator import (
    DEFAULT_INPUT_DELTA,
    generate_distinguisher_dataset,
    generate_or_load,
    stratified_split_indices,
)
from thesis.data.cache import load_blind_npz


def test_concat_pair_bits_shape_and_values():
    c0 = np.array([0x0001, 0x0000], dtype=np.uint16)
    c1 = np.array([0xFFFF, 0x0000], dtype=np.uint16)
    v = concat_pair_bits(c0, c1)
    assert v.shape == (PAIR_BITS,)
    assert v.dtype == np.float32
    assert set(np.unique(v)).issubset({0.0, 1.0})


def test_concat_pairs_batch_matches_scalar():
    c0 = np.array([[0x1234, 0x5678], [0, 0]], dtype=np.uint16)
    c1 = c0 ^ 0x00FF
    batch = concat_pairs_batch(c0, c1)
    for i in range(2):
        row = concat_pair_bits(c0[i], c1[i])
        np.testing.assert_array_equal(batch[i], row)


def test_fixed_profile_encoding_rejects_wrong_width_and_shape():
    with pytest.raises(ValueError, match="require n=16"):
        block_to_bits(np.array([1, 2]), n=8)
    with pytest.raises(ValueError, match="expected"):
        reshape_for_cnn(np.zeros((2, 63), dtype=np.float32))


def test_apply_delta_default():
    p = np.array([[0, 0], [0xFFFE, 0xFFFF]], dtype=np.uint16)
    d = apply_delta(p, DEFAULT_INPUT_DELTA)
    assert d[0, 0] == 1
    assert d[1, 0] == 0xFFFF


@pytest.mark.parametrize("cipher_name", ["simon", "speck"])
def test_generate_distinguisher_blind_small(cipher_name):
    rng = np.random.default_rng(0)
    X, y = generate_distinguisher_dataset(
        cipher_name, rounds=3, n_samples=200, rng=rng
    )
    assert X.shape == (200, 64)
    assert X.dtype == np.float32
    assert y.shape == (200,)
    assert int(y.sum()) == 100
    assert int((y == 0).sum()) == 100


def test_generate_distinguisher_rejects_zero_delta():
    with pytest.raises(ValueError, match="must be nonzero"):
        generate_distinguisher_dataset(
            "simon",
            rounds=3,
            n_samples=20,
            rng=np.random.default_rng(0),
            input_delta=(0, 0),
        )


def test_encrypt_rounds_api():
    simon = get_cipher("simon")
    speck = get_cipher("speck")
    pt = np.array([0x6565, 0x6877], dtype=np.uint16)
    key = sample_keys(1, np.random.default_rng(1))[0]
    c_full = simon.encrypt(pt, key)
    c_r = simon.encrypt(pt, key, rounds=32)
    np.testing.assert_array_equal(c_full, c_r)
    c3 = simon.encrypt(pt, key, rounds=3)
    assert not np.array_equal(c3, c_full)

    pt2 = np.array([0x6574, 0x694C], dtype=np.uint16)
    key2 = sample_keys(1, np.random.default_rng(2))[0]
    assert speck.encrypt(pt2, key2, rounds=22).shape == (1, 2)


def test_npz_cache_has_no_secrets(tmp_path):
    out = generate_or_load(
        "simon",
        rounds=3,
        n_samples=100,
        seed=1,
        data_dir=tmp_path,
        force_regen=True,
    )
    with np.load(out["cache_path"], allow_pickle=False) as z:
        assert set(z.files) <= {"X", "y", "rounds"}
    assert "X" in out and "y" in out


def test_cache_schema_rejects_unexpected_arrays(tmp_path):
    path = tmp_path / "invalid.npz"
    np.savez_compressed(
        path,
        X=np.zeros((2, 64), dtype=np.float32),
        y=np.array([0, 1], dtype=np.int8),
        rounds=np.array([3]),
        key=np.array([1]),
    )

    with pytest.raises(ValueError, match="cache leak"):
        load_blind_npz(path)


def test_cache_schema_rejects_non_binary_features(tmp_path):
    path = tmp_path / "invalid_features.npz"
    X = np.zeros((2, 64), dtype=np.float32)
    X[0, 0] = 0.5
    np.savez_compressed(
        path,
        X=X,
        y=np.array([0, 1], dtype=np.int8),
        rounds=np.array([3]),
    )

    with pytest.raises(ValueError, match="not binary"):
        load_blind_npz(path)


def test_generate_or_load_rejects_unbalanced_cache(tmp_path):
    out = generate_or_load(
        "simon",
        rounds=3,
        n_samples=100,
        seed=1,
        data_dir=tmp_path,
        force_regen=True,
    )
    with np.load(out["cache_path"], allow_pickle=False) as cached:
        X = cached["X"]
    np.savez_compressed(
        out["cache_path"],
        X=X,
        y=np.zeros(100, dtype=np.int8),
        rounds=np.array([3]),
    )

    with pytest.raises(ValueError, match="not class-balanced"):
        generate_or_load(
            "simon",
            rounds=3,
            n_samples=100,
            seed=1,
            data_dir=tmp_path,
        )


def test_stratified_split_rejects_non_binary_labels_and_negative_seed():
    with pytest.raises(ValueError, match="binary"):
        stratified_split_indices(np.array([0, 1, 2, 0, 1, 2]))
    with pytest.raises(ValueError, match="seed"):
        stratified_split_indices(np.array([0, 0, 0, 1, 1, 1]), seed=-1)
