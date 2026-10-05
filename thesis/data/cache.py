"""NPZ cache for blind distinguisher tensors (X, y only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ciphers.registry import CipherName

DATASET_SCHEMA_VERSION = 3


def config_fingerprint(
    *,
    cipher: CipherName,
    rounds: int,
    n_samples: int,
    delta: tuple[int, int],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> str:
    spec = {
        "cipher": cipher,
        "rounds": rounds,
        "n": n_samples,
        "delta": list(delta),
        "seed": seed,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "blind": True,
        "feature": "concat_bits_64",
        "schema_version": DATASET_SCHEMA_VERSION,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


def cache_path(
    data_dir: Path,
    cipher: CipherName,
    rounds: int,
    tag: str,
) -> Path:
    return data_dir / f"{cipher}_r{rounds}_{tag}.npz"


def save_blind_npz(path: Path, X: np.ndarray, y: np.ndarray, rounds: int) -> None:
    """Persist only features and labels, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                X=X.astype(np.float32),
                y=y.astype(np.int8),
                rounds=np.array([rounds], dtype=np.int64),
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_blind_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        forbidden = {"plaintext", "key", "keys", "P", "K", "pt", "recovery"}
        for name in data.files:
            if name.lower() in forbidden or name.startswith("P_") or name.startswith("K_"):
                raise ValueError(f"cache leak: unexpected array {name!r} in {path}")
        expected = {"X", "y", "rounds"}
        if set(data.files) != expected:
            raise ValueError(
                f"cache schema mismatch in {path}: expected {sorted(expected)}, "
                f"found {sorted(data.files)}"
            )
        loaded = {key: data[key] for key in data.files}

    X = loaded["X"]
    y = loaded["y"]
    rounds = loaded["rounds"]
    if X.ndim != 2 or X.shape[1] != 64 or X.dtype != np.float32:
        raise ValueError(f"invalid feature tensor in cache {path}: {X.shape}, {X.dtype}")
    if not np.isfinite(X).all() or not np.isin(X, (0.0, 1.0)).all():
        raise ValueError(f"feature tensor is not binary in cache {path}")
    if y.ndim != 1 or len(y) != len(X) or not np.isin(y, (0, 1)).all():
        raise ValueError(f"invalid labels in cache {path}")
    if rounds.shape != (1,) or int(rounds[0]) < 1:
        raise ValueError(f"invalid round metadata in cache {path}")
    return loaded
