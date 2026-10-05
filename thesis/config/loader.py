"""Load thesis YAML configs (requires PyYAML: pip install pyyaml)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

ProfileName = Literal["full", "quick"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = Path(__file__).resolve().parent

PROFILE_FILES: dict[ProfileName, str] = {
    "full": "thesis.yaml",
    "quick": "thesis_quick.yaml",
}

_CIPHER_MAX_ROUNDS = {"simon": 32, "speck": 22}
_TOP_LEVEL_KEYS = {
    "cipher",
    "ciphers",
    "input_delta",
    "n_samples_per_round",
    "train_ratio",
    "val_ratio",
    "rounds",
    "seed",
    "data_dir",
    "model_dir",
    "results_dir",
    "training",
    "classical",
}
_TRAINING_KEYS = {
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "device",
    "channels",
}
_CLASSICAL_KEYS = {
    "n_samples_simon",
    "n_samples_speck",
    "top_k_dp",
    "monte_carlo_repetitions",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _number(value: Any, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def validate_config(data: dict[str, Any], *, source: Path | str = "<config>") -> None:
    """Validate the complete thesis experiment configuration before execution."""
    label = str(source)
    unknown_top_level = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise ValueError(f"{label}: unknown configuration options {unknown_top_level}")

    ciphers = data.get("ciphers")
    if not isinstance(ciphers, list) or not ciphers:
        raise ValueError(f"{label}: ciphers must be a non-empty list")
    if len(set(ciphers)) != len(ciphers):
        raise ValueError(f"{label}: ciphers must not contain duplicates")
    unknown_ciphers = [cipher for cipher in ciphers if cipher not in _CIPHER_MAX_ROUNDS]
    if unknown_ciphers:
        raise ValueError(f"{label}: unsupported ciphers {unknown_ciphers}")

    primary_cipher = data.get("cipher")
    if primary_cipher not in _CIPHER_MAX_ROUNDS:
        raise ValueError(f"{label}: cipher must be 'simon' or 'speck'")

    delta = data.get("input_delta")
    if not isinstance(delta, (list, tuple)) or len(delta) != 2:
        raise ValueError(f"{label}: input_delta must contain exactly two 16-bit words")
    for index, word in enumerate(delta):
        value = _integer(word, f"{label}: input_delta[{index}]")
        if value > 0xFFFF:
            raise ValueError(f"{label}: input_delta[{index}] must be <= 0xffff")
    if not any(int(word) for word in delta):
        raise ValueError(f"{label}: input_delta must be nonzero")

    sample_count = _integer(
        data.get("n_samples_per_round"),
        f"{label}: n_samples_per_round",
        minimum=2,
    )
    if sample_count % 2:
        raise ValueError(f"{label}: n_samples_per_round must be even")

    train_ratio = _number(data.get("train_ratio"), f"{label}: train_ratio")
    val_ratio = _number(data.get("val_ratio"), f"{label}: val_ratio")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"{label}: train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"{label}: val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"{label}: train_ratio + val_ratio must be less than 1")
    samples_per_class = sample_count // 2
    train_per_class = int(samples_per_class * train_ratio)
    val_per_class = int(samples_per_class * val_ratio)
    test_per_class = samples_per_class - train_per_class - val_per_class
    if min(train_per_class, val_per_class, test_per_class) < 1:
        raise ValueError(
            f"{label}: sample count and split ratios must leave at least one "
            "sample per class in train, validation, and test"
        )

    rounds = data.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError(f"{label}: rounds must be a non-empty list")
    if len(set(rounds)) != len(rounds):
        raise ValueError(f"{label}: rounds must not contain duplicates")
    max_shared_rounds = min(_CIPHER_MAX_ROUNDS[cipher] for cipher in ciphers)
    for index, rounds_value in enumerate(rounds):
        value = _integer(rounds_value, f"{label}: rounds[{index}]", minimum=1)
        if value > max_shared_rounds:
            raise ValueError(
                f"{label}: rounds[{index}]={value} exceeds the supported "
                f"maximum {max_shared_rounds} for ciphers {ciphers}"
            )

    _integer(data.get("seed"), f"{label}: seed")
    for path_key in ("data_dir", "model_dir", "results_dir"):
        if not isinstance(data.get(path_key), str) or not data[path_key].strip():
            raise ValueError(f"{label}: {path_key} must be a non-empty path string")

    training = _mapping(data.get("training"), f"{label}: training")
    unknown_training = sorted(set(training) - _TRAINING_KEYS)
    if unknown_training:
        raise ValueError(f"{label}: unknown training options {unknown_training}")
    _integer(training.get("epochs"), f"{label}: training.epochs", minimum=1)
    _integer(training.get("batch_size"), f"{label}: training.batch_size", minimum=1)
    _number(training.get("lr"), f"{label}: training.lr", minimum=1e-15)
    _number(training.get("weight_decay"), f"{label}: training.weight_decay")
    _integer(training.get("patience"), f"{label}: training.patience", minimum=1)
    if training.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"{label}: training.device must be auto, cpu, or cuda")
    if "channels" in training:
        channels = training["channels"]
        if not isinstance(channels, list) or len(channels) not in (2, 3):
            raise ValueError(f"{label}: training.channels must contain two or three values")
        for index, channel in enumerate(channels):
            _integer(channel, f"{label}: training.channels[{index}]", minimum=1)

    classical = _mapping(data.get("classical"), f"{label}: classical")
    unknown_classical = sorted(set(classical) - _CLASSICAL_KEYS)
    if unknown_classical:
        raise ValueError(f"{label}: unknown classical options {unknown_classical}")
    _integer(
        classical.get("n_samples_simon"),
        f"{label}: classical.n_samples_simon",
        minimum=1,
    )
    _integer(
        classical.get("n_samples_speck"),
        f"{label}: classical.n_samples_speck",
        minimum=1,
    )
    _integer(classical.get("top_k_dp"), f"{label}: classical.top_k_dp", minimum=1)
    _integer(
        classical.get("monte_carlo_repetitions"),
        f"{label}: classical.monte_carlo_repetitions",
        minimum=1,
    )


def config_path_for_profile(profile: ProfileName = "full") -> Path:
    """Resolve config file path for a named profile."""
    if profile not in PROFILE_FILES:
        raise ValueError(f"unknown profile {profile!r}; use {list(PROFILE_FILES)}")
    return _CONFIG_DIR / PROFILE_FILES[profile]


def load_config(path: Path | str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for thesis config. Install with: pip install pyyaml"
        ) from e
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping, got {type(data)}")
    validate_config(data, source=path)
    return data


def load_profile(profile: ProfileName = "full") -> dict[str, Any]:
    """Load thesis.yaml (full) or thesis_quick.yaml (smoke)."""
    return load_config(config_path_for_profile(profile))
