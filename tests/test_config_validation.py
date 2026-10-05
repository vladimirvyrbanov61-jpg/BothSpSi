"""Validation tests for thesis experiment configurations."""

from __future__ import annotations

from copy import deepcopy
import json
import math

import pytest

from thesis.config.loader import config_path_for_profile, load_config, validate_config
from scripts.multi_seed_sweep import _prepare_new_results_dir, run_seed_sweep


def test_shipped_profiles_are_valid() -> None:
    load_config(config_path_for_profile("full"))
    load_config(config_path_for_profile("quick"))
    load_config("thesis/config/thesis_delta_0040.yaml")


def test_invalid_delta_fails_before_experiment() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["input_delta"] = [1]

    with pytest.raises(ValueError, match="exactly two"):
        validate_config(invalid)


@pytest.mark.parametrize(
    ("section", "key"),
    [(None, "n_sample_per_round"), ("classical", "top_k")],
)
def test_unknown_configuration_keys_are_rejected(section, key) -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    target = invalid if section is None else invalid[section]
    target[key] = 123

    with pytest.raises(ValueError, match="unknown"):
        validate_config(invalid)


def test_zero_delta_fails_before_experiment() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["input_delta"] = [0, 0]

    with pytest.raises(ValueError, match="must be nonzero"):
        validate_config(invalid)


def test_invalid_split_ratios_fail_before_experiment() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["train_ratio"] = 0.9
    invalid["val_ratio"] = 0.2

    with pytest.raises(ValueError, match="must be less than 1"):
        validate_config(invalid)


def test_odd_sample_count_is_rejected() -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["n_samples_per_round"] = 101

    with pytest.raises(ValueError, match="must be even"):
        validate_config(invalid)


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1]])
def test_seed_sweep_rejects_invalid_seed_lists(seeds) -> None:
    with pytest.raises(ValueError):
        run_seed_sweep(seeds, skip_tests=True)


def test_seed_sweep_rejects_nonempty_results_directory(tmp_path) -> None:
    results_dir = tmp_path / "existing_run"
    results_dir.mkdir()
    (results_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_new_results_dir(results_dir)


def test_seed_sweep_accepts_new_or_empty_results_directory(tmp_path) -> None:
    new_dir = tmp_path / "new_run"
    _prepare_new_results_dir(new_dir)
    assert new_dir.is_dir()

    _prepare_new_results_dir(new_dir)


@pytest.mark.parametrize(
    ("skip_tests", "expected_stage"),
    [(False, "test_gate"), (True, "round_sweep")],
)
def test_seed_sweep_records_keyboard_interrupt(
    tmp_path,
    monkeypatch,
    skip_tests,
    expected_stage,
) -> None:
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("scripts.multi_seed_sweep.subprocess.call", interrupt)
    results_dir = tmp_path / f"interrupted_{expected_stage}"

    with pytest.raises(KeyboardInterrupt):
        run_seed_sweep(
            [1],
            profile="quick",
            ciphers=["simon"],
            results_dir=results_dir,
            skip_tests=skip_tests,
        )

    manifest = json.loads(
        (results_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "interrupted"
    assert manifest["failure"]["stage"] == expected_stage


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_training_numbers_are_rejected(value) -> None:
    config = load_config(config_path_for_profile("quick"))
    invalid = deepcopy(config)
    invalid["training"]["lr"] = value

    with pytest.raises(ValueError, match="must be finite"):
        validate_config(invalid)
