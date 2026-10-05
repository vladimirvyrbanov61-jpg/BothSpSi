"""Round-count sweep: train CNN distinguisher and log test metrics per cipher."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from thesis.config.loader import config_path_for_profile, load_config, validate_config
from thesis.eval.manifest import (
    artifact_record,
    config_digest,
    file_digest,
    load_artifact_index,
    write_artifact_index,
)
from thesis.data.generator import DEFAULT_INPUT_DELTA, generate_or_load
from thesis.eval.manifest import artifact_inventory, build_manifest, utc_now, write_manifest
from thesis.eval.metrics import classification_metrics, metrics_row
from thesis.eval.plot_results import plot_all
from thesis.models.train import TrainConfig, _predict_scores, train_distinguisher

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = config_path_for_profile("full")

CSV_FIELDS = [
    "seed",
    "cipher",
    "rounds",
    "split",
    "n_samples",
    "input_delta_left",
    "input_delta_right",
    "accuracy",
    "auc_roc",
    "tpr",
    "tnr",
    "accuracy_advantage",
    "advantage_edge",
    "auc_advantage",
    "accuracy_null_p_value",
    "accuracy_null_log10_p_value",
    "accuracy_null_test",
    "youden_j",
]


def load_thesis_config(path: Path | None = None) -> dict[str, Any]:
    return load_config(path or DEFAULT_CONFIG)


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def _create_timestamped_run_dir(base_results_path: Path) -> Path:
    """Create a timestamped results directory for experimental hygiene."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_results_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_csv_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing)
            writer.writerow(row)
        temporary.replace(csv_path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_round_sweep_impl(
    config_path: Path | None = None,
    *,
    ciphers: list[str] | None = None,
    rounds_list: list[int] | None = None,
    n_samples: int | None = None,
    data_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    seed_override: int | None = None,
    force_regen: bool = False,
    fresh_csv: bool = False,
    use_timestamped_dir: bool = True,
    training_overrides: dict[str, Any] | None = None,
    _manifest_context: dict[str, Path] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Sweep rounds for each cipher; append test metrics to multi-seed CSV files."""
    cfg = load_thesis_config(config_path)
    base = _REPO_ROOT

    cipher_names = ciphers or cfg.get("ciphers") or [cfg.get("cipher", "simon")]
    if isinstance(cipher_names, str):
        cipher_names = [cipher_names]

    rounds_list = rounds_list or cfg.get("rounds", [3, 4, 5, 6, 7, 8, 9, 10])
    n = n_samples if n_samples is not None else int(cfg.get("n_samples_per_round", 100_000))
    seed = seed_override if seed_override is not None else int(cfg.get("seed", 1))
    delta = tuple(cfg.get("input_delta", list(DEFAULT_INPUT_DELTA)))
    data_path = _resolve_path(
        base, str(data_dir) if data_dir is not None else cfg.get("data_dir", "thesis/data/cache")
    )
    model_path = _resolve_path(
        base, str(model_dir) if model_dir is not None else cfg.get("model_dir", "models/thesis")
    )
    results_path = _resolve_path(
        base,
        str(results_dir) if results_dir is not None else cfg.get("results_dir", "results/thesis"),
    )
    log_path = None
    if log_dir is not None:
        log_path = _resolve_path(base, str(log_dir)) if not Path(log_dir).is_absolute() else Path(log_dir)
    training_values = {**cfg.get("training", {}), **(training_overrides or {})}
    resolved_cfg = deepcopy(cfg)
    resolved_cfg.update(
        {
            "ciphers": list(cipher_names),
            "rounds": list(rounds_list),
            "n_samples_per_round": n,
            "seed": seed,
            "data_dir": str(data_path),
            "model_dir": str(model_path),
            "results_dir": str(results_path),
            "training": training_values,
        }
    )
    validate_config(resolved_cfg, source="<resolved runtime configuration>")
    train_cfg = TrainConfig.from_dict(training_values)
    train_cfg.seed = seed
    train_ratio = float(cfg.get("train_ratio", 0.7))
    val_ratio = float(cfg.get("val_ratio", 0.15))

    # Create timestamped results directory for experimental hygiene
    if use_timestamped_dir:
        results_path = _create_timestamped_run_dir(results_path)
        manifest_path = results_path / "manifest.json"
        manifest = build_manifest(
            repo_root=base,
            run_type="round_sweep",
            config_path=Path(config_path or DEFAULT_CONFIG),
            config=cfg,
            parameters={
                "seeds": [seed],
                "ciphers": cipher_names,
                "rounds": rounds_list,
                "input_delta": list(delta),
                "n_samples_per_round": n,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "training": asdict(train_cfg),
                "force_regen": force_regen,
                "fresh_csv": fresh_csv,
            },
            paths={
                "results_dir": results_path,
                "data_dir": data_path,
                "model_dir": model_path,
                "log_dir": log_path,
            },
        )
        write_manifest(manifest_path, manifest)
        if _manifest_context is not None:
            _manifest_context.update(
                {"manifest_path": manifest_path, "results_path": results_path}
            )
        print(f"[sweep] Using timestamped results directory: {results_path}")
    else:
        manifest_path = None
        manifest = None

    all_rows: list[dict[str, Any]] = []

    for cipher in cipher_names:
        csv_path = results_path / f"{cipher}_multi_seed_raw.csv"
        round_csv_path = results_path / f"{cipher}_round_sweep.csv"
        if fresh_csv:
            if csv_path.exists():
                csv_path.unlink()
            if round_csv_path.exists():
                round_csv_path.unlink()

        for rounds in rounds_list:
            print(f"[sweep] seed={seed} {cipher} R={rounds} — loading data (n={n}) …")
            data = generate_or_load(
                cipher,
                rounds=rounds,
                n_samples=n,
                input_delta=delta,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                data_dir=data_path,
                force_regen=force_regen,
            )
            X, y = data["X"], data["y"]
            splits = data["splits"]

            print(f"[sweep] seed={seed} {cipher} R={rounds} — training …")
            result = train_distinguisher(
                X,
                y,
                splits,
                cipher=cipher,
                rounds=rounds,
                model_dir=model_path,
                cfg=train_cfg,
                log_dir=log_path,
                checkpoint_metadata={
                    "input_delta": list(delta),
                    "config_sha256": config_digest(resolved_cfg),
                    "cache_path": str(data["cache_path"]),
                    "cache_sha256": file_digest(Path(data["cache_path"])),
                    "run_id": results_path.name,
                },
            )
            model = result["model"]
            device = result["device"]

            test_idx = splits["test"]
            X_te, y_te = X[test_idx], y[test_idx]
            scores = _predict_scores(model, X_te, device)
            test_m = classification_metrics(
                y_te,
                scores,
                threshold=result["decision_threshold"],
            )
            row = metrics_row(
                cipher=cipher,
                rounds=rounds,
                split="test",
                n_samples=len(y_te),
                metrics=test_m,
                seed=seed,
                input_delta=delta,
            )
            _append_csv_row(csv_path, row)
            _append_csv_row(round_csv_path, row)
            all_rows.append(row)
            provenance_records = []
            for path, role in (
                (Path(data["cache_path"]), "dataset_cache"),
                (Path(result["checkpoint_path"]), "model_checkpoint"),
            ):
                record = artifact_record(path, role=role, repo_root=base)
                record.update({"cipher": cipher, "rounds": rounds, "seed": seed})
                provenance_records.append(record)
            tensorboard_dir = result.get("tensorboard_dir")
            if tensorboard_dir is not None:
                for event_path in sorted(
                    Path(tensorboard_dir).glob("events.out.tfevents.*")
                ):
                    record = artifact_record(
                        event_path,
                        role="tensorboard_event",
                        repo_root=base,
                    )
                    record.update({"cipher": cipher, "rounds": rounds, "seed": seed})
                    provenance_records.append(record)
            write_artifact_index(
                results_path / "external_artifacts.json",
                provenance_records,
            )
            print(
                f"[sweep] seed={seed} {cipher} R={rounds} — "
                f"acc={test_m['accuracy']:.4f} auc={test_m['auc_roc']:.4f} "
                f"edge={test_m['advantage_edge']:.4f} "
                f"log10_p_null={test_m['accuracy_null_log10_p_value']:.2f}"
            )

    if manifest is not None and manifest_path is not None:
        manifest["status"] = "postprocessing"
        manifest["training_completed_at"] = utc_now()
        manifest["progress"]["completed_seeds"] = [seed]
        manifest["external_artifacts"] = load_artifact_index(
            results_path / "external_artifacts.json"
        )
        import torch

        manifest["environment"]["torch"]["deterministic_algorithms_after_training"] = (
            torch.are_deterministic_algorithms_enabled()
        )
        manifest["artifacts"] = artifact_inventory(results_path)
        write_manifest(manifest_path, manifest)

    try:
        plot_all(results_path, cipher_names)
    except Exception as exc:
        if manifest is not None and manifest_path is not None:
            manifest["status"] = "failed"
            manifest["failure"] = {
                "stage": "neural_aggregation_and_plots",
                "error_type": type(exc).__name__,
                "reason": str(exc),
            }
            manifest["completed_at"] = utc_now()
            manifest["artifacts"] = artifact_inventory(results_path)
            write_manifest(manifest_path, manifest)
        raise

    if manifest is not None and manifest_path is not None:
        manifest["status"] = "completed"
        manifest["completed_at"] = utc_now()
        manifest["artifacts"] = artifact_inventory(results_path)
        write_manifest(manifest_path, manifest)

    return all_rows, results_path


def run_round_sweep(
    config_path: Path | None = None,
    *,
    ciphers: list[str] | None = None,
    rounds_list: list[int] | None = None,
    n_samples: int | None = None,
    data_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    log_dir: Path | str | None = None,
    seed_override: int | None = None,
    force_regen: bool = False,
    fresh_csv: bool = False,
    use_timestamped_dir: bool = True,
    training_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Run a sweep and persist terminal failure state when a manifest exists."""
    manifest_context: dict[str, Path] = {}
    try:
        return _run_round_sweep_impl(
            config_path,
            ciphers=ciphers,
            rounds_list=rounds_list,
            n_samples=n_samples,
            data_dir=data_dir,
            model_dir=model_dir,
            results_dir=results_dir,
            log_dir=log_dir,
            seed_override=seed_override,
            force_regen=force_regen,
            fresh_csv=fresh_csv,
            use_timestamped_dir=use_timestamped_dir,
            training_overrides=training_overrides,
            _manifest_context=manifest_context,
        )
    except BaseException as exc:
        manifest_path = manifest_context.get("manifest_path")
        run_path = manifest_context.get("results_path")
        if manifest_path is not None and run_path is not None and manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("status") not in {"failed", "interrupted"}:
                manifest["status"] = (
                    "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                )
                manifest["failure"] = {
                    "stage": "round_sweep",
                    "error_type": type(exc).__name__,
                    "reason": str(exc) or type(exc).__name__,
                }
                manifest["completed_at"] = utc_now()
                manifest["artifacts"] = artifact_inventory(run_path)
                write_manifest(manifest_path, manifest)
        raise


def resolve_config_path(
    config: Path | None = None,
    profile: str | None = None,
) -> Path:
    if config is not None:
        return config
    if profile in ("full", "quick", None):
        return config_path_for_profile(profile or "full")
    raise ValueError(f"unknown profile {profile!r}; use full or quick")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis neural distinguisher round sweep")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config (overrides --profile)",
    )
    parser.add_argument(
        "--profile",
        choices=["full", "quick"],
        default=None,
        help="Use thesis.yaml (full) or thesis_quick.yaml (quick)",
    )
    parser.add_argument(
        "--cipher",
        action="append",
        choices=["simon", "speck"],
        help="Restrict to cipher(s); default: all in config",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        help="Override round list from config",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override n_samples_per_round from config",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override dataset cache directory",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override results CSV directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory (e.g., runs/thesis)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Regenerate cached datasets",
    )
    parser.add_argument(
        "--fresh-csv",
        action="store_true",
        help="Delete existing per-cipher multi_seed_raw.csv before writing",
    )
    parser.add_argument(
        "--no-timestamped-dir",
        action="store_true",
        help="Disable timestamped results directory (use base results_dir directly)",
    )
    args = parser.parse_args()
    cfg_path = resolve_config_path(args.config, args.profile)
    run_round_sweep(
        cfg_path,
        ciphers=args.cipher,
        rounds_list=args.rounds,
        n_samples=args.n_samples,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        log_dir=args.log_dir,
        seed_override=args.seed,
        force_regen=args.force_regen,
        fresh_csv=args.fresh_csv,
        use_timestamped_dir=not args.no_timestamped_dir,
    )


if __name__ == "__main__":
    main()
