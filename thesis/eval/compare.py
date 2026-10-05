"""Publication plots: neural advantage vs classical beam-search estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from thesis.classical.characteristic import (
    load_classical_bounds_csv,
    load_classical_uncertainty_csv,
    save_classical_bounds_csv,
    track_characteristic_over_rounds,
)
from thesis.config.loader import config_path_for_profile, load_config, validate_config
from thesis.data.generator import DEFAULT_INPUT_DELTA
from thesis.eval.aggregate import aggregate_csv, summarize_values
from thesis.eval.manifest import (
    artifact_inventory,
    record_postprocessing,
    write_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = config_path_for_profile("full")


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_neural_sweep(csv_path: Path) -> dict[int, dict[str, float]]:
    """Map rounds to aggregate signed accuracy edge and its 95% confidence interval."""
    if not csv_path.exists():
        return {}
    rows, _ = aggregate_csv(csv_path)
    return {
        int(row["rounds"]): {
            "mean": float(row["advantage_edge_mean"]),
            "ci95_low": float(row["advantage_edge_ci95_low"]),
            "ci95_high": float(row["advantage_edge_ci95_high"]),
        }
        for row in rows
    }


def require_neural_overlap(
    cipher: str,
    rounds: list[int],
    neural: dict[int, dict[str, float]],
    neural_path: Path,
) -> None:
    if not neural_path.exists():
        raise FileNotFoundError(
            f"Neural sweep results missing for {cipher}: {neural_path}\n"
            "Run the round sweep first, for example:\n"
            "  py -3 -m thesis.eval.round_sweep --profile quick --fresh-csv\n"
            "  py -3 -m thesis.run_thesis --profile quick"
        )
    missing_rounds = sorted(set(rounds) - set(neural))
    if missing_rounds:
        raise ValueError(
            f"Missing test-split neural metrics for {cipher} at rounds "
            f"{missing_rounds} in {neural_path}."
        )


def ensure_classical_estimates(
    cipher: str,
    rounds: list[int],
    delta: tuple[int, int],
    results_dir: Path,
    *,
    n_samples_row: int | None,
    top_k: int,
    seed: int,
    repetitions: int,
    force: bool,
) -> dict[int, float]:
    bounds_path = results_dir / f"{cipher}_classical_bounds.csv"
    if n_samples_row is None:
        n_samples_row = 250_000 if cipher == "simon" else 1_000_000
    provenance = {
        "cipher": cipher,
        "delta": delta,
        "n_samples_row": n_samples_row,
        "top_k": top_k,
        "seed": seed,
        "repetitions": repetitions if cipher == "speck" else 1,
    }
    if not force and bounds_path.exists():
        loaded = load_classical_bounds_csv(bounds_path, expected=provenance)
        if all(rounds_value in loaded for rounds_value in rounds):
            return loaded

    effective_repetitions = repetitions if cipher == "speck" else 1
    repeated_rows = [
        track_characteristic_over_rounds(
            cipher,
            rounds,
            delta,
            n_samples_row=n_samples_row,
            top_k=top_k,
            seed=seed + repetition * 100_000,
        )
        for repetition in range(effective_repetitions)
    ]
    rows = []
    for index, rounds_value in enumerate(rounds):
        values = [
            float(repetition_rows[index]["max_characteristic_prob"])
            for repetition_rows in repeated_rows
        ]
        summary = summarize_values(values, bounds=(0.0, 1.0))
        row = dict(repeated_rows[0][index])
        row.update(
            {
                "rounds": rounds_value,
                "max_characteristic_prob": summary["mean"],
                "repetitions": effective_repetitions,
                "probability_std": summary["std"],
                "probability_ci95_low": summary["ci95_low"],
                "probability_ci95_high": summary["ci95_high"],
                "seed": seed,
            }
        )
        rows.append(row)
    save_classical_bounds_csv(bounds_path, rows)
    return load_classical_bounds_csv(bounds_path, expected=provenance)


def plot_classical_estimates(
    cipher: str,
    rounds: list[int],
    classical: dict[int, float],
    out_path: Path,
    uncertainty: dict[int, dict[str, float]] | None = None,
    *,
    delta: tuple[int, int],
    top_k: int,
) -> None:
    xs = sorted(set(rounds) & set(classical))
    probabilities = [max(classical[rounds_value], 1e-300) for rounds_value in xs]
    log2_probability = [np.log2(probability) for probability in probabilities]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    lower = [
        max((uncertainty or {}).get(r, {}).get("ci95_low", classical[r]), 1e-300)
        for r in xs
    ]
    upper = [
        max((uncertainty or {}).get(r, {}).get("ci95_high", classical[r]), 1e-300)
        for r in xs
    ]
    ax1.errorbar(
        xs,
        probabilities,
        yerr=np.vstack(
            [np.asarray(probabilities) - lower, np.asarray(upper) - probabilities]
        ),
        fmt="o-",
        capsize=4,
        linewidth=2,
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Round count R")
    ax1.set_ylabel("Best retained trail probability estimate")
    ax1.set_xticks(xs)
    ax1.set_title("Probability (log scale)")

    ax2.plot(xs, log2_probability, "s-", color="#d62728", linewidth=2)
    ax2.set_xlabel("Round count R")
    ax2.set_ylabel("log2(best retained trail estimate)")
    ax2.set_xticks(xs)
    ax2.set_title("Log2 probability")

    fig.suptitle(
        f"{cipher.upper()}32/64 - Classical characteristic beam-search estimate\n"
        f"Delta_P=(0x{delta[0]:04x}, 0x{delta[1]:04x}), beam width={top_k}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cipher_comparison(
    cipher: str,
    rounds: list[int],
    neural: dict[int, dict[str, float]],
    classical: dict[int, float],
    out_path: Path,
    classical_uncertainty: dict[int, dict[str, float]] | None = None,
    *,
    delta: tuple[int, int],
    top_k: int,
) -> None:
    xs = sorted(set(rounds) & set(neural) & set(classical))
    if not xs:
        raise ValueError(
            f"No overlapping rounds for {cipher} (config rounds={rounds}, "
            f"neural={sorted(neural)}, classical={sorted(classical)})."
        )

    advantage = [neural[rounds_value]["mean"] for rounds_value in xs]
    lower = [neural[rounds_value]["ci95_low"] for rounds_value in xs]
    upper = [neural[rounds_value]["ci95_high"] for rounds_value in xs]
    advantage_error = np.vstack(
        [np.asarray(advantage) - lower, np.asarray(upper) - np.asarray(advantage)]
    )
    probabilities = [max(classical[rounds_value], 1e-300) for rounds_value in xs]
    log2_probability = [np.log2(probability) for probability in probabilities]
    classical_lower = [
        np.log2(
            max(
                (classical_uncertainty or {})
                .get(rounds_value, {})
                .get("ci95_low", classical[rounds_value]),
                1e-300,
            )
        )
        for rounds_value in xs
    ]
    classical_upper = [
        np.log2(
            max(
                (classical_uncertainty or {})
                .get(rounds_value, {})
                .get("ci95_high", classical[rounds_value]),
                1e-300,
            )
        )
        for rounds_value in xs
    ]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    color_ai = "#1f77b4"
    color_classical = "#d62728"

    ax1.errorbar(
        xs,
        advantage,
        yerr=advantage_error,
        fmt="o-",
        capsize=4,
        color=color_ai,
        linewidth=2,
        markersize=7,
        label="AI signed accuracy edge 2(acc - 0.5) (95% CI)",
    )
    ax1.set_ylabel("Neural signed accuracy edge", color=color_ai)
    ax1.tick_params(axis="y", labelcolor=color_ai)
    ax1.set_xticks(xs)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.0, color=color_ai, linewidth=1, alpha=0.35)

    ax2.errorbar(
        xs,
        log2_probability,
        yerr=np.vstack(
            [
                np.asarray(log2_probability) - classical_lower,
                np.asarray(classical_upper) - log2_probability,
            ]
        ),
        fmt="s--",
        capsize=4,
        color=color_classical,
        linewidth=2,
        markersize=7,
        label="Classical log2(best retained trail estimate)",
    )
    ax2.set_xlabel("Round count R")
    ax2.set_ylabel("log2(best retained trail estimate)", color=color_classical)
    ax2.tick_params(axis="y", labelcolor=color_classical)
    ax2.set_xticks(xs)
    ax2.grid(True, alpha=0.3)
    fig.suptitle(
        f"{cipher.upper()}32/64 - Neural and classical round trends\n"
        f"Delta_P=(0x{delta[0]:04x}, 0x{delta[1]:04x}), beam width={top_k}"
    )
    ax1.legend(loc="best", framealpha=0.95)
    ax2.legend(loc="best", framealpha=0.95)
    fig.text(
        0.5,
        0.01,
        "Separate panels show different quantities; compare round trends, not magnitudes.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_compare(
    config_path: Path | None = None,
    *,
    ciphers: list[str] | None = None,
    rounds_list: list[int] | None = None,
    force_classical: bool = False,
    results_dir: Path | str | None = None,
) -> list[Path]:
    cfg = load_config(config_path or DEFAULT_CONFIG)
    results_path = _resolve_path(
        _REPO_ROOT,
        str(results_dir) if results_dir is not None else cfg.get("results_dir", "results/thesis"),
    )
    classical_cfg = cfg.get("classical", {})
    rounds = rounds_list or cfg.get("rounds", [3, 4, 5, 6, 7, 8, 9, 10])
    delta = tuple(cfg.get("input_delta", list(DEFAULT_INPUT_DELTA)))
    seed = int(cfg.get("seed", 1))
    top_k = int(classical_cfg.get("top_k_dp", 32))
    repetitions = int(classical_cfg.get("monte_carlo_repetitions", 1))
    sample_counts = {
        "simon": int(classical_cfg.get("n_samples_simon", 250_000)),
        "speck": int(classical_cfg.get("n_samples_speck", 1_000_000)),
    }
    cipher_names = ciphers or cfg.get("ciphers") or ["simon", "speck"]
    if isinstance(cipher_names, str):
        cipher_names = [cipher_names]
    resolved_cfg = dict(cfg)
    resolved_cfg["ciphers"] = list(cipher_names)
    resolved_cfg["rounds"] = list(rounds)
    validate_config(resolved_cfg, source="<resolved comparison configuration>")

    manifest_path = results_path / "manifest.json"
    manifest: dict | None = None
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        parameters = manifest.get("parameters", {})
        mismatches = {}
        if parameters.get("input_delta") != list(delta):
            mismatches["input_delta"] = (
                parameters.get("input_delta"),
                list(delta),
            )
        manifest_rounds = set(parameters.get("rounds", []))
        if not set(rounds).issubset(manifest_rounds):
            mismatches["rounds"] = (sorted(manifest_rounds), list(rounds))
        manifest_ciphers = set(parameters.get("ciphers", []))
        if not set(cipher_names).issubset(manifest_ciphers):
            mismatches["ciphers"] = (sorted(manifest_ciphers), list(cipher_names))
        if mismatches:
            raise ValueError(
                f"Results manifest does not match requested comparison: {mismatches}"
            )

    outputs: list[Path] = []
    for cipher in cipher_names:
        neural_path = results_path / f"{cipher}_multi_seed_raw.csv"
        neural = load_neural_sweep(neural_path)
        require_neural_overlap(cipher, rounds, neural, neural_path)
        classical = ensure_classical_estimates(
            cipher,
            rounds,
            delta,
            results_path,
            n_samples_row=sample_counts[cipher],
            top_k=top_k,
            seed=seed,
            repetitions=repetitions,
            force=force_classical,
        )
        classical_uncertainty = load_classical_uncertainty_csv(
            results_path / f"{cipher}_classical_bounds.csv"
        )
        classical_path = results_path / f"{cipher}_classical_ddt.png"
        plot_classical_estimates(
            cipher,
            rounds,
            classical,
            classical_path,
            uncertainty=classical_uncertainty,
            delta=delta,
            top_k=top_k,
        )
        outputs.append(classical_path)
        print(f"[compare] saved {classical_path}")
        out_path = results_path / f"{cipher}_vs_classical.png"
        plot_cipher_comparison(
            cipher,
            rounds,
            neural,
            classical,
            out_path,
            classical_uncertainty=classical_uncertainty,
            delta=delta,
            top_k=top_k,
        )
        outputs.append(out_path)
        print(f"[compare] saved {out_path}")

    if manifest is not None:
        manifest["artifacts"] = artifact_inventory(results_path)
        expected_ciphers = manifest.get("parameters", {}).get("ciphers", cipher_names)
        expected_outputs = [
            results_path / f"{expected_cipher}_{suffix}"
            for expected_cipher in expected_ciphers
            for suffix in (
                "classical_bounds.csv",
                "classical_ddt.png",
                "vs_classical.png",
            )
        ]
        if (
            manifest.get("run_type") == "multi_seed_sweep"
            and all(path.exists() for path in expected_outputs)
            and manifest.get("progress", {}).get("completed_seeds")
            == manifest.get("parameters", {}).get("seeds")
        ):
            manifest["status"] = "completed"
        record_postprocessing(
            manifest,
            "neural_vs_classical_plots",
            repo_root=_REPO_ROOT,
        )
        write_manifest(manifest_path, manifest)
    return outputs


def resolve_config_path(config: Path | None = None, profile: str | None = None) -> Path:
    if config is not None:
        return config
    if profile in ("full", "quick", None):
        return config_path_for_profile(profile or "full")
    raise ValueError(f"unknown profile {profile!r}; use full or quick")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AI metrics vs classical characteristic estimates"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", choices=["full", "quick"], default=None)
    parser.add_argument("--cipher", action="append", choices=["simon", "speck"])
    parser.add_argument("--rounds", type=int, nargs="+")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--force-classical", action="store_true")
    args = parser.parse_args()
    run_compare(
        resolve_config_path(args.config, args.profile),
        ciphers=args.cipher,
        rounds_list=args.rounds,
        force_classical=args.force_classical,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
