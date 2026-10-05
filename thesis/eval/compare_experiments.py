"""Compare two completed multi-seed experiments round by round."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from thesis.eval.aggregate import aggregate_csv, read_metric_rows, summarize_values

COMPARISON_FIELDS = [
    "cipher",
    "rounds",
    "metric",
    "baseline_mean",
    "baseline_ci95_low",
    "baseline_ci95_high",
    "candidate_mean",
    "candidate_ci95_low",
    "candidate_ci95_high",
    "mean_difference",
    "ci95_overlap",
    "paired_n",
    "paired_difference_mean",
    "paired_difference_ci95_low",
    "paired_difference_ci95_high",
    "paired_difference_excludes_zero",
    "paired_p_value",
    "paired_p_value_holm",
    "paired_significant_holm",
]


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_delta: list[int],
    expected_seeds: list[int],
    expected_rounds: list[int],
) -> None:
    parameters = manifest.get("parameters", {})
    actual_delta = parameters.get("input_delta")
    if actual_delta != expected_delta:
        raise ValueError(f"Expected input_delta={expected_delta}, found {actual_delta}")
    if parameters.get("seeds") != expected_seeds:
        raise ValueError(
            f"Expected seeds={expected_seeds}, found {parameters.get('seeds')}"
        )
    if parameters.get("rounds") != expected_rounds:
        raise ValueError(
            f"Expected rounds={expected_rounds}, found {parameters.get('rounds')}"
        )
    if manifest.get("status") != "completed":
        raise ValueError(f"Experiment status must be completed, found {manifest.get('status')}")


def _controlled_manifest_view(manifest: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(manifest.get("config", {}).get("resolved", {}))
    for key in ("input_delta", "model_dir", "results_dir"):
        config.pop(key, None)
    parameters = deepcopy(manifest.get("parameters", {}))
    for key in ("input_delta", "force_regen", "fresh_csv_first_seed"):
        parameters.pop(key, None)
    environment = manifest.get("environment", {})
    source = manifest.get("source", {})
    return {
        "config": config,
        "parameters": parameters,
        "dependencies": environment.get("dependencies"),
        "python": environment.get("python"),
        "source_sha256": source.get("execution_sha256") or source.get("sha256"),
        "source_scope": (
            "execution"
            if source.get("execution_sha256") is not None
            else "full_maintained_tree_legacy"
        ),
    }


def _validate_controlled_pair(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> None:
    baseline = _controlled_manifest_view(baseline_manifest)
    candidate = _controlled_manifest_view(candidate_manifest)
    for key in ("config", "parameters", "dependencies", "python"):
        if baseline[key] != candidate[key]:
            raise ValueError(
                f"Experiments differ in controlled field {key}: "
                f"{baseline[key]!r} != {candidate[key]!r}"
            )
    if (
        baseline["source_sha256"] is not None
        and candidate["source_sha256"] is not None
        and baseline["source_sha256"] != candidate["source_sha256"]
    ):
        raise ValueError(
            "Experiments were produced from different maintained execution sources"
        )


def _load_aggregates(
    run_dir: Path,
    ciphers: list[str],
    expected_seeds: int,
    expected_rounds: list[int],
) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for cipher in ciphers:
        rows, _ = aggregate_csv(run_dir / f"{cipher}_multi_seed_raw.csv")
        by_round = {int(row["rounds"]): row for row in rows}
        if sorted(by_round) != expected_rounds:
            raise ValueError(
                f"{cipher} in {run_dir} has rounds {sorted(by_round)}, "
                f"expected {expected_rounds}"
            )
        for row in rows:
            if int(row["n_seeds"]) != expected_seeds:
                raise ValueError(
                    f"{cipher} round {row['rounds']} has {row['n_seeds']} seeds, "
                    f"expected {expected_seeds}"
                )
        output[cipher] = by_round
    return output


def _load_raw_metrics(
    run_dir: Path,
    ciphers: list[str],
    expected_seeds: list[int],
    expected_rounds: list[int],
) -> dict[str, dict[int, dict[int, dict[str, Any]]]]:
    output: dict[str, dict[int, dict[int, dict[str, Any]]]] = {}
    expected_seed_set = set(expected_seeds)
    for cipher in ciphers:
        by_round: dict[int, dict[int, dict[str, Any]]] = {}
        for row in read_metric_rows(run_dir / f"{cipher}_multi_seed_raw.csv"):
            if row.get("split", "test") != "test":
                continue
            rounds = int(row["rounds"])
            seed = int(row["seed"])
            if rounds not in expected_rounds:
                continue
            seed_rows = by_round.setdefault(rounds, {})
            if seed in seed_rows:
                raise ValueError(
                    f"Duplicate seed {seed} for {cipher} round {rounds} in {run_dir}"
                )
            seed_rows[seed] = row

        if sorted(by_round) != expected_rounds:
            raise ValueError(
                f"{cipher} raw results in {run_dir} have rounds {sorted(by_round)}, "
                f"expected {expected_rounds}"
            )
        for rounds, seed_rows in by_round.items():
            if set(seed_rows) != expected_seed_set:
                raise ValueError(
                    f"{cipher} round {rounds} in {run_dir} has seeds "
                    f"{sorted(seed_rows)}, expected {expected_seeds}"
                )
        output[cipher] = by_round
    return output


def build_comparison_rows(
    baseline: dict[str, dict[int, dict[str, Any]]],
    candidate: dict[str, dict[int, dict[str, Any]]],
    *,
    metrics: tuple[str, ...] = ("auc_roc", "advantage_edge"),
    baseline_raw: dict[str, dict[int, dict[int, dict[str, Any]]]] | None = None,
    candidate_raw: dict[str, dict[int, dict[int, dict[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cipher in sorted(baseline):
        for rounds in sorted(baseline[cipher]):
            for metric in metrics:
                base = baseline[cipher][rounds]
                new = candidate[cipher][rounds]
                base_low = float(base[f"{metric}_ci95_low"])
                base_high = float(base[f"{metric}_ci95_high"])
                new_low = float(new[f"{metric}_ci95_low"])
                new_high = float(new[f"{metric}_ci95_high"])
                base_mean = float(base[f"{metric}_mean"])
                new_mean = float(new[f"{metric}_mean"])
                comparison: dict[str, Any] = {
                    "cipher": cipher,
                    "rounds": rounds,
                    "metric": metric,
                    "baseline_mean": base_mean,
                    "baseline_ci95_low": base_low,
                    "baseline_ci95_high": base_high,
                    "candidate_mean": new_mean,
                    "candidate_ci95_low": new_low,
                    "candidate_ci95_high": new_high,
                    "mean_difference": new_mean - base_mean,
                    "ci95_overlap": max(base_low, new_low) <= min(base_high, new_high),
                    "paired_n": "",
                    "paired_difference_mean": "",
                    "paired_difference_ci95_low": "",
                    "paired_difference_ci95_high": "",
                    "paired_difference_excludes_zero": "",
                    "paired_p_value": "",
                    "paired_p_value_holm": "",
                    "paired_significant_holm": "",
                }
                if baseline_raw is not None and candidate_raw is not None:
                    base_seed_rows = baseline_raw[cipher][rounds]
                    candidate_seed_rows = candidate_raw[cipher][rounds]
                    if set(base_seed_rows) != set(candidate_seed_rows):
                        raise ValueError(
                            f"Paired seed mismatch for {cipher} round {rounds}"
                        )
                    differences = [
                        float(candidate_seed_rows[seed][metric])
                        - float(base_seed_rows[seed][metric])
                        for seed in sorted(base_seed_rows)
                    ]
                    paired = summarize_values(differences)
                    from scipy.stats import ttest_1samp

                    difference_array = np.asarray(differences, dtype=np.float64)
                    scale = max(1.0, float(np.max(np.abs(difference_array))))
                    constant_tolerance = 8.0 * np.finfo(np.float64).eps * scale
                    if float(np.ptp(difference_array)) <= constant_tolerance:
                        p_value = 0.0 if differences[0] != 0.0 else 1.0
                    else:
                        p_value = float(
                            ttest_1samp(difference_array, popmean=0.0).pvalue
                        )
                    comparison.update(
                        {
                            "paired_n": len(differences),
                            "paired_difference_mean": paired["mean"],
                            "paired_difference_ci95_low": paired["ci95_low"],
                            "paired_difference_ci95_high": paired["ci95_high"],
                            "paired_difference_excludes_zero": (
                                paired["ci95_low"] > 0.0 or paired["ci95_high"] < 0.0
                            ),
                            "paired_p_value": p_value,
                        }
                    )
                rows.append(comparison)
    tested = [
        (index, float(row["paired_p_value"]))
        for index, row in enumerate(rows)
        if row["paired_p_value"] != ""
    ]
    ordered = sorted(tested, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * p_value)
        running = max(running, adjusted)
        rows[index]["paired_p_value_holm"] = running
        rows[index]["paired_significant_holm"] = running < 0.05
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _plot_metric(
    cipher: str,
    metric: str,
    rows: list[dict[str, Any]],
    output_path: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    selected = [
        row for row in rows if row["cipher"] == cipher and row["metric"] == metric
    ]
    rounds = [int(row["rounds"]) for row in selected]
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for prefix, label, color, offset in (
        ("baseline", baseline_label, "#1f77b4", -0.06),
        ("candidate", candidate_label, "#ff7f0e", 0.06),
    ):
        means = np.asarray([float(row[f"{prefix}_mean"]) for row in selected])
        lower = np.asarray([float(row[f"{prefix}_ci95_low"]) for row in selected])
        upper = np.asarray([float(row[f"{prefix}_ci95_high"]) for row in selected])
        ax.errorbar(
            np.asarray(rounds) + offset,
            means,
            yerr=np.vstack([means - lower, upper - means]),
            fmt="o-",
            capsize=4,
            linewidth=2,
            color=color,
            label=label,
        )

    ax.set_xticks(rounds)
    ax.set_xlabel("Round count R")
    ax.set_ylabel(metric)
    ax.set_title(f"{cipher.upper()}32/64 - Input-difference sensitivity ({metric})")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compare_experiments(
    baseline_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> list[Path]:
    ciphers = ["simon", "speck"]
    seeds = list(range(1, 11))
    rounds = list(range(3, 11))
    baseline_manifest = _load_manifest(baseline_dir)
    candidate_manifest = _load_manifest(candidate_dir)
    _validate_manifest(
        baseline_manifest,
        expected_delta=[1, 0],
        expected_seeds=seeds,
        expected_rounds=rounds,
    )
    _validate_controlled_pair(baseline_manifest, candidate_manifest)
    _validate_manifest(
        candidate_manifest,
        expected_delta=[0x0040, 0],
        expected_seeds=seeds,
        expected_rounds=rounds,
    )
    baseline = _load_aggregates(baseline_dir, ciphers, len(seeds), rounds)
    candidate = _load_aggregates(candidate_dir, ciphers, len(seeds), rounds)
    baseline_raw = _load_raw_metrics(baseline_dir, ciphers, seeds, rounds)
    candidate_raw = _load_raw_metrics(candidate_dir, ciphers, seeds, rounds)
    rows = build_comparison_rows(
        baseline,
        candidate,
        baseline_raw=baseline_raw,
        candidate_raw=candidate_raw,
    )

    csv_path = output_dir / "input_delta_comparison.csv"
    _write_csv(csv_path, rows)
    outputs = [csv_path]
    for cipher in ciphers:
        for metric in ("auc_roc", "advantage_edge"):
            plot_path = output_dir / f"{cipher}_{metric}_input_delta_comparison.png"
            _plot_metric(
                cipher,
                metric,
                rows,
                plot_path,
                "Delta=(0x0001, 0x0000)",
                "Delta=(0x0040, 0x0000)",
            )
            outputs.append(plot_path)
    comparison_manifest = output_dir / "comparison_manifest.json"
    _write_json(
        comparison_manifest,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_results_dir": str(baseline_dir.resolve()),
            "candidate_results_dir": str(candidate_dir.resolve()),
            "metrics": ["auc_roc", "advantage_edge"],
            "multiple_testing_correction": "Holm",
            "baseline_manifest": {
                "config_sha256": baseline_manifest.get("config", {}).get("sha256"),
                "source_sha256": baseline_manifest.get("source", {}).get("sha256"),
                "execution_source_sha256": baseline_manifest.get("source", {}).get(
                    "execution_sha256"
                ),
                "input_delta": baseline_manifest.get("parameters", {}).get("input_delta"),
            },
            "candidate_manifest": {
                "config_sha256": candidate_manifest.get("config", {}).get("sha256"),
                "source_sha256": candidate_manifest.get("source", {}).get("sha256"),
                "execution_source_sha256": candidate_manifest.get("source", {}).get(
                    "execution_sha256"
                ),
                "input_delta": candidate_manifest.get("parameters", {}).get("input_delta"),
            },
            "controlled_manifest_view": _controlled_manifest_view(baseline_manifest),
            "outputs": [str(path.resolve()) for path in outputs],
        },
    )
    outputs.append(comparison_manifest)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two input-difference experiments")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.candidate_dir / "baseline_comparison"
    outputs = compare_experiments(args.baseline_dir, args.candidate_dir, output_dir)
    for path in outputs:
        print(f"[experiment-compare] saved {path}")


if __name__ == "__main__":
    main()
