"""Plot aggregate neural metrics with uncertainty bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from thesis.eval.aggregate import METRICS, aggregate_csv


def plot_metric_rounds(
    rounds: list[int],
    means: list[float],
    lower: list[float],
    upper: list[float],
    title: str,
    out_path: Path,
    ylabel: str,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = np.vstack(
        [
            np.asarray(means) - np.asarray(lower),
            np.asarray(upper) - np.asarray(means),
        ]
    )
    ax.errorbar(rounds, means, yerr=yerr, fmt="o-", capsize=4, linewidth=2)
    ax.set_xticks(rounds)
    ax.set_xlabel("Round count R")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_multi_seed_for_cipher(cipher: str, results_dir: Path) -> list[Path]:
    raw_path = results_dir / f"{cipher}_multi_seed_raw.csv"
    rows, aggregate_path = aggregate_csv(raw_path)
    if not rows:
        print(f"[plot] No multi-seed rows for {cipher} at {raw_path}")
        return []

    outputs: list[Path] = [aggregate_path]
    rounds = [int(row["rounds"]) for row in rows]
    delta_label = ""
    if rows[0].get("input_delta_left") not in (None, ""):
        delta_label = (
            f", Delta_P=(0x{int(rows[0]['input_delta_left']):04x}, "
            f"0x{int(rows[0]['input_delta_right']):04x})"
        )
    for metric in METRICS:
        means = [float(row[f"{metric}_mean"]) for row in rows]
        lower = [float(row[f"{metric}_ci95_low"]) for row in rows]
        upper = [float(row[f"{metric}_ci95_high"]) for row in rows]
        title = (
            f"{cipher.upper()}32/64 - {metric}\n"
            f"Mean with 95% CI across seeds{delta_label}"
        )
        out_path = results_dir / f"{cipher}_{metric}.png"
        plot_metric_rounds(rounds, means, lower, upper, title, out_path, ylabel=metric)
        outputs.append(out_path)
        print(f"[plot] saved {out_path}")
    return outputs


def plot_all(results_dir: Path, ciphers: list[str] | None = None) -> list[Path]:
    cipher_names = ciphers or sorted(
        {path.stem.replace("_multi_seed_raw", "") for path in results_dir.glob("*_multi_seed_raw.csv")}
    )
    outputs: list[Path] = []
    for cipher in cipher_names:
        outputs.extend(plot_multi_seed_for_cipher(cipher, results_dir))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate seeds and plot neural metrics")
    parser.add_argument("--results-dir", type=Path, default=Path("results/thesis"))
    parser.add_argument("--cipher", action="append", choices=["simon", "speck"])
    args = parser.parse_args()
    plot_all(args.results_dir, args.cipher)


if __name__ == "__main__":
    main()
