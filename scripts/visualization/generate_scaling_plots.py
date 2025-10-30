#!/usr/bin/env python3
"""
Generate scaling plots for Combined A-BB analysis sweeps.

This script runs the Combined A-BB analysis multiple times while sweeping over
key hyperparameters (ABB dimensionality, target tau, target delta). After each
run it aggregates the resulting correlation metrics across judges, computes
95% confidence intervals, and saves visualization plots to disk.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("default")
sns.set_palette("husl")

# Sweep definitions
DEFAULT_TAU_SWEEP: Sequence[float] = (0.001, 0.01, 0.1, 0.5, 1.0, 2.0)
DEFAULT_DELTA_SWEEP: Sequence[float] = (0.001, 0.01, 0.05, 0.1, 0.15)
DEFAULT_ABB_DIM_SWEEP: Sequence[int] = (1, 500, 12000)

DEFAULT_TAU: float = 0.1
DEFAULT_DELTA: float = 0.05
DEFAULT_ABB_DIM: int = 12000

APPROACH_DEFAULT = "combined_abb_rms"


def find_repo_root(start: Path) -> Path:
    """Locate the repository root, preferring a .git directory when present."""
    pyproject_candidate: Optional[Path] = None
    for parent in start.parents:
        if (parent / ".git").exists():
            return parent
        if pyproject_candidate is None and (parent / "pyproject.toml").exists():
            pyproject_candidate = parent
    if pyproject_candidate is not None:
        return pyproject_candidate
    # Fallback to two levels up if markers are missing
    return start.parents[2]


@dataclass(frozen=True)
class SweepConfig:
    name: str
    values: Sequence[float]
    xlabel: str
    filename: str


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    default_data_path = repo_root / "sos-addl-data" / "InDepthAnalysis"
    default_output_dir = (
        repo_root / "combined_abb_visualizations" / "scaling-sweeps"
    )

    parser = argparse.ArgumentParser(
        description="Generate scaling plots for Combined A-BB analysis."
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=Path,
        default=default_data_path,
        help="Base path to Arena-Hard evaluation data (judge folders).",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "Path to combined_abb_analysis_results.json. "
            "Defaults to <data-path>/combined_abb_analysis_results.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where output PNG plots will be written.",
    )
    parser.add_argument(
        "--analysis-script",
        type=Path,
        default=repo_root / "scripts" / "analysis" / "run_combined_abb_analysis.py",
        help="Path to run_combined_abb_analysis.py.",
    )
    parser.add_argument(
        "--approach",
        type=str,
        default=APPROACH_DEFAULT,
        help="Approach key to extract from the analysis results.",
    )
    parser.add_argument(
        "--tau-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_TAU_SWEEP),
        help="Override sweep values for target tau.",
    )
    parser.add_argument(
        "--delta-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTA_SWEEP),
        help="Override sweep values for target delta.",
    )
    parser.add_argument(
        "--abb-dim-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_ABB_DIM_SWEEP),
        help="Override sweep values for ABB dimensionality.",
    )
    parser.add_argument(
        "--default-tau",
        type=float,
        default=DEFAULT_TAU,
        help="Default target tau when it is not the swept parameter.",
    )
    parser.add_argument(
        "--default-delta",
        type=float,
        default=DEFAULT_DELTA,
        help="Default target delta when it is not the swept parameter.",
    )
    parser.add_argument(
        "--default-abb-dim",
        type=int,
        default=DEFAULT_ABB_DIM,
        help="Default ABB dimensionality when it is not the swept parameter.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip running the analysis script; only read existing results file.",
    )
    return parser.parse_args()


def build_command(
    analysis_script: Path,
    data_path: Path,
    enable_shrinkage: bool,
    shrink_center: str,
    strict_abb: bool,
    target_tau: float,
    target_delta: float,
    abb_dimensionality: int,
) -> List[str]:
    """Construct the command to run the Combined A-BB analysis script."""
    command: List[str] = [
        sys.executable,
        str(analysis_script),
        "-d",
        str(data_path),
        "--target-tau",
        str(target_tau),
        "--target-delta",
        str(target_delta),
        "--abb-dimensionality",
        str(abb_dimensionality),
    ]

    if enable_shrinkage:
        command.append("--enable-shrinkage")
        command.extend(["--shrink-center", shrink_center])

    if strict_abb:
        command.append("--strict-abb")

    return command


def load_correlations(
    results_file: Path, approach: str
) -> List[Tuple[str, float]]:
    """Load correlation metrics for each judge from the results file."""
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}. Ensure the analysis script has produced it."
        )

    with open(results_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    correlations: List[Tuple[str, float]] = []
    for judge_name, judge_data in data.items():
        if not isinstance(judge_data, dict):
            continue
        approach_data = (
            judge_data.get("approaches", {}).get(approach) if judge_data else None
        )
        if not isinstance(approach_data, dict):
            continue
        if not approach_data.get("success", False):
            continue
        validation = approach_data.get("validation", {})
        if not isinstance(validation, dict):
            continue
        correlation = validation.get("correlation")
        if correlation is None:
            continue
        correlations.append((judge_name, float(correlation)))

    return correlations


def compute_confidence_interval(values: Iterable[float]) -> Tuple[float, float]:
    """Return mean and 95% CI half-width using normal approximation."""
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")

    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, float("nan")

    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(arr.size)
    ci = 1.96 * se
    return mean, ci


def summarize_records(
    records: List[Dict[str, Any]], value_order: Sequence[float]
) -> List[Dict[str, float]]:
    """Aggregate correlation results for plotting."""
    summary: List[Dict[str, float]] = []
    for value in value_order:
        subset = [r["correlation"] for r in records if r["parameter_value"] == value]
        if not subset:
            continue
        mean, ci = compute_confidence_interval(subset)
        summary.append(
            {
                "parameter_value": value,
                "mean_correlation": mean,
                "ci": 0.0 if np.isnan(ci) else ci,
            }
        )
    return summary


def plot_scaling_curve(
    summary: List[Dict[str, float]],
    xlabel: str,
    output_path: Path,
) -> None:
    """Plot mean correlation with confidence intervals."""
    if not summary:
        raise ValueError(f"No data available to plot for {output_path.name}.")

    x_values = [item["parameter_value"] for item in summary]
    y_values = [item["mean_correlation"] for item in summary]
    y_err = [item["ci"] for item in summary]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        x_values,
        y_values,
        yerr=y_err,
        fmt="-o",
        capsize=4,
        linewidth=2.0,
        markersize=6,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Correlation")
    ax.set_title(f"Correlation vs {xlabel}")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_sweep(
    sweep: SweepConfig,
    args: argparse.Namespace,
    repo_root: Path,
    enable_shrinkage: bool,
    shrink_center: str,
    strict_abb: bool,
) -> List[Dict[str, Any]]:
    """Execute the analysis for each value in the sweep and capture correlations."""
    records: List[Dict[str, float]] = []

    for value in sweep.values:
        target_tau = args.default_tau
        target_delta = args.default_delta
        abb_dimensionality = args.default_abb_dim

        if sweep.name == "abb_dimensionality":
            abb_dimensionality = int(value)
        elif sweep.name == "target_tau":
            target_tau = float(value)
        elif sweep.name == "target_delta":
            target_delta = float(value)

        command = build_command(
            analysis_script=args.analysis_script,
            data_path=args.data_path,
            enable_shrinkage=enable_shrinkage,
            shrink_center=shrink_center,
            strict_abb=strict_abb,
            target_tau=target_tau,
            target_delta=target_delta,
            abb_dimensionality=int(abb_dimensionality),
        )

        print(
            f"\n▶️  Running sweep '{sweep.name}' with value {value} "
            f"(tau={target_tau}, delta={target_delta}, dim={abb_dimensionality})"
        )

        if not args.skip_analysis:
            subprocess.run(command, check=True, cwd=repo_root)
        else:
            print("   ⚠️  Skipping analysis run; reusing existing results file.")

        correlations = load_correlations(args.results_file, args.approach)
        if not correlations:
            print(
                f"   ⚠️  No correlations found for approach '{args.approach}'. "
                "Skipping this configuration."
            )
            continue

        for judge, corr in correlations:
            records.append(
                {
                    "parameter_value": float(value),
                    "judge": judge,
                    "correlation": float(corr),
                }
            )

        print(
            f"   ✅ Recorded {len(correlations)} correlations for value {value}."
        )

    return records


def ensure_output_dir(path: Path) -> None:
    """Create the output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())

    if args.results_file is None:
        args.results_file = args.data_path / "combined_abb_analysis_results.json"

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Base path does not exist: {args.data_path}. "
            "Use --data-path to specify the Arena-Hard data directory."
        )

    if not args.analysis_script.exists():
        raise FileNotFoundError(
            f"Analysis script not found: {args.analysis_script}. "
            "Ensure run_combined_abb_analysis.py is accessible."
        )

    ensure_output_dir(args.output_dir)

    sweeps = [
        SweepConfig(
            name="abb_dimensionality",
            values=tuple(args.abb_dim_values),
            xlabel="ABB Dimensionality",
            filename="correlation_vs_abb_dimensionality.png",
        ),
        SweepConfig(
            name="target_tau",
            values=tuple(args.tau_values),
            xlabel="Target Tau",
            filename="correlation_vs_target_tau.png",
        ),
        SweepConfig(
            name="target_delta",
            values=tuple(args.delta_values),
            xlabel="Target Delta",
            filename="correlation_vs_target_delta.png",
        ),
    ]

    for sweep in sweeps:
        records = run_sweep(
            sweep=sweep,
            args=args,
            repo_root=repo_root,
            enable_shrinkage=True,
            shrink_center="mean",
            strict_abb=True,
        )

        summary = summarize_records(records, sweep.values)
        output_path = args.output_dir / sweep.filename
        try:
            plot_scaling_curve(summary, sweep.xlabel, output_path)
            print(f"📈 Saved plot to {output_path}")
        except ValueError as exc:
            print(f"⚠️  Skipped plot for {sweep.name}: {exc}")


if __name__ == "__main__":
    main()
