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
DEFAULT_DELTA_SWEEP: Sequence[float] = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.07,
    0.1,
    0.12,
    0.15,
)
DEFAULT_ABB_DIM_SWEEP: Sequence[int] = (
    1,
    25,
    50,
    100,
    250,
    500,
    1000,
    2000,
    5000,
    8000,
    12000,
)

DEFAULT_TAU: float = 0.1
DEFAULT_DELTA: float = 0.05
DEFAULT_ABB_DIM: int = 12000

DEFAULT_APPROACHES: Sequence[str] = (
    "combined_abb_conservative",
    "combined_abb_rms",
    "abb_formatting_only",
)

APPROACH_LABELS: Dict[str, str] = {
    "combined_abb_conservative": "Combined A-BB (Conservative)",
    "combined_abb_rms": "Combined A-BB (RMS)",
    "abb_formatting_only": "ABB Formatting Only",
}


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
        "--approaches",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Approach keys to extract from the analysis results. "
            "Defaults to Conservative, RMS, and Formatting-only Combined A-BB."
        ),
    )
    parser.add_argument(
        "--approach",
        type=str,
        default=None,
        help="Deprecated. Specify a single approach; prefer --approaches.",
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


def load_approach_records(
    results_file: Path, approaches: Sequence[str]
) -> List[Dict[str, Any]]:
    """Load per-judge metrics for the requested approaches."""
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}. Ensure the analysis script has produced it."
        )

    with open(results_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    records: List[Dict[str, Any]] = []
    for judge_name, judge_data in data.items():
        if not isinstance(judge_data, dict):
            continue
        approach_map = judge_data.get("approaches", {})
        if not isinstance(approach_map, dict):
            continue
        for approach in approaches:
            approach_data = approach_map.get(approach)
            if not isinstance(approach_data, dict):
                continue
            if not approach_data.get("success", False):
                continue
            validation = approach_data.get("validation", {})
            diagnostics = approach_data.get("diagnostics", {})
            if not isinstance(validation, dict):
                continue
            correlation = validation.get("correlation")
            if correlation is None:
                continue
            combined_sensitivity = (
                diagnostics.get("combined_sensitivity")
                if isinstance(diagnostics, dict)
                else None
            )
            formatting_rms = (
                diagnostics.get("context_adjusted_formatting_rms")
                if isinstance(diagnostics, dict)
                else None
            )
            schematic_rms = (
                diagnostics.get("schematic_context_rms")
                if isinstance(diagnostics, dict)
                else None
            )
            intrinsic_rms = (
                diagnostics.get("context_adjust_intrinsic_rms")
                if isinstance(diagnostics, dict)
                else None
            )
            combined_rms = None
            try:
                fmt_val = float(formatting_rms) if formatting_rms is not None else 0.0
                sch_val = float(schematic_rms) if schematic_rms is not None else 0.0
                combined_rms = float(np.sqrt(fmt_val**2 + sch_val**2))
            except (TypeError, ValueError):
                combined_rms = None
            records.append(
                {
                    "judge": judge_name,
                    "approach": approach,
                    "correlation": float(correlation),
                    "combined_sensitivity": (
                        float(combined_sensitivity)
                        if combined_sensitivity is not None
                        else float("nan")
                    ),
                    "formatting_rms": (
                        float(formatting_rms)
                        if formatting_rms is not None
                        else float("nan")
                    ),
                    "schematic_rms": (
                        float(schematic_rms)
                        if schematic_rms is not None
                        else float("nan")
                    ),
                    "intrinsic_rms": (
                        float(intrinsic_rms)
                        if intrinsic_rms is not None
                        else float("nan")
                    ),
                    "combined_rms": (
                        float(combined_rms)
                        if combined_rms is not None
                        else float("nan")
                    ),
                }
            )

    return records


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
    records: List[Dict[str, Any]],
    value_order: Sequence[float],
    approaches: Sequence[str],
) -> Dict[str, List[Dict[str, float]]]:
    """Aggregate correlation results for plotting."""
    summary: Dict[str, List[Dict[str, float]]] = {approach: [] for approach in approaches}
    for approach in approaches:
        for value in value_order:
            subset = [
                record
                for record in records
                if record["approach"] == approach
                and float(record["parameter_value"]) == float(value)
            ]
            if not subset:
                continue
            correlations = [item["correlation"] for item in subset]
            mean_corr, ci = compute_confidence_interval(correlations)
            sensitivities = np.array(
                [item.get("combined_sensitivity", float("nan")) for item in subset],
                dtype=float,
            )
            finite_sens = sensitivities[np.isfinite(sensitivities)]
            mean_sens = (
                float(np.mean(finite_sens)) if finite_sens.size else float("nan")
            )
            rms_values = np.array(
                [item.get("combined_rms", float("nan")) for item in subset], dtype=float
            )
            finite_rms = rms_values[np.isfinite(rms_values)]
            mean_rms = float(np.mean(finite_rms)) if finite_rms.size else float("nan")
            summary[approach].append(
                {
                    "parameter_value": float(value),
                    "mean_correlation": mean_corr,
                    "ci": 0.0 if np.isnan(ci) else ci,
                    "mean_sensitivity": mean_sens,
                    "mean_combined_rms": mean_rms,
                }
            )
    return summary


def plot_scaling_curve(
    summaries_by_approach: Dict[str, List[Dict[str, float]]],
    xlabel: str,
    output_path: Path,
) -> None:
    """Plot mean correlation with confidence intervals."""
    if not any(summaries_by_approach.values()):
        raise ValueError(f"No data available to plot for {output_path.name}.")

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (approach, summary) in enumerate(summaries_by_approach.items()):
        if not summary:
            continue
        x_values = [item["parameter_value"] for item in summary]
        y_values = [item["mean_correlation"] for item in summary]
        y_err = [item["ci"] for item in summary]
        sensitivity_values = [
            item.get("mean_sensitivity", float("nan")) for item in summary
        ]
        rms_values = [item.get("mean_combined_rms", float("nan")) for item in summary]
        sens_array = np.array(sensitivity_values, dtype=float)
        finite_sens = sens_array[np.isfinite(sens_array)]
        avg_sensitivity = (
            float(np.mean(finite_sens)) if finite_sens.size else float("nan")
        )
        rms_array = np.array(rms_values, dtype=float)
        finite_rms = rms_array[np.isfinite(rms_array)]
        avg_combined_rms = (
            float(np.mean(finite_rms)) if finite_rms.size else float("nan")
        )
        approach_label = APPROACH_LABELS.get(approach, approach)
        legend_details: List[str] = []
        if np.isfinite(avg_combined_rms):
            legend_details.append(f"mean combined RMS {avg_combined_rms:.3f}")
        elif np.isfinite(avg_sensitivity):
            # Fall back to normalized sensitivity when combined RMS unavailable
            legend_details.append(f"normalized sensitivity {avg_sensitivity:.4g}")
        legend_label = (
            f"{approach_label} ({', '.join(legend_details)})"
            if legend_details
            else approach_label
        )
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_err,
            fmt="-o",
            capsize=4,
            linewidth=2.0,
            markersize=6,
            label=legend_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Correlation")
    ax.set_title(f"Correlation vs {xlabel}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Scaling Strategy")

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
    approaches: Sequence[str],
) -> List[Dict[str, Any]]:
    """Execute the analysis for each value in the sweep and capture correlations."""
    records: List[Dict[str, Any]] = []

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

        approach_records = load_approach_records(args.results_file, approaches)
        if not approach_records:
            print(
                "   ⚠️  No correlations found for requested approaches. "
                "Skipping this configuration."
            )
            continue

        counts: Dict[str, int] = {}
        for record in approach_records:
            records.append(
                {
                    **record,
                    "parameter_value": float(value),
                }
            )
            counts[record["approach"]] = counts.get(record["approach"], 0) + 1

        print(
            "   ✅ Recorded correlations for value {value}: {details}".format(
                value=value,
                details=", ".join(
                    f"{APPROACH_LABELS.get(approach, approach)}={count}"
                    for approach, count in sorted(counts.items())
                ),
            )
        )

    return records


def ensure_output_dir(path: Path) -> None:
    """Create the output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())

    if args.approaches is None:
        if args.approach:
            args.approaches = [args.approach]
        else:
            args.approaches = list(DEFAULT_APPROACHES)
    elif args.approach and args.approach not in args.approaches:
        args.approaches.insert(0, args.approach)

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
            approaches=args.approaches,
        )

        summary_by_approach = summarize_records(
            records, sweep.values, args.approaches
        )
        output_path = args.output_dir / sweep.filename
        try:
            plot_scaling_curve(summary_by_approach, sweep.xlabel, output_path)
            print(f"📈 Saved plot to {output_path}")
        except ValueError as exc:
            print(f"⚠️  Skipped plot for {sweep.name}: {exc}")


if __name__ == "__main__":
    main()
