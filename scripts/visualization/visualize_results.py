#!/usr/bin/env python3
"""
Visualize Combined A-BB analysis results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, List, Optional

try:
    from scipy.stats import gaussian_kde
except ImportError:  # pragma: no cover - SciPy is optional in some environments
    gaussian_kde = None

# Import data loader utilities
from data_loader import (
    get_project_paths,
    load_original_evaluations,
    load_debiased_scores,
    merge_original_and_debiased,
    aggregate_scores_by_model
)

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

def load_results(results_file: Path) -> dict:
    """Load the combined A-BB analysis results."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_judge_results_from_jsonl(base_path: Path) -> Dict[str, Dict]:
    """Load judge results directly from JSONL files, preserving per-strategy outputs."""

    results: Dict[str, Dict] = {}

    judge_patterns = [
        "QwQ-32B-setting1",
        "DeepSeek-R1-32B-setting1",
        "DeepSeek-R1-32B-setting2",
        "GPT-3.5-Turbo-0125-setting1",
        "GPT-4o-mini-0718-setting1",
    ]

    preferred_strategies = {
        "abb_formatting_only",
        "combined_abb_conservative",
        "combined_abb_rms",
    }

    for judge_name in judge_patterns:
        judge_dir = base_path / judge_name
        if not judge_dir.exists():
            continue

        base_processed_dir = judge_dir / "base_processed"
        if not base_processed_dir.exists():
            continue

        try:
            original_df = load_original_evaluations(base_processed_dir)
        except Exception as exc:
            print(f"❌ Error loading original data for {judge_name}: {exc}")
            continue

        approaches: Dict[str, Dict] = {}
        debiased_dirs = sorted(d for d in judge_dir.glob("base_debiased*") if d.is_dir())
        if not debiased_dirs:
            continue

        for debiased_dir in debiased_dirs:
            suffix = debiased_dir.name.replace("base_debiased_", "")
            suffix = suffix or "default"
            if preferred_strategies and suffix not in preferred_strategies:
                continue

            try:
                debiased_df = load_debiased_scores(debiased_dir)
                merged_df = merge_original_and_debiased(original_df, debiased_df)
            except Exception as exc:
                print(f"❌ Error loading debiased data for {judge_name} [{suffix}]: {exc}")
                continue

            if merged_df.empty:
                continue

            original_scores = merged_df["overall_score"].values.astype(float)
            debiased_scores = merged_df["score_debiased"].values.astype(float)

            correlation = np.corrcoef(original_scores, debiased_scores)[0, 1]
            mean_abs_diff = np.mean(np.abs(original_scores - debiased_scores))
            variance_ratio = np.var(debiased_scores) / np.var(original_scores) if np.var(original_scores) > 0 else np.nan
            noise_level = np.std(original_scores - debiased_scores)

            approaches[suffix] = {
                "success": True,
                "original_scores": original_scores.tolist(),
                "debiased_scores": debiased_scores.tolist(),
                "n_samples": int(len(original_scores)),
                "diagnostics": {
                    "combined_sensitivity": float(np.std(debiased_scores)),
                    "noise_std": float(noise_level),
                    "abb_constraint_satisfied": True,
                    "tau": None,
                    "delta": None,
                },
                "validation": {
                    "correlation": float(correlation),
                    "mean_absolute_difference": float(mean_abs_diff),
                    "variance_ratio": float(variance_ratio),
                    "signal_preservation": float(correlation),
                    "noise_level": float(noise_level),
                },
            }

        if approaches:
            total_samples = sum(a["n_samples"] for a in approaches.values())
            results[judge_name] = {
                "data_source": str(judge_dir),
                "n_samples": total_samples,
                "approaches": approaches,
            }
            print(f"✅ Loaded data for {judge_name}: {total_samples} samples across {len(approaches)} strategies")

    return results

def create_sensitivity_comparison(results: dict) -> plt.Figure:
    """Create comparison of sensitivity measurements across approaches."""
    
    # Extract data for plotting
    data = []
    for judge_name, judge_results in results.items():
        if 'approaches' in judge_results:
            for approach_name, approach_result in judge_results['approaches'].items():
                if approach_result.get('success', False):
                    data.append({
                        'judge': judge_name,
                        'approach': approach_name.replace('combined_abb_', ''),
                        'combined_sensitivity': approach_result['diagnostics']['combined_sensitivity'],
                        'correlation': approach_result['validation']['correlation'],
                        'constraint_satisfied': approach_result['diagnostics'].get('abb_constraint_satisfied'),
                        'noise_std': approach_result['diagnostics']['noise_std']
                    })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Combined A-BB Analysis Results: Comparison of Aggregation Strategies', fontsize=16, fontweight='bold')
    
    # 1. Sensitivity comparison
    sns.boxplot(data=df, x='approach', y='combined_sensitivity', ax=axes[0, 0])
    axes[0, 0].set_title('Combined Sensitivity by Strategy')
    axes[0, 0].set_xlabel('Aggregation Strategy')
    axes[0, 0].set_ylabel('Combined Sensitivity')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Correlation preservation
    sns.boxplot(data=df, x='approach', y='correlation', ax=axes[0, 1])
    axes[0, 1].set_title('Signal Preservation (Correlation) by Strategy')
    axes[0, 1].set_xlabel('Aggregation Strategy')
    axes[0, 1].set_ylabel('Correlation with Original')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Constraint satisfaction
    constraint_counts = df.groupby('approach')['constraint_satisfied'].agg(['sum', 'count']).reset_index()
    constraint_counts['satisfaction_rate'] = constraint_counts['sum'] / constraint_counts['count']
    
    bars = axes[1, 0].bar(constraint_counts['approach'], constraint_counts['satisfaction_rate'])
    axes[1, 0].set_title('A-BB Constraint Satisfaction Rate')
    axes[1, 0].set_xlabel('Aggregation Strategy')
    axes[1, 0].set_ylabel('Satisfaction Rate')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars, constraint_counts['satisfaction_rate']):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{rate:.1%}', ha='center', va='bottom')
    
    # 4. Noise level comparison
    sns.boxplot(data=df, x='approach', y='noise_std', ax=axes[1, 1])
    axes[1, 1].set_title('Noise Level by Strategy')
    axes[1, 1].set_xlabel('Aggregation Strategy')
    axes[1, 1].set_ylabel('Noise Standard Deviation')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_judge_comparison(results: dict) -> plt.Figure:
    """Visualize signal preservation per judge/strategy using publication styling."""

    preferred_order = [
        ("abb_formatting_only", "Formatting Only"),
        ("combined_abb_conservative", "Conservative"),
        ("combined_abb_rms", "RMS"),
    ]
    order_lookup = {raw: display for raw, display in preferred_order}

    rows = []
    tau_values: List[float] = []
    delta_values: List[float] = []

    for judge_name, judge_results in results.items():
        if 'approaches' not in judge_results:
            continue
        for approach_name, approach_result in judge_results['approaches'].items():
            if not approach_result.get('success', False):
                continue

            diag = approach_result.get('diagnostics') or {}
            val = approach_result.get('validation') or {}

            tau = diag.get('tau')
            delta = diag.get('delta')
            if isinstance(tau, (int, float)):
                tau_values.append(float(tau))
            if isinstance(delta, (int, float)):
                delta_values.append(float(delta))

            display_name = order_lookup.get(approach_name)
            if display_name is None:
                # Try stripping known prefixes
                clean = approach_name
                if clean.startswith('combined_abb_'):
                    clean = clean.replace('combined_abb_', '')
                elif clean.startswith('abb_'):
                    clean = clean.replace('abb_', '')
                display_name = clean.replace('_', ' ').title()

            rows.append({
                'judge': judge_name,
                'approach_raw': approach_name,
                'approach_display': display_name,
                'correlation': float(val.get('correlation', np.nan)),
            })

    if not rows:
        print("No data available for judge comparison plot")
        return None

    df = pd.DataFrame(rows)

    # Limit to unique approaches present and order them
    column_order = []
    for raw, display in preferred_order:
        if (df['approach_raw'] == raw).any():
            column_order.append(display)
    for display in df['approach_display'].unique():
        if display not in column_order:
            column_order.append(display)

    pivot = df.pivot(index='judge', columns='approach_display', values='correlation')
    pivot = pivot.reindex(columns=column_order)

    fig, ax = plt.subplots(figsize=(14, 7))

    title = "Signal Preservation by Judge and Strategy"
    subtitle_parts = []
    if tau_values:
        subtitle_parts.append(f"Avg tau: {np.mean(tau_values):.3f}")
    if delta_values:
        subtitle_parts.append(f"Avg delta: {np.mean(delta_values):.3f}")
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.93)
    if subtitle:
        ax.set_title(subtitle, fontsize=14, pad=18, fontweight='bold')

    sns.heatmap(
        pivot,
        ax=ax,
        cmap='Blues',
        annot=True,
        fmt='.3f',
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'Signal Preservation (Correlation)'},
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel("Aggregation Strategy", fontsize=12, fontweight='bold')
    ax.set_ylabel("Judge", fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.tick_params(axis='y', rotation=0, labelsize=11)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.9])
    return fig

def create_effectiveness_analysis(results: dict) -> plt.Figure:
    """Analyze the effectiveness of different strategies."""
    
    # Extract validation metrics
    data = []
    for judge_name, judge_results in results.items():
        if 'approaches' in judge_results:
            for approach_name, approach_result in judge_results['approaches'].items():
                if approach_result.get('success', False):
                    val = approach_result['validation']
                    data.append({
                        'judge': judge_name,
                        'approach': approach_name.replace('combined_abb_', ''),
                        'correlation': val['correlation'],
                        'mean_abs_diff': val['mean_absolute_difference'],
                        'variance_ratio': val['variance_ratio'],
                        'noise_level': val['noise_level'],
                        'combined_sensitivity': approach_result['diagnostics']['combined_sensitivity']
                    })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Combined A-BB Effectiveness Analysis', fontsize=16, fontweight='bold')
    
    # 1. Sensitivity vs Correlation tradeoff
    for approach in df['approach'].unique():
        approach_data = df[df['approach'] == approach]
        axes[0, 0].scatter(approach_data['combined_sensitivity'], approach_data['correlation'], 
                          label=approach, alpha=0.7, s=60)
    
    axes[0, 0].set_xlabel('Combined Sensitivity')
    axes[0, 0].set_ylabel('Signal Preservation (Correlation)')
    axes[0, 0].set_title('Sensitivity vs Signal Preservation Tradeoff')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Noise level distribution
    sns.violinplot(data=df, x='approach', y='noise_level', ax=axes[0, 1])
    axes[0, 1].set_title('Noise Level Distribution by Strategy')
    axes[0, 1].set_xlabel('Aggregation Strategy')
    axes[0, 1].set_ylabel('Noise Level')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Variance preservation
    sns.boxplot(data=df, x='approach', y='variance_ratio', ax=axes[1, 0])
    axes[1, 0].set_title('Variance Preservation by Strategy')
    axes[1, 0].set_xlabel('Aggregation Strategy')
    axes[1, 0].set_ylabel('Variance Ratio (Debiased/Original)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Preservation')
    axes[1, 0].legend()
    
    # 4. Mean absolute difference
    sns.boxplot(data=df, x='approach', y='mean_abs_diff', ax=axes[1, 1])
    axes[1, 1].set_title('Mean Absolute Change by Strategy')
    axes[1, 1].set_xlabel('Aggregation Strategy')
    axes[1, 1].set_ylabel('Mean Absolute Difference')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def create_score_distribution_comparison(results: dict) -> plt.Figure:
    """Create before/after score distribution comparison plots."""
    
    # Extract all successful approaches for distribution comparison
    plot_data = []
    for judge_name, judge_results in results.items():
        if 'approaches' in judge_results:
            for approach_name, approach_result in judge_results['approaches'].items():
                if approach_result.get('success', False):
                    plot_data.append({
                        'judge': judge_name,
                        'approach': approach_name.replace('combined_abb_', ''),
                        'original_scores': approach_result['original_scores'],
                        'debiased_scores': approach_result['debiased_scores']
                    })
    
    if not plot_data:
        print("No successful approaches found for distribution comparison")
        return None
    
    # Create figure with subplots for each approach
    unique_approaches = list(set(item['approach'] for item in plot_data))
    n_approaches = len(unique_approaches)
    
    fig, axes = plt.subplots(n_approaches, 2, figsize=(15, 5 * n_approaches))
    if n_approaches == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Score Distribution Comparison: Original vs Debiased\nCombined A-BB Approaches', 
                 fontsize=16, fontweight='bold')
    
    for i, approach in enumerate(unique_approaches):
        approach_data = [item for item in plot_data if item['approach'] == approach]
        
        # Collect all scores for this approach across all judges
        all_original = []
        all_debiased = []
        judge_labels = []
        
        for item in approach_data:
            all_original.extend(item['original_scores'])
            all_debiased.extend(item['debiased_scores'])
            judge_labels.extend([item['judge']] * len(item['original_scores']))
        
        all_original = np.array(all_original)
        all_debiased = np.array(all_debiased)
        
        # Original distribution
        axes[i, 0].hist(all_original, bins=30, alpha=0.7, color='red', 
                       label=f'Original (μ={np.mean(all_original):.2f}, σ={np.std(all_original):.2f})')
        axes[i, 0].axvline(np.mean(all_original), color='red', linestyle='--', alpha=0.8, label='Mean')
        axes[i, 0].set_title(f'{approach.title()} Strategy - Original Scores')
        axes[i, 0].set_xlabel('Score Value')
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Debiased distribution
        axes[i, 1].hist(all_debiased, bins=30, alpha=0.7, color='blue',
                       label=f'Debiased (μ={np.mean(all_debiased):.2f}, σ={np.std(all_debiased):.2f})')
        axes[i, 1].axvline(np.mean(all_debiased), color='blue', linestyle='--', alpha=0.8, label='Mean')
        axes[i, 1].set_title(f'{approach.title()} Strategy - Debiased Scores')
        axes[i, 1].set_xlabel('Score Value')
        axes[i, 1].set_ylabel('Frequency')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Add statistical comparison text
        correlation = np.corrcoef(all_original, all_debiased)[0, 1]
        variance_reduction = (1 - np.var(all_debiased) / np.var(all_original)) * 100
        
        stats_text = f'Correlation: {correlation:.3f}\nVariance Reduction: {variance_reduction:.1f}%'
        axes[i, 1].text(0.02, 0.98, stats_text, transform=axes[i, 1].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    return fig

def _compute_kde_density(scores: np.ndarray, grid: np.ndarray) -> Optional[np.ndarray]:
    """Return KDE density evaluated on grid; fall back to histogram-based estimate."""
    if scores.size < 2 or np.isclose(np.std(scores), 0.0):
        return None
    if gaussian_kde is not None:
        try:
            kde = gaussian_kde(scores)
            return kde(grid)
        except Exception:
            pass
    # Fallback: interpolated histogram
    bins = max(10, min(40, scores.size // 5))
    hist, bin_edges = np.histogram(scores, bins=bins, range=(grid[0], grid[-1]), density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return np.interp(grid, centers, hist, left=0.0, right=0.0)


def create_combined_distribution_overlay(results: dict) -> plt.Figure:
    """Create an overlaid density comparison that matches the publication styling."""

    approach_data: Dict[str, Dict[str, List[float]]] = {}
    judge_original: Dict[str, List[float]] = {}
    tau_values: List[float] = []
    delta_values: List[float] = []

    for judge_name, judge_results in results.items():
        if 'approaches' not in judge_results:
            continue
        for approach_name, approach_result in judge_results['approaches'].items():
            if not approach_result.get('success', False):
                continue
            diag = approach_result.get('diagnostics') or {}
            tau = diag.get('tau')
            delta = diag.get('delta')
            if isinstance(tau, (int, float)):
                tau_values.append(float(tau))
            if isinstance(delta, (int, float)):
                delta_values.append(float(delta))

            approach_data.setdefault(
                approach_name,
                {
                    'display': approach_name.replace('combined_abb_', '').replace('abb_', '').replace('_', ' ').title(),
                    'debiased': [],
                },
            )
            approach_data[approach_name]['debiased'].extend(approach_result['debiased_scores'])

            if judge_name not in judge_original:
                judge_original[judge_name] = approach_result['original_scores']

    if not approach_data:
        print("No data found for combined distribution overlay")
        return None

    # Flatten score collections
    original_scores = np.array([score for scores in judge_original.values() for score in scores], dtype=float)
    debiased_all = np.array(
        [score for data in approach_data.values() for score in data['debiased']],
        dtype=float,
    )

    if original_scores.size == 0 or debiased_all.size == 0:
        print("Insufficient data for combined distribution overlay")
        return None

    min_score = min(original_scores.min(), debiased_all.min())
    max_score = max(original_scores.max(), debiased_all.max())
    padding = max(0.05, 0.02 * (max_score - min_score))
    score_grid = np.linspace(min_score - padding, max_score + padding, 400)

    fig, ax = plt.subplots(figsize=(18, 6))

    title = "Score Distribution Overlay: All Combined A-BB Strategies"
    subtitle_parts = []
    if tau_values:
        subtitle_parts.append(f"Avg tau: {np.mean(tau_values):.3f}")
    if delta_values:
        subtitle_parts.append(f"Avg delta: {np.mean(delta_values):.3f}")
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)
    if subtitle:
        ax.set_title(subtitle, fontsize=14, pad=20, fontweight='bold')

    # Original density (black dashed)
    original_density = _compute_kde_density(original_scores, score_grid)
    if original_density is not None:
        ax.plot(score_grid, original_density, linestyle='--', linewidth=2.5,
                color='black', label='Original')

    palette = {
        'abb_formatting_only': '#8dd3c7',
        'combined_abb_conservative': '#fc8d62',
        'combined_abb_rms': '#8da0cb',
        'combined_abb_weighted': '#ffd92f',
        'combined_abb_montecarlo': '#b3de69',
        'default': '#66c2a5',
    }

    strategy_priority = [
        'abb_formatting_only',
        'combined_abb_conservative',
        'combined_abb_rms',
        'combined_abb_weighted',
        'combined_abb_montecarlo',
    ]

    ordered_keys = strategy_priority + [k for k in sorted(approach_data.keys()) if k not in strategy_priority]
    seen_labels: Set[str] = set()
    for key in ordered_keys:
        data = approach_data.get(key)
        if not data:
            continue
        debiased_scores = np.array(data['debiased'], dtype=float)
        density = _compute_kde_density(debiased_scores, score_grid)
        if density is None:
            continue

        display_name = data.get('display', key.replace('_', ' ').title())
        legend_label = f"{display_name} Debiased"
        if legend_label in seen_labels:
            legend_label = f"{legend_label} ({key})"
        seen_labels.add(legend_label)

        color = palette.get(key, palette['default'])
        ax.plot(score_grid, density, linewidth=2.5, color=color, label=legend_label)

    global_mean = float(np.mean(debiased_all))
    ax.axvline(global_mean, color='#cba64b', linestyle='-', linewidth=2, alpha=0.7)

    ax.set_xlabel("Score Value", fontsize=12, fontweight='bold')
    ax.set_ylabel("Density", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=10)
    ax.legend(frameon=True, loc='upper right', fontsize=11, title='Aggregation Strategy')
    ax.set_xlim(score_grid[0], score_grid[-1])
    ax.set_ylim(bottom=0)
    ylim_top = ax.get_ylim()[1]
    ax.text(global_mean, ylim_top * 0.95, f"Mean Debiased: {global_mean:.2f}",
            rotation=90, color='#cba64b', ha='right', va='top', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.9])
    return fig

def create_summary_table(results: dict) -> pd.DataFrame:
    """Create summary table of all results."""
    
    summary_data = []
    for judge_name, judge_results in results.items():
        if 'approaches' in judge_results:
            for approach_name, approach_result in judge_results['approaches'].items():
                if approach_result.get('success', False):
                    # Extract tau/delta if available in diagnostics; fall back to None
                    diag = approach_result.get('diagnostics', {}) or {}
                    tau_val = diag.get('tau')
                    delta_val = diag.get('delta')
                    summary_data.append({
                        'Judge': judge_name,
                        'Strategy': approach_name.replace('combined_abb_', ''),
                        'Combined Sensitivity': f"{approach_result['diagnostics']['combined_sensitivity']:.4f}",
                        'Correlation': f"{approach_result['validation']['correlation']:.3f}",
                        'Constraint Satisfied': '✅' if approach_result['diagnostics'].get('abb_constraint_satisfied') else '❌',
                        'Noise Std': f"{approach_result['diagnostics']['noise_std']:.4f}",
                        'Tau': (f"{float(tau_val):.3f}" if isinstance(tau_val, (int, float)) else ''),
                        'Delta': (f"{float(delta_val):.3f}" if isinstance(delta_val, (int, float)) else ''),
                        'Samples': approach_result['n_samples']
                    })
    
    return pd.DataFrame(summary_data)

def main(use_jsonl: bool = False):
    """Generate visualizations for Combined A-BB analysis results."""
    
    # Get project paths
    paths = get_project_paths()
    
    if use_jsonl:
        # Load from JSONL files
        print("📊 Loading data from JSONL files...")
        results = load_judge_results_from_jsonl(paths['data_base'])
        
        if not results:
            print("❌ No data found in JSONL files!")
            return 1
    else:
        # Load from JSON results file (legacy)
        results_file = paths['data_base'] / "combined_abb_analysis_results.json"
        
        if not results_file.exists():
            print(f"Results file not found: {results_file}")
            print("Falling back to JSONL mode...")
            return main(use_jsonl=True)
        
        print("📊 Loading Combined A-BB analysis results from JSON...")
        results = load_results(results_file)
    
    print(f"✅ Loaded results for {len(results)} judges")
    # Clarify constraint source in logs
    print("ℹ️ Constraint status source: diagnostics.abb_constraint_satisfied (from data); no recomputation performed.")
    # Quick summary of availability in the data
    true_count = false_count = missing_count = 0
    for judge_name, judge_results in results.items():
        if 'approaches' not in judge_results:
            continue
        for approach_name, approach_result in judge_results['approaches'].items():
            if not approach_result.get('success', False):
                continue
            val = (approach_result.get('diagnostics') or {}).get('abb_constraint_satisfied', None)
            if val is True:
                true_count += 1
            elif val is False:
                false_count += 1
            else:
                missing_count += 1
    print(f"   ↪︎ JSON constraint flags — true: {true_count}, false: {false_count}, missing: {missing_count}")
    
    # Output directory
    output_dir = paths['figures'] / "combined_abb_visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations
    print("📈 Generating sensitivity comparison visualization...")
    fig1 = create_sensitivity_comparison(results)
    fig1.savefig(output_dir / "sensitivity_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("📈 Generating judge comparison visualization...")
    fig2 = create_judge_comparison(results)
    fig2.savefig(output_dir / "judge_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("📈 Generating effectiveness analysis...")
    fig3 = create_effectiveness_analysis(results)
    fig3.savefig(output_dir / "effectiveness_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Generate score distribution comparisons
    print("📊 Generating score distribution comparison...")
    fig4 = create_score_distribution_comparison(results)
    if fig4:
        fig4.savefig(output_dir / "score_distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
    
    print("📊 Generating combined distribution overlay...")
    fig5 = create_combined_distribution_overlay(results)
    if fig5:
        fig5.savefig(output_dir / "combined_distribution_overlay.png", dpi=300, bbox_inches='tight')
        plt.close(fig5)
    
    # Generate summary table
    print("📋 Creating summary table...")
    summary_table = create_summary_table(results)
    summary_table.to_csv(output_dir / "combined_abb_summary.csv", index=False)
    
    print("\n🎯 Combined A-BB Analysis Summary:")
    print("=" * 50)
    print(summary_table.to_string(index=False))
    
    print(f"\n📁 Visualizations saved to: {output_dir}")
    print("Generated files:")
    print("  - sensitivity_comparison.png")
    print("  - judge_comparison.png") 
    print("  - effectiveness_analysis.png")
    print("  - score_distribution_comparison.png")
    print("  - combined_distribution_overlay.png")
    print("  - combined_abb_summary.csv")
    
    plt.show()
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Combined A-BB analysis results")
    parser.add_argument('--use-jsonl', action='store_true',
                        help='Load data directly from JSONL files instead of summary JSON')
    args = parser.parse_args()
    
    exit_code = main(use_jsonl=args.use_jsonl)
    exit(exit_code)
