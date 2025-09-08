#!/usr/bin/env python3
"""
Visualize Bias Transformation: Compare Original vs Debiased Judge Rankings
Creates line plots with confidence intervals and critical difference plots.
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import re
from scipy import stats
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Import data loader utilities
from data_loader import get_project_paths, load_judge_data_for_visualization

# --- Model name normalization -------------------------------------------------
def normalize_model_name(name: str) -> str:
    """Normalize model names so originals and ELO outputs align.

    Rules:
    - Drop HF org prefixes like "org/model" -> "model" (keep suffix after last '/').
    - Strip whitespace; collapse multiple spaces; replace spaces with '-'.
    - Preserve case and punctuation (hyphens/underscores) to match common naming.
    """
    if not isinstance(name, str):
        return str(name)
    s = name.strip()
    if '/' in s:
        s = s.split('/')[-1]
    # Collapse whitespace and replace with '-'
    s = '-'.join(s.split())
    return s

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- ELO metric discovery -----------------------------------------------------
def discover_debiased_metrics(elo_debiased_dir: Path) -> Dict[str, Path]:
    """Discover available debiased ELO metrics by scanning filenames.

    Expected pattern: arena_hard_leaderboard_debiased_<baseline>_<metric>_elo.csv
    Returns a mapping: metric -> csv path
    """
    mapping: Dict[str, Path] = {}
    for p in elo_debiased_dir.glob('*.csv'):
        name = p.name
        if 'factor_reliability' in name:
            continue
        m = re.match(r'^arena_hard_leaderboard_debiased_.*?_(.+?)_elo\.csv$', name)
        if m:
            metric = m.group(1)
            mapping[metric] = p
    return mapping


def try_load_from_csv(judge_dir: Path, output_dir: Path) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
    """Try to load ranking data from CSV files that have proper CIs."""
    
    # Extract approach from output_dir name
    output_name = output_dir.name
    approach = None
    for strategy in ['conservative', 'rms', 'weighted', 'montecarlo', 'formatting_only']:
        if strategy in output_name:
            approach = strategy
            break
    
    if not approach:
        print("Could not determine approach from output directory name")
        return None
    
    # Look for CSV files
    original_csv_dir = judge_dir / 'tables' / 'factor_scores_original_cis'
    debiased_csv_dir = judge_dir / f'tables_debiased_{approach}' / 'tables' / 'factor_scores_updated_cis'
    
    if not original_csv_dir.exists() or not debiased_csv_dir.exists():
        print(f"CSV directories not found: {original_csv_dir} or {debiased_csv_dir}")
        return None
    
    # Load CSV files
    original_data = {}
    debiased_data = {}
    
    # Map factor names to file patterns
    factor_map = {
        'score': '_base_score_factor',
        'correctness_score': '_base_correctness_score_factor',
        'completeness_score': '_base_completeness_score_factor',
        'safety_score': '_base_safety_score_factor',
        'conciseness_score': '_base_conciseness_score_factor',
        'style_score': '_base_style_score_factor'
    }
    
    for metric, pattern in factor_map.items():
        # Find original CSV
        original_files = list(original_csv_dir.glob(f'*{pattern}*.csv'))
        if original_files:
            try:
                df = pd.read_csv(original_files[0])
                # Rename score column to match metric name
                score_col = None
                for col in df.columns:
                    if col.endswith('_score') or col == 'score':
                        score_col = col
                        break
                if score_col and score_col != 'score':
                    df = df.rename(columns={score_col: 'score'})
                
                original_data[metric] = df
                print(f"Loaded original CSV for {metric}")
            except Exception as e:
                print(f"Error loading original CSV for {metric}: {e}")
        
        # Find debiased CSV
        debiased_files = list(debiased_csv_dir.glob(f'*{pattern}*{approach}.csv'))
        if debiased_files:
            try:
                df = pd.read_csv(debiased_files[0])
                # Rename score column to match metric name
                score_col = None
                for col in df.columns:
                    if col.endswith('_score') or col == 'score':
                        score_col = col
                        break
                if score_col and score_col != 'score':
                    df = df.rename(columns={score_col: 'score'})
                
                debiased_data[metric] = df
                print(f"Loaded debiased CSV for {metric}")
            except Exception as e:
                print(f"Error loading debiased CSV for {metric}: {e}")
    
    if original_data and debiased_data:
        return {'original': original_data, 'debiased': debiased_data}
    
    return None


def load_ranking_data(rankings_input: Union[Path, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Load all ranking CSV files from a directory or use provided DataFrames."""
    
    # If already a dict of DataFrames, return it
    if isinstance(rankings_input, dict):
        return rankings_input
        
    # Otherwise, load from directory
    rankings_dir = Path(rankings_input)
    rankings: Dict[str, pd.DataFrame] = {}
    csv_files = list(rankings_dir.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {rankings_dir}")
    
    # Collect candidates per metric to pick best match after scanning all
    candidates: Dict[str, List[pd.DataFrame]] = {}
    for csv_file in csv_files:
        # Skip non-ranking artifacts (e.g., factor reliability summaries)
        if 'factor_reliability' in csv_file.name:
            continue
        # Extract metric name from filename
        # Pattern: arena_hard_leaderboard_*_base_METRIC_factor*.csv
        filename = csv_file.name
        # Try: old base_*_factor pattern
        metric = None
        match = re.search(r'base_(.+?)_factor', filename)
        if match:
            metric = match.group(1)
        # Try: ELO pattern arena_hard_leaderboard_debiased_<baseline>_<metric>_elo.csv
        if metric is None:
            m2 = re.match(r'arena_hard_leaderboard_debiased_.*_(.+)_elo\.csv', filename)
            if m2:
                metric = m2.group(1)
        # Fallback: use last token (may be 'elo')
        if metric is None:
            metric = csv_file.stem.split('_')[-1]
        
        try:
            df = pd.read_csv(csv_file)
            # Coerce common schema variants
            model_col = _pick_first_existing(df, ['model', 'model_id', 'model_name', 'Model', 'name'])
            score_col = _pick_first_existing(df, ['score', 'rating', 'rating_mean', 'rating_median', 'overall_score'])
            # If not found, try metric-specific column name (e.g., 'completeness_score')
            if not score_col and metric and metric in df.columns:
                score_col = metric
            # If still not found, try any single column that endswith '_score'
            if not score_col:
                score_cols = [c for c in df.columns if isinstance(c, str) and c.endswith('_score')]
                if len(score_cols) == 1:
                    score_col = score_cols[0]
                elif len(score_cols) > 1 and metric and f"{metric}" in score_cols:
                    score_col = metric
            if model_col and score_col:
                if model_col != 'model':
                    df = df.rename(columns={model_col: 'model'})
                if score_col != 'score':
                    df = df.rename(columns={score_col: 'score'})
                candidates.setdefault(metric, []).append(df)
            else:
                print(f"Warning: {csv_file} missing model/score columns after detection; columns present: {list(df.columns)}; skipping")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    # Choose best candidate per metric
    for metric, dfs in candidates.items():
        # Prefer DF with original_score present (paired original + debiased case)
        with_orig = [d for d in dfs if 'original_score' in d.columns]
        if with_orig:
            df = with_orig[0]
        else:
            # Otherwise choose the smallest DF (likely per-model ranking vs per-sample)
            df = sorted(dfs, key=lambda d: len(d))[0]
        rankings[metric] = df
        print(f"Loaded {len(df)} models for {metric}")
    return rankings

def parse_confidence_interval(ci_str: str) -> Tuple[float, float]:
    """Parse confidence interval string like '(-1.52, +2.04)' to (lower, upper)."""
    try:
        # Extract numbers from string like "(-1.52, +2.04)"
        import re
        matches = re.findall(r'([-+]?\d*\.?\d+)', ci_str)
        if len(matches) >= 2:
            lower = float(matches[0])
            upper = float(matches[1])
            return lower, upper
        else:
            return 0.0, 0.0
    except:
        return 0.0, 0.0

def extract_confidence_intervals(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract confidence interval bounds from CI column."""
    if 'CI' not in df.columns:
        # Fallback: use rating_q025 and rating_q975 if available
        if 'rating_q025' in df.columns and 'rating_q975' in df.columns:
            lower = df['rating_q025'].values
            upper = df['rating_q975'].values
        else:
            # No CI data available
            lower = df['score'].values
            upper = df['score'].values
        return lower, upper
    
    ci_bounds = df['CI'].apply(parse_confidence_interval)
    lower_deltas = np.array([ci[0] for ci in ci_bounds])
    upper_deltas = np.array([ci[1] for ci in ci_bounds])
    
    # Convert deltas to absolute bounds
    scores = df['score'].values
    lower = scores + lower_deltas
    upper = scores + upper_deltas
    
    return lower, upper

def create_line_plot_comparison(original_data: Dict[str, pd.DataFrame], 
                              debiased_data: Dict[str, pd.DataFrame],
                              metric: str, 
                              output_path: Path):
    """Create line plot comparing original vs debiased rankings with confidence intervals."""
    
    if metric not in original_data and metric not in debiased_data:
        print(f"Warning: {metric} not found in either dataset")
        return
    
    orig_df = original_data.get(metric, pd.DataFrame()).copy()
    debiased_df = debiased_data.get(metric, pd.DataFrame()).copy()
    
    # Fallback: reconstruct original from debiased if needed
    if ('model' not in orig_df.columns or 'score' not in orig_df.columns) and 'original_score' in debiased_df.columns:
        print("  ↪︎ Reconstructing original scores from debiased CSV (original_score)")
        orig_df = debiased_df[['model', 'original_score']].rename(columns={'original_score': 'score'})
        if 'CI' not in orig_df.columns:
            orig_df['CI'] = ['(0.00, +0.00)'] * len(orig_df)
    
    # Merge dataframes on normalized model name
    # Ensure CI column exists for merge convenience
    if 'CI' not in orig_df.columns:
        orig_df['CI'] = ['(0.00, +0.00)'] * len(orig_df)
    if 'CI' not in debiased_df.columns:
        debiased_df['CI'] = ['(0.00, +0.00)'] * len(debiased_df)

    # Build normalized keys and deduplicate by key
    orig_df = orig_df.copy()
    debiased_df = debiased_df.copy()
    orig_df['model_key'] = orig_df['model'].apply(normalize_model_name)
    debiased_df['model_key'] = debiased_df['model'].apply(normalize_model_name)
    orig_df = orig_df.drop_duplicates('model_key', keep='first')
    debiased_df = debiased_df.drop_duplicates('model_key', keep='first')

    merged = pd.merge(orig_df[['model_key', 'model', 'score', 'CI']], 
                       debiased_df[['model_key', 'model', 'score', 'CI']], 
                       on='model_key', suffixes=('_orig', '_debiased'))
    if len(merged) == 0 and 'original_score' in debiased_df.columns:
        # Rebuild original from debiased CSV
        print("  ↪︎ No overlap; rebuilding original from debiased original_score")
        orig_df = debiased_df[['model', 'original_score']].rename(columns={'original_score': 'score'}).copy()
        if 'CI' not in orig_df.columns:
            orig_df['CI'] = ['(0.00, +0.00)'] * len(orig_df)
        orig_df['model_key'] = orig_df['model'].apply(normalize_model_name)
        debiased_df['model_key'] = debiased_df['model'].apply(normalize_model_name)
        orig_df = orig_df.drop_duplicates('model_key', keep='first')
        debiased_df = debiased_df.drop_duplicates('model_key', keep='first')
        merged = pd.merge(orig_df[['model_key', 'model', 'score', 'CI']],
                          debiased_df[['model_key', 'model', 'score', 'CI']],
                          on='model_key', suffixes=('_orig', '_debiased'))
        if len(merged) == 0:
            print(f"Warning: No common models found for {metric} even after rebuild")
            return
    
    if len(merged) == 0:
        print(f"Warning: No common models found for {metric}")
        return
    
    # Sort by debiased score (descending)
    merged = merged.sort_values('score_debiased', ascending=False).reset_index(drop=True)
    
    # Extract confidence intervals
    orig_lower, orig_upper = extract_confidence_intervals(
        pd.DataFrame({
            'score': merged['score_orig'],
            'CI': merged['CI_orig']
        })
    )
    
    debiased_lower, debiased_upper = extract_confidence_intervals(
        pd.DataFrame({
            'score': merged['score_debiased'],
            'CI': merged['CI_debiased']
        })
    )
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    x_positions = np.arange(len(merged))
    
    # Plot lines with confidence intervals
    plt.fill_between(x_positions, orig_lower, orig_upper, alpha=0.2, color='red', label='Original 95% CI')
    plt.plot(x_positions, merged['score_orig'], 'o-', color='red', linewidth=2, markersize=6, label='Original Ranking')
    
    plt.fill_between(x_positions, debiased_lower, debiased_upper, alpha=0.2, color='blue', label='Debiased 95% CI')
    plt.plot(x_positions, merged['score_debiased'], 's-', color='blue', linewidth=2, markersize=6, label='Debiased Ranking')
    
    # Customize plot
    plt.xlabel('Models (Ranked by Debiased Score)', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title(f'Ranking Transformation: {metric.replace("_", " ").title()}\n'
              f'Original vs Debiased Judge Evaluations', fontsize=14, fontweight='bold')
    
    # Set x-axis labels - use debiased model names; strip \n and replace '/' with '_'
    model_labels = []
    name_series = merged['model_debiased'] if 'model_debiased' in merged.columns else (
        merged['model'] if 'model' in merged.columns else pd.Series([''] * len(merged))
    )
    for name in name_series:
        # Only strip \n characters and replace / with _
        clean_name = name.replace('\\n', '').replace('\n', '').replace('/', '_')
        model_labels.append(clean_name)
    
    plt.xticks(x_positions, model_labels, rotation=45, ha='right', fontsize=9)
    
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add statistics text box
    correlation = np.corrcoef(merged['score_orig'], merged['score_debiased'])[0, 1]
    orig_std = merged['score_orig'].std()
    debiased_std = merged['score_debiased'].std()
    
    stats_text = f'Correlation: {correlation:.3f}\n'
    stats_text += f'Original Std: {orig_std:.2f}\n'
    stats_text += f'Debiased Std: {debiased_std:.2f}\n'
    stats_text += f'Variance Reduction: {(1 - debiased_std/orig_std)*100:.1f}%'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top', fontsize=9)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved line plot: {output_path}")

def create_critical_difference_plot(original_data: Dict[str, pd.DataFrame], 
                                   debiased_data: Dict[str, pd.DataFrame],
                                   metric: str, 
                                   output_path: Path):
    """Create critical difference plot showing ranking changes."""
    
    if metric not in original_data or metric not in debiased_data:
        print(f"Warning: {metric} not found in both datasets")
        return
    
    orig_df = original_data.get(metric, pd.DataFrame()).copy()
    debiased_df = debiased_data.get(metric, pd.DataFrame()).copy()
    
    # Add rank information
    orig_df['orig_rank'] = orig_df['score'].rank(method='dense', ascending=False)
    debiased_df['debiased_rank'] = debiased_df['score'].rank(method='dense', ascending=False)
    
    # Normalize names and merge on key
    orig_df = orig_df.copy(); debiased_df = debiased_df.copy()
    orig_df['model_key'] = orig_df['model'].apply(normalize_model_name)
    debiased_df['model_key'] = debiased_df['model'].apply(normalize_model_name)
    orig_df = orig_df.drop_duplicates('model_key', keep='first')
    debiased_df = debiased_df.drop_duplicates('model_key', keep='first')
    merged = pd.merge(orig_df[['model_key', 'model', 'score', 'orig_rank']], 
                      debiased_df[['model_key', 'model', 'score', 'debiased_rank']], 
                      on='model_key', suffixes=('_orig', '_debiased'))
    if len(merged) == 0 and 'original_score' in debiased_df.columns:
        print("  ↪︎ No overlap; rebuilding original from debiased original_score (critical diff)")
        orig_df = debiased_df[['model', 'original_score']].rename(columns={'original_score': 'score'})
        orig_df['orig_rank'] = orig_df['score'].rank(method='dense', ascending=False)
        orig_df['model_key'] = orig_df['model'].apply(normalize_model_name)
        debiased_df['model_key'] = debiased_df['model'].apply(normalize_model_name)
        orig_df = orig_df.drop_duplicates('model_key', keep='first')
        debiased_df = debiased_df.drop_duplicates('model_key', keep='first')
        merged = pd.merge(orig_df[['model_key', 'model', 'score', 'orig_rank']],
                          debiased_df[['model_key', 'model', 'score', 'debiased_rank']],
                          on='model_key', suffixes=('_orig', '_debiased'))
        if len(merged) == 0:
            print(f"Warning: No common models found for {metric} even after rebuild")
            return
    
    # Calculate rank changes
    merged['rank_change'] = merged['orig_rank'] - merged['debiased_rank']
    merged['abs_rank_change'] = abs(merged['rank_change'])
    
    # Sort by debiased rank
    merged = merged.sort_values('debiased_rank').reset_index(drop=True)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Rank positions comparison
    x_positions = np.arange(len(merged))
    
    # Color code by rank change magnitude
    colors = plt.cm.RdYlBu_r(merged['abs_rank_change'] / merged['abs_rank_change'].max())
    
    for i, (_, row) in enumerate(merged.iterrows()):
        ax1.plot([i, i], [row['orig_rank'], row['debiased_rank']], 
                'o-', color=colors[i], linewidth=3, markersize=8, alpha=0.8)
        
        # Add arrows for large rank changes
        if abs(row['rank_change']) >= 2:
            ax1.annotate('', xy=(i, row['debiased_rank']), xytext=(i, row['orig_rank']),
                        arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))
    
    ax1.set_xlabel('Models (Ordered by Debiased Rank)', fontweight='bold')
    ax1.set_ylabel('Rank Position', fontweight='bold')
    ax1.set_title(f'Rank Position Changes: {metric.replace("_", " ").title()}', fontweight='bold')
    ax1.invert_yaxis()  # Lower rank numbers at top
    ax1.grid(True, alpha=0.3)
    
    # Clean up model labels - use debiased model names; strip \n and replace '/' with '_'
    model_labels = []
    name_series = merged['model_debiased'] if 'model_debiased' in merged.columns else (
        merged['model'] if 'model' in merged.columns else pd.Series([''] * len(merged))
    )
    for name in name_series:
        # Only strip \n characters and replace / with _
        clean_name = name.replace('\\n', '').replace('\n', '').replace('/', '_')
        model_labels.append(clean_name)
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=8)
    
    # Add legend for original vs debiased
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='gray', linestyle='-', 
                             markersize=8, label='Rank Change')]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Rank change magnitude heatmap
    rank_changes = merged['rank_change'].values.reshape(1, -1)
    
    im = ax2.imshow(rank_changes, cmap='RdBu_r', aspect='auto', 
                    vmin=-max(abs(merged['rank_change'])), 
                    vmax=max(abs(merged['rank_change'])))
    
    ax2.set_xlabel('Models', fontweight='bold')
    ax2.set_title(f'Rank Change Magnitude: {metric.replace("_", " ").title()}', fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xticks(range(len(merged)))
    ax2.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1)
    cbar.set_label('Rank Change (Original - Debiased)', fontweight='bold')
    
    # Add text annotations for significant changes
    for i, change in enumerate(merged['rank_change']):
        if abs(change) >= 2:
            ax2.text(i, 0, f'{change:+.0f}', ha='center', va='center', 
                    fontweight='bold', color='white' if abs(change) > 3 else 'black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved critical difference plot: {output_path}")

def create_summary_comparison(original_data: Dict[str, pd.DataFrame], 
                            debiased_data: Dict[str, pd.DataFrame],
                            output_path: Path):
    """Create summary comparison across all metrics."""
    
    common_metrics = set(original_data.keys()) & set(debiased_data.keys())
    if not common_metrics:
        print("Warning: No common metrics found")
        return
    
    # Collect statistics for all metrics
    stats_data = []
    
    for metric in sorted(common_metrics):
        orig_df = original_data[metric]
        debiased_df = debiased_data[metric]
        
        # Normalize names and merge on key
        orig_df = orig_df.copy(); debiased_df = debiased_df.copy()
        orig_df['model_key'] = orig_df['model'].apply(normalize_model_name)
        debiased_df['model_key'] = debiased_df['model'].apply(normalize_model_name)
        orig_df = orig_df.drop_duplicates('model_key', keep='first')
        debiased_df = debiased_df.drop_duplicates('model_key', keep='first')
        merged = pd.merge(orig_df[['model_key', 'score']], 
                         debiased_df[['model_key', 'score']], 
                         on='model_key', suffixes=('_orig', '_debiased'))
        
        if len(merged) > 0:
            correlation = np.corrcoef(merged['score_orig'], merged['score_debiased'])[0, 1]
            orig_range = merged['score_orig'].max() - merged['score_orig'].min()
            debiased_range = merged['score_debiased'].max() - merged['score_debiased'].min()
            variance_reduction = (1 - merged['score_debiased'].std() / merged['score_orig'].std()) * 100
            
            # Calculate rank correlation (Spearman)
            orig_ranks = merged['score_orig'].rank(ascending=False)
            debiased_ranks = merged['score_debiased'].rank(ascending=False)
            rank_correlation = stats.spearmanr(orig_ranks, debiased_ranks)[0]
            
            stats_data.append({
                'metric': metric.replace('_', ' ').title(),
                'score_correlation': correlation,
                'rank_correlation': rank_correlation,
                'orig_range': orig_range,
                'debiased_range': debiased_range,
                'range_reduction': (1 - debiased_range/orig_range) * 100,
                'variance_reduction': variance_reduction
            })
    
    # Create summary plot
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score correlation
        bars1 = ax1.bar(stats_df['metric'], stats_df['score_correlation'], 
                       color='steelblue', alpha=0.7)
        ax1.set_title('Score Correlation (Original vs Debiased)', fontweight='bold')
        ax1.set_ylabel('Pearson Correlation')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars1, stats_df['score_correlation']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Rank correlation
        bars2 = ax2.bar(stats_df['metric'], stats_df['rank_correlation'], 
                       color='darkgreen', alpha=0.7)
        ax2.set_title('Rank Correlation (Spearman)', fontweight='bold')
        ax2.set_ylabel('Spearman Correlation')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars2, stats_df['rank_correlation']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Range reduction
        bars3 = ax3.bar(stats_df['metric'], stats_df['range_reduction'], 
                       color='orange', alpha=0.7)
        ax3.set_title('Score Range Reduction', fontweight='bold')
        ax3.set_ylabel('Range Reduction (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars3, stats_df['range_reduction']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Variance reduction
        bars4 = ax4.bar(stats_df['metric'], stats_df['variance_reduction'], 
                       color='red', alpha=0.7)
        ax4.set_title('Score Variance Reduction', fontweight='bold')
        ax4.set_ylabel('Variance Reduction (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars4, stats_df['variance_reduction']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved summary comparison: {output_path}")

def visualize_from_jsonl(judge_dir: Union[str, Path], output_dir: Union[str, Path], 
                        metrics: Optional[List[str]] = None):
    """Create visualizations directly from JSONL data."""
    judge_dir = Path(judge_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {judge_dir}...")
    
    # Load judge data from JSONL files
    judge_data = load_judge_data_for_visualization(judge_dir)
    
    if judge_data['original'].empty or judge_data['debiased'].empty:
        print("Error: Missing original or debiased data")
        return 1
    
    # Create data dictionaries for visualization functions
    original_data = {'score': judge_data['original']}
    debiased_data = {'score': judge_data['debiased']}
    
    # Add factor-specific dataframes if available
    agg_df = judge_data.get('aggregated', pd.DataFrame())
    if not agg_df.empty:
        factor_cols = ['correctness_score', 'completeness_score', 'safety_score', 
                      'conciseness_score', 'style_score']
        
        for factor in factor_cols:
            if factor in agg_df.columns:
                # Create factor-specific rankings, preferring per-factor debiased scores when present
                from data_loader import create_ranking_dataframe
                original_data[factor] = create_ranking_dataframe(agg_df, factor, 'original')
                factor_base = factor.replace('_score', '')
                debiased_col = f'score_debiased_{factor_base}' if f'score_debiased_{factor_base}' in agg_df.columns else 'score_debiased'
                deb_df = create_ranking_dataframe(agg_df, debiased_col, 'debiased')
                # Prefer original factor CI bounds for visualization parity
                ci_lower_col = f'{factor}_CI_lower'
                ci_upper_col = f'{factor}_CI_upper'
                if ci_lower_col in agg_df.columns and ci_upper_col in agg_df.columns:
                    lower_map = dict(zip(agg_df['model'], agg_df[ci_lower_col]))
                    upper_map = dict(zip(agg_df['model'], agg_df[ci_upper_col]))
                    deb_df['rating_q025'] = deb_df['model'].map(lower_map)
                    deb_df['rating_q975'] = deb_df['model'].map(upper_map)
                    deb_df['CI'] = deb_df.apply(
                        lambda r: f"({(r['rating_q025'] - r['score']):.2f}, +{(r['rating_q975'] - r['score']):.2f})",
                        axis=1
                    )
                # Ensure original scores for comparison reflect the factor
                if 'original_score' not in deb_df.columns:
                    deb_df['original_score'] = agg_df.loc[deb_df.index, factor]
                debiased_data[factor] = deb_df
    
    # Process metrics
    if metrics:
        metrics_to_process = set(metrics) & set(original_data.keys())
    else:
        metrics_to_process = set(original_data.keys()) & set(debiased_data.keys())
    
    print(f"Processing metrics: {sorted(metrics_to_process)}")
    
    # Create visualizations
    for metric in sorted(metrics_to_process):
        print(f"\nCreating plots for {metric}...")
        
        # Line plot with confidence intervals
        line_plot_path = output_dir / f"line_comparison_{metric}.png"
        create_line_plot_comparison(original_data, debiased_data, metric, line_plot_path)
        
        # Critical difference plot
        cd_plot_path = output_dir / f"critical_difference_{metric}.png"
        create_critical_difference_plot(original_data, debiased_data, metric, cd_plot_path)
    
    # Create summary comparison
    print("\nCreating summary comparison...")
    summary_path = output_dir / "summary_comparison.png"
    create_summary_comparison(original_data, debiased_data, summary_path)
    
    print(f"\nAll plots saved to: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Visualize transformation from biased to debiased judge rankings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From CSV directories (legacy):
  python visualize_bias_transformation.py \\
    --original "/path/to/original/factor_scores_updated_cis" \\
    --debiased "/path/to/debiased/tables/factor_scores_updated_cis"
    
  # From JSONL data (new):
  python visualize_bias_transformation.py \\
    --judge-dir "/path/to/judge/directory" \\
    --output-dir "./visualizations"
        """
    )
    
    # Legacy arguments
    parser.add_argument('--original', type=str,
                       help='Path to original rankings directory (CSV mode)')
    parser.add_argument('--debiased', type=str, 
                       help='Path to debiased rankings directory (CSV mode)')
    # New: allow consuming ELO bootstrap CSVs directly
    parser.add_argument('--elo-debiased-dir', type=str,
                       help='Path to ELO-bootstrapped debiased CSVs (overrides JSONL)')
    parser.add_argument('--elo-original-dir', type=str,
                       help='Path to ELO-bootstrapped original CSVs (optional)')
    
    # New arguments for JSONL mode
    parser.add_argument('--judge-dir', type=str,
                       help='Path to judge directory with base_processed and base_debiased (JSONL mode)')
    
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for plots')
    parser.add_argument('--metrics', type=str, nargs='*',
                       help='Specific metrics to plot (default: all available)')
    
    args = parser.parse_args()
    
    # Determine mode and validate arguments
    if args.elo_debiased_dir or args.elo_original_dir:
        # ELO CSV mode: load ranking CSVs directly
        if not args.output_dir:
            paths = get_project_paths()
            args.output_dir = paths['figures'] / 'bias_transformation'
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CSVs
        original_dir = Path(args.elo_original_dir) if args.elo_original_dir else None
        debiased_dir = Path(args.elo_debiased_dir) if args.elo_debiased_dir else None
        if debiased_dir is None or not debiased_dir.exists():
            print('Error: --elo-debiased-dir is required and must exist')
            sys.exit(1)
        # Use local loader to parse ranking CSVs
        debiased_data = load_ranking_data(debiased_dir)
        original_data = load_ranking_data(original_dir) if (original_dir and original_dir.exists()) else {}
        # Discover debiased metrics from filenames to drive plotting
        metric_file_map = discover_debiased_metrics(debiased_dir)
        available = sorted(metric_file_map.keys())
        if args.metrics:
            requested = list(args.metrics)
            metrics_to_process = [m for m in requested if m in metric_file_map]
        else:
            metrics_to_process = available
        print(f"Debiased metrics available: {available}")
        if args.metrics:
            print(f"Requested metrics: {list(args.metrics)}")
        print(f"Processing metrics: {metrics_to_process}")
        # Ensure debiased_data has frames for each selected metric (lazy load if missing)
        for metric in metrics_to_process:
            if metric not in debiased_data:
                try:
                    df = pd.read_csv(metric_file_map[metric])
                    model_col = _pick_first_existing(df, ['model', 'model_id', 'model_name', 'Model', 'name']) or 'model'
                    if model_col != 'model':
                        df = df.rename(columns={model_col: 'model'})
                    # Rename metric-specific score to 'score' if necessary
                    if metric in df.columns and 'score' not in df.columns:
                        df = df.rename(columns={metric: 'score'})
                    debiased_data[metric] = df
                except Exception as e:
                    print(f"Warning: Failed to load debiased metric {metric} from {metric_file_map.get(metric)}: {e}")
        for metric in metrics_to_process:
            print(f"\nCreating plots for {metric}...")
            line_plot_path = output_dir / f"line_comparison_{metric}.png"
            create_line_plot_comparison(original_data, debiased_data, metric, line_plot_path)
            cd_plot_path = output_dir / f"critical_difference_{metric}.png"
            create_critical_difference_plot(original_data, debiased_data, metric, cd_plot_path)
        print(f"\nAll plots saved to: {output_dir}")
        return 0
    
    if args.judge_dir:
        # JSONL mode
        if not args.output_dir:
            paths = get_project_paths()
            args.output_dir = paths['figures'] / 'bias_transformation'
        
        return visualize_from_jsonl(args.judge_dir, args.output_dir, args.metrics)
    
    elif args.original and args.debiased:
        # Legacy CSV mode
        if not args.output_dir:
            args.output_dir = '.'
    
        # Convert paths
        original_path = Path(args.original)
        debiased_path = Path(args.debiased)
        output_dir = Path(args.output_dir)
        
        # Validate paths
        if not original_path.exists():
            print(f"Error: Original path does not exist: {original_path}")
            sys.exit(1)
            
        if not debiased_path.exists():
            print(f"Error: Debiased path does not exist: {debiased_path}")
            sys.exit(1)
    else:
        print("Error: Please specify either --judge-dir (for JSONL) or both --original and --debiased (for CSV)")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading original rankings...")
    original_data = load_ranking_data(original_path)
    
    print("Loading debiased rankings...")  
    debiased_data = load_ranking_data(debiased_path)
    
    if not original_data:
        print("Error: No original ranking data loaded")
        sys.exit(1)
        
    if not debiased_data:
        print("Error: No debiased ranking data loaded")
        sys.exit(1)
    
    # Determine metrics to process
    common_metrics = set(original_data.keys()) & set(debiased_data.keys())
    if args.metrics:
        metrics_to_process = set(args.metrics) & common_metrics
        if not metrics_to_process:
            print(f"Warning: None of the specified metrics {args.metrics} found in data")
            metrics_to_process = common_metrics
    else:
        metrics_to_process = common_metrics
    
    print(f"Processing metrics: {sorted(metrics_to_process)}")
    
    # Create visualizations
    for metric in sorted(metrics_to_process):
        print(f"\\nCreating plots for {metric}...")
        
        # Line plot with confidence intervals
        line_plot_path = output_dir / f"line_comparison_{metric}.png"
        create_line_plot_comparison(original_data, debiased_data, metric, line_plot_path)
        
        # Critical difference plot
        cd_plot_path = output_dir / f"critical_difference_{metric}.png"
        create_critical_difference_plot(original_data, debiased_data, metric, cd_plot_path)
    
    # Create summary comparison
    print("\\nCreating summary comparison...")
    summary_path = output_dir / "summary_comparison.png"
    create_summary_comparison(original_data, debiased_data, summary_path)
    
    print(f"\\nAll plots saved to: {output_dir}")
    print("\\nPlots created:")
    for metric in sorted(metrics_to_process):
        print(f"  - Line plot: line_comparison_{metric}.png")
        print(f"  - Critical difference: critical_difference_{metric}.png")
    print(f"  - Summary: summary_comparison.png")

if __name__ == "__main__":
    main()
