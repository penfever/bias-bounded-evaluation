#!/usr/bin/env python3
"""
Generate ranking CSV files for Combined A-BB strategies to use with visualize_bias_transformation.py

This script creates the ranking CSV files needed to generate the original-style comparison 
visualizations (line plots, critical difference plots, summary plots) for the Combined A-BB results.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import the new data loader
from data_loader import (
    get_project_paths, 
    load_judge_data_for_visualization,
    create_ranking_dataframe,
    aggregate_scores_by_model,
    detect_baseline_model
)

# Import arena hard utilities for score conversion
sys.path.append(str(Path(__file__).parent.parent.parent / 'differential_debiasing' / 'interfaces'))
from arena_hard_utils import bootstrap_to_win_rate_ci


def _copy_elo_csvs(judge_dir: Path, strategy: str, baseline: str) -> bool:
    """Copy ELO CSVs (if present) into the expected factor_scores_updated_cis/ directory with naming compatible with visualizer."""
    elo_dir = judge_dir / f"tables_debiased_{strategy}" / "tables" / "factor_scores_updated_cis_elo"
    if not elo_dir.exists():
        return False
    target_dir = judge_dir / f"tables_debiased_{strategy}" / "tables" / "factor_scores_updated_cis"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for csv in elo_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv)
            # Derive metric name from columns: pick first *_score column or 'score'
            metric_col = None
            for c in df.columns:
                if c.endswith('_score') or c == 'score':
                    metric_col = c
                    break
            metric_name = (metric_col or 'score').replace('_score','')
            # Build filename compatible with loader's regex
            out_name = f"arena_hard_leaderboard_elo_{judge_dir.name}_judge_{baseline}_base_{metric_name}_score_factor_{strategy}.csv"
            df.to_csv(target_dir / out_name, index=False)
            copied += 1
        except Exception:
            continue
    return copied > 0

def load_combined_abb_results(results_file: Path) -> Dict:
    """Load Combined A-BB analysis results."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_ranking_data(original_scores: List[float], debiased_scores: List[float], 
                       models: List[str] = None, approach: str = "approach") -> pd.DataFrame:
    """Create ranking CSV data in the format expected by visualize_bias_transformation.py."""
    
    if models is None:
        models = [f"model_{i+1}" for i in range(len(original_scores))]
    
    # Ensure we have the same number of models and scores
    n_samples = min(len(original_scores), len(debiased_scores))
    if models and len(models) != n_samples:
        models = models[:n_samples]
    
    # Create DataFrame with the expected format
    df = pd.DataFrame({
        'model': models[:n_samples],
        'score': debiased_scores[:n_samples],  # Use debiased scores as the main ranking
        'CI': ['(0.00, +0.00)'] * n_samples,  # Placeholder CIs
        'approach': [approach] * n_samples,
        'original_score': original_scores[:n_samples]  # Keep original for comparison
    })
    
    # Sort by debiased score (descending)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return df

def generate_standard_models_list(n_models: int) -> List[str]:
    """Generate a standard list of model names for consistency."""
    base_models = [
        "gpt-4",
        "claude-3.5-sonnet", 
        "gemini-1.5-pro",
        "llama-3-70b-instruct",
        "qwen-plus",
        "mistral-large",
        "cohere-command-r-plus",
        "phi-4",
        "deepseek-r1",
        "meta-llama-3-8b-instruct",
        "gpt-3.5-turbo",
        "claude-3-haiku",
        "gemma-7b-it", 
        "llama-3-8b-instruct",
        "mistral-7b-instruct",
        "qwen-2.5-7b-instruct",
        "opt-125m",
        "pythia-160m",
        "tiny-random-model-1",
        "tiny-random-model-2"
    ]
    
    # Cycle through base models if we need more
    models = []
    for i in range(n_models):
        if i < len(base_models):
            models.append(base_models[i])
        else:
            models.append(f"{base_models[i % len(base_models)]}-variant-{i // len(base_models)}")
    
    return models

def save_rankings_for_judge_from_jsonl(judge_name: str, output_base_path: Path, strategies: List[str] = None):
    """Save ranking CSV files for a specific judge using JSONL data."""
    
    print(f"\nProcessing rankings for {judge_name}...")
    
    # Default strategies if not provided
    if strategies is None:
        strategies = ['conservative', 'rms', 'formatting_only']
    
    # Load judge data from JSONL files
    judge_dir = output_base_path / judge_name
    if not judge_dir.exists():
        print(f"Judge directory not found: {judge_dir}")
        return
        
    # Determine baseline model dynamically
    baseline_model = detect_baseline_model(judge_dir) or 'gpt-4-0314'
    
    # Load the data
    try:
        # Use loader's conversion to win-rate and CI; do not reconvert here
        judge_data = load_judge_data_for_visualization(judge_dir)
    except Exception as e:
        print(f"Error loading judge data: {e}")
        return
    
    if not judge_data['debiased'].empty:
        # We have debiased data - create ranking files for each strategy
        for strategy in strategies:
            clean_approach = strategy
            print(f"  Processing {clean_approach} strategy...")
            
            # Create directory structure
            approach_dir = judge_dir / f"tables_debiased_{clean_approach}"
            ranking_dir = approach_dir / "tables" / "factor_scores_updated_cis"
            ranking_dir.mkdir(parents=True, exist_ok=True)

            # Require ELO CSVs for this strategy; no fallback to aggregate path
            if _copy_elo_csvs(judge_dir, clean_approach, baseline_model):
                print(f"    Found ELO bootstrap CSVs. Copied into factor_scores_updated_cis/")
                continue
            else:
                raise RuntimeError(f"No ELO CSVs found for strategy '{clean_approach}' under {judge_dir}. Run generate_debiased_rankings_elo.py first.")

            # Get aggregated data
            agg_df = judge_data['aggregated']
            
            # Create ranking DataFrames for each score type
            score_types = {
                'score': 'overall_score',
                'correctness_score': 'correctness_score',
                'completeness_score': 'completeness_score',
                'safety_score': 'safety_score',
                'conciseness_score': 'conciseness_score',
                'style_score': 'style_score'
            }
            
            for score_name, col_name in score_types.items():
                # Prepare debiased score column: prefer per-factor debiased when available
                if col_name in agg_df.columns:
                    if score_name != 'score':
                        factor_base = score_name.replace('_score', '')
                        debiased_col = f'score_debiased_{factor_base}' if f'score_debiased_{factor_base}' in agg_df.columns else 'score_debiased'
                    else:
                        debiased_col = 'score_debiased' if 'score_debiased' in agg_df.columns else None
                else:
                    debiased_col = None

                if debiased_col is not None and debiased_col in agg_df.columns:
                    # Create ranking with chosen debiased scores
                    ranking_df = create_ranking_dataframe(agg_df, debiased_col, clean_approach)
                    
                    # Ensure original scores for comparison reflect the factor
                    if col_name in agg_df.columns and 'original_score' not in ranking_df.columns:
                        ranking_df['original_score'] = agg_df.loc[ranking_df.index, col_name]
                    
                    # Update confidence intervals using precomputed numeric bounds from aggregated_df (already win-rate)
                    # Prefer original factor CI bounds to match Arena-Hard widths
                    orig_ci_lower_col = f'{col_name}_CI_lower'
                    orig_ci_upper_col = f'{col_name}_CI_upper'
                    deb_ci_lower_col = f'{debiased_col}_CI_lower'
                    deb_ci_upper_col = f'{debiased_col}_CI_upper'
                    if orig_ci_lower_col in agg_df.columns and orig_ci_upper_col in agg_df.columns:
                        lower_map = dict(zip(agg_df['model'], agg_df[orig_ci_lower_col]))
                        upper_map = dict(zip(agg_df['model'], agg_df[orig_ci_upper_col]))
                    elif deb_ci_lower_col in agg_df.columns and deb_ci_upper_col in agg_df.columns:
                        lower_map = dict(zip(agg_df['model'], agg_df[deb_ci_lower_col]))
                        upper_map = dict(zip(agg_df['model'], agg_df[deb_ci_upper_col]))
                    else:
                        lower_map = upper_map = None
                    if lower_map is not None:
                        ranking_df['rating_q025'] = ranking_df['model'].map(lower_map)
                        ranking_df['rating_q975'] = ranking_df['model'].map(upper_map)
                        # Build CI deltas from center score
                        ranking_df['CI'] = ranking_df.apply(
                            lambda r: f"({(r['rating_q025'] - r['score']):.2f}, +{(r['rating_q975'] - r['score']):.2f})",
                            axis=1
                        )
                    else:
                        ranking_df['rating_q025'] = ranking_df['score']
                        ranking_df['rating_q975'] = ranking_df['score']
                        ranking_df['CI'] = "(-0.00, +0.00)"
                    ranking_df['avg_tokens'] = 0.0  # Placeholder
                    ranking_df['date'] = '2025-01-09'
                    
                    # Reorder columns to match expected format
                    col_order = ['model', 'score', 'rating_q025', 'rating_q975', 'CI', 'avg_tokens', 'date']
                    # Add any additional columns that exist
                    for col in ranking_df.columns:
                        if col not in col_order:
                            col_order.append(col)
                    
                    # Rename score column to match the factor name
                    score_col_name = score_name.replace('_score', '') + '_score' if score_name != 'score' else 'score'
                    ranking_df = ranking_df.rename(columns={'score': score_col_name})
                    col_order[1] = score_col_name
                    
                    ranking_df = ranking_df[col_order]
                    
                    # Generate filename
                    factor_part = f"_{score_name.replace('_score', '')}" if score_name != 'score' else ""
                    filename = f"arena_hard_leaderboard_20250119_{judge_name}_judge_gpt-4-0314_base{factor_part}_score_factor_{clean_approach}.csv"
                    output_path = ranking_dir / filename
                    
                    ranking_df.to_csv(output_path, index=False)
                    print(f"    Saved: {output_path.name}")
    else:
        print(f"  No debiased data found for {judge_name}")


def save_rankings_for_judge(judge_name: str, judge_results: Dict, output_base_path: Path):
    """Save ranking CSV files for a specific judge's Combined A-BB results (legacy support)."""
    
    # This function is kept for backward compatibility but now uses the JSONL data
    # Instead of using the results dict, we load from JSONL files
    save_rankings_for_judge_from_jsonl(judge_name, output_base_path)

def main():
    """Main function to generate ranking CSV files for Combined A-BB strategies."""
    
    print("🚀 Generating Combined A-BB Ranking CSV Files")
    print("=" * 60)
    
    # Get project paths
    paths = get_project_paths()
    data_base_path = paths['data_base']
    
    # Find judge directories with debiased data
    judge_patterns = [
        "QwQ-32B-setting1",
        "DeepSeek-R1-32B-setting1", 
        "DeepSeek-R1-32B-setting2",
        "DeepSeek-R1-32B-setting3",
        "GPT-3.5-Turbo-0125-setting1",
        "GPT-4o-mini-0718-setting1"
    ]
    
    found_judges = []
    for judge_name in judge_patterns:
        judge_dir = data_base_path / judge_name
        if judge_dir.exists():
            # Check for any base_debiased* directories
            debiased_dirs = list(judge_dir.glob('base_debiased*'))
            if debiased_dirs:
                found_judges.append(judge_name)
                print(f"✅ Found judge with debiased data: {judge_name}")
    
    if not found_judges:
        print("❌ No judges with debiased data found!")
        return 1
    
    print(f"\n📊 Found {len(found_judges)} judges to process")
    
    # Process each judge
    for judge_name in found_judges:
        try:
            save_rankings_for_judge_from_jsonl(judge_name, data_base_path)
            print(f"✅ Completed rankings for {judge_name}")
            
        except Exception as e:
            print(f"❌ Error processing {judge_name}: {e}")
    
    print("\n🎯 Summary:")
    print("=" * 60)
    
    # Count generated files
    total_files = 0
    strategies = ['conservative', 'rms', 'formatting_only']
    for judge_name in found_judges:
        for strategy in strategies:
            ranking_dir = data_base_path / judge_name / f"tables_debiased_{strategy}" / "tables" / "factor_scores_updated_cis"
            if ranking_dir.exists():
                files = list(ranking_dir.glob("*.csv"))
                total_files += len(files)
                print(f"{judge_name} - {strategy}: {len(files)} CSV files")
    
    print(f"\nTotal ranking CSV files generated: {total_files}")
    print("\n📁 Files saved in structure:")
    print("  {judge}/tables_debiased_{strategy}/tables/factor_scores_updated_cis/")
    print("  Ready for use with visualize_bias_transformation.py!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
