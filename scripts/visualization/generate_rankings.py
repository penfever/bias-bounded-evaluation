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
    aggregate_scores_by_model
)

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
        strategies = ['conservative', 'rms', 'weighted', 'montecarlo', 'formatting_only']
    
    # Load judge data from JSONL files
    judge_dir = output_base_path / judge_name
    if not judge_dir.exists():
        print(f"Judge directory not found: {judge_dir}")
        return
        
    # Load the data
    try:
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
                # Use debiased scores
                if col_name in agg_df.columns and 'score_debiased' in agg_df.columns:
                    # Create ranking with debiased scores
                    ranking_df = create_ranking_dataframe(agg_df, 'score_debiased', clean_approach)
                    
                    # Add original scores for comparison
                    if col_name in agg_df.columns:
                        ranking_df['original_score'] = agg_df.loc[ranking_df.index, col_name]
                    
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
    strategies = ['conservative', 'rms', 'weighted', 'montecarlo', 'formatting_only']
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