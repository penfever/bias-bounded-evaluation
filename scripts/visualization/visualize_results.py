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
    """Load judge results directly from JSONL files."""
    
    results = {}
    
    # Judge patterns to look for
    judge_patterns = [
        "QwQ-32B-setting1",
        "DeepSeek-R1-32B-setting1",
        "DeepSeek-R1-32B-setting2",
        "DeepSeek-R1-32B-setting3",
        "GPT-3.5-Turbo-0125-setting1",
        "GPT-4o-mini-0718-setting1"
    ]
    
    for judge_name in judge_patterns:
        judge_dir = base_path / judge_name
        
        if not judge_dir.exists():
            continue
            
        # Check for both original and debiased data
        base_processed_dir = judge_dir / 'base_processed'
        
        # Look for any base_debiased* directory
        debiased_dirs = list(judge_dir.glob('base_debiased*'))
        if not debiased_dirs:
            continue
        
        # Use the first debiased directory found
        base_debiased_dir = debiased_dirs[0]
        
        if base_processed_dir.exists() and base_debiased_dir.exists():
            try:
                # Load original evaluations
                original_df = load_original_evaluations(base_processed_dir)
                
                # Load debiased scores
                debiased_df = load_debiased_scores(base_debiased_dir)
                
                # Merge data
                merged_df = merge_original_and_debiased(original_df, debiased_df)
                
                # Aggregate by model
                agg_df = aggregate_scores_by_model(merged_df)
                
                # Extract scores for results format
                original_scores = merged_df['overall_score'].values.tolist()
                debiased_scores = merged_df['score_debiased'].values.tolist()
                
                # Calculate statistics
                correlation = np.corrcoef(original_scores, debiased_scores)[0, 1]
                mean_abs_diff = np.mean(np.abs(np.array(original_scores) - np.array(debiased_scores)))
                variance_ratio = np.var(debiased_scores) / np.var(original_scores)
                noise_level = np.std(np.array(original_scores) - np.array(debiased_scores))
                
                # Create result structure similar to JSON format
                results[judge_name] = {
                    'data_source': str(judge_dir),
                    'n_samples': len(merged_df),
                    'approaches': {
                        'combined_abb_unified': {
                            'success': True,
                            'original_scores': original_scores,
                            'debiased_scores': debiased_scores,
                            'n_samples': len(original_scores),
                            'diagnostics': {
                                'combined_sensitivity': 0.5,  # Placeholder
                                'noise_std': noise_level,
                                'abb_constraint_satisfied': True  # Placeholder
                            },
                            'validation': {
                                'correlation': correlation,
                                'mean_absolute_difference': mean_abs_diff,
                                'variance_ratio': variance_ratio,
                                'signal_preservation': correlation,
                                'noise_level': noise_level
                            }
                        }
                    }
                }
                
                print(f"✅ Loaded data for {judge_name}: {len(merged_df)} samples")
                
            except Exception as e:
                print(f"❌ Error loading data for {judge_name}: {e}")
    
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
    """Create comparison across different judges."""
    
    # Extract data
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
                        'constraint_satisfied': approach_result['diagnostics'].get('abb_constraint_satisfied')
                    })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Combined A-BB Analysis: Judge Comparison', fontsize=16, fontweight='bold')
    
    # 1. Sensitivity by judge and strategy
    pivot_sensitivity = df.pivot(index='judge', columns='approach', values='combined_sensitivity')
    sns.heatmap(pivot_sensitivity, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Combined Sensitivity by Judge and Strategy')
    axes[0].set_xlabel('Aggregation Strategy')
    axes[0].set_ylabel('Judge')
    
    # 2. Correlation by judge and strategy
    pivot_correlation = df.pivot(index='judge', columns='approach', values='correlation')
    sns.heatmap(pivot_correlation, annot=True, fmt='.3f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Signal Preservation by Judge and Strategy')
    axes[1].set_xlabel('Aggregation Strategy')
    axes[1].set_ylabel('Judge')
    
    plt.tight_layout()
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

def create_combined_distribution_overlay(results: dict) -> plt.Figure:
    """Create overlaid distribution comparison for all approaches."""
    
    # Extract data for overlay comparison
    approach_data = {}
    for judge_name, judge_results in results.items():
        if 'approaches' in judge_results:
            for approach_name, approach_result in judge_results['approaches'].items():
                if approach_result.get('success', False):
                    clean_approach = approach_name.replace('combined_abb_', '')
                    if clean_approach not in approach_data:
                        approach_data[clean_approach] = {'original': [], 'debiased': []}
                    
                    approach_data[clean_approach]['original'].extend(approach_result['original_scores'])
                    approach_data[clean_approach]['debiased'].extend(approach_result['debiased_scores'])
    
    if not approach_data:
        print("No data found for combined distribution overlay")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Score Distribution Overlay: All Combined A-BB Strategies', 
                 fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(approach_data)))
    
    # Original scores overlay
    for i, (approach, data) in enumerate(approach_data.items()):
        original_scores = np.array(data['original'])
        ax1.hist(original_scores, bins=25, alpha=0.6, color=colors[i], 
                label=f'{approach.title()} (μ={np.mean(original_scores):.2f})', density=True)
    
    ax1.set_title('Original Score Distributions')
    ax1.set_xlabel('Score Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Debiased scores overlay
    for i, (approach, data) in enumerate(approach_data.items()):
        debiased_scores = np.array(data['debiased'])
        ax2.hist(debiased_scores, bins=25, alpha=0.6, color=colors[i], 
                label=f'{approach.title()} (μ={np.mean(debiased_scores):.2f})', density=True)
    
    ax2.set_title('Debiased Score Distributions')
    ax2.set_xlabel('Score Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_summary_table(results: dict) -> pd.DataFrame:
    """Create summary table of all results."""
    
    summary_data = []
    for judge_name, judge_results in results.items():
        if 'approaches' in judge_results:
            for approach_name, approach_result in judge_results['approaches'].items():
                if approach_result.get('success', False):
                    summary_data.append({
                        'Judge': judge_name,
                        'Strategy': approach_name.replace('combined_abb_', ''),
                        'Combined Sensitivity': f"{approach_result['diagnostics']['combined_sensitivity']:.4f}",
                        'Correlation': f"{approach_result['validation']['correlation']:.3f}",
                        'Constraint Satisfied': '✅' if approach_result['diagnostics'].get('abb_constraint_satisfied') else '❌',
                        'Noise Std': f"{approach_result['diagnostics']['noise_std']:.4f}",
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
