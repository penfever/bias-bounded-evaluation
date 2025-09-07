#!/usr/bin/env python3
"""
Create Combined A-BB comparison visualizations using the existing visualize_bias_transformation.py script.

This script generates the original-style comparison plots (line plots, critical difference plots, 
summary plots) for each Combined A-BB strategy by comparing original and debiased rankings.
"""

import sys
import subprocess
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import List, Dict

def _sanitize(name: str) -> str:
    # Normalize to a filesystem-friendly, compact folder name
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()

def run_visualization_comparison(judge_name: str, strategy: str, base_path: Path, 
                               output_base_path: Path) -> bool:
    """Run the visualization comparison for a specific judge and strategy."""
    
    print(f"\n📊 Creating visualizations for {judge_name} - {strategy} strategy...")
    
    # Paths for original and debiased rankings
    original_path = base_path / judge_name / "tables" / "factor_scores_updated_cis"
    debiased_path = base_path / judge_name / f"tables_debiased_{strategy}" / "tables" / "factor_scores_updated_cis"
    
    # Check if paths exist
    if not original_path.exists():
        print(f"  ⚠️ Original path not found; will attempt fallback from debiased: {original_path}")
    
    if not debiased_path.exists():
        print(f"  ❌ Debiased path not found: {debiased_path}")
        return False
    
    # Create output directory
    safe_folder = f"{_sanitize(judge_name)}_{_sanitize(strategy)}_comparison"
    output_dir = output_base_path / safe_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prefer in-process generation to avoid shell/env issues
    try:
        # visualize_bias_transformation.py lives alongside this file
        vbt_path = Path(__file__).parent / "visualize_bias_transformation.py"
        if not vbt_path.exists():
            raise FileNotFoundError(f"visualize_bias_transformation.py not found at {vbt_path}")

        mod = SourceFileLoader("visualize_bias_transformation", str(vbt_path)).load_module()

        # Load data and generate plots
        print("  Loading ranking data...")
        # Always rely on debiased CSVs (they include original_score for reconstruction)
        original_data = {}
        debiased_data = mod.load_ranking_data(debiased_path)
        common_metrics = sorted(set(original_data.keys()) & set(debiased_data.keys()))
        if not common_metrics:
            print("  ❌ No common metrics to plot")
            return False

        print(f"  Generating plots for metrics: {common_metrics}")
        for metric in common_metrics:
            mod.create_line_plot_comparison(
                original_data, debiased_data, metric, output_dir / f"line_comparison_{metric}.png"
            )
            mod.create_critical_difference_plot(
                original_data, debiased_data, metric, output_dir / f"critical_difference_{metric}.png"
            )
        mod.create_summary_comparison(original_data, debiased_data, output_dir / "summary_comparison.png")

        print(f"  ✅ Success! Visualizations saved to: {output_dir}")
        return True
    except Exception as e:
        print(f"  ❌ Exception generating visualizations: {e}")
        return False

def find_available_judges(base_path: Path) -> Dict[str, List[str]]:
    """Find judges that have both original and debiased rankings available."""
    
    judges_data = {}
    
    judge_patterns = [
        "QwQ-32B-setting1",
        "DeepSeek-R1-32B-setting1",
        "DeepSeek-R1-32B-setting2", 
        "GPT-3.5-Turbo-0125-setting1",
        "GPT-4o-mini-0718-setting1"
    ]
    
    strategies = ["conservative", "rms", "weighted"]
    
    for judge in judge_patterns:
        judge_dir = base_path / judge
        if not judge_dir.exists():
            continue
            
        # Check for original rankings
        original_path = judge_dir / "tables" / "factor_scores_updated_cis"
        if not original_path.exists():
            print(f"Warning: No original rankings found for {judge}")
            continue
            
        # Check which strategies have debiased rankings
        available_strategies = []
        for strategy in strategies:
            debiased_path = judge_dir / f"tables_debiased_{strategy}" / "tables" / "factor_scores_updated_cis"
            if debiased_path.exists():
                available_strategies.append(strategy)
        
        if available_strategies:
            judges_data[judge] = available_strategies
            print(f"✅ {judge}: {len(available_strategies)} strategies available ({', '.join(available_strategies)})")
        else:
            print(f"⚠️  {judge}: No debiased rankings found")
    
    return judges_data

def main():
    """Main function to create Combined A-BB comparison visualizations."""
    
    print("🚀 Creating Combined A-BB Comparison Visualizations")
    print("=" * 70)
    
    # Paths
    base_path = Path("/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis")
    output_base_path = Path("/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/figures/combined_abb_comparisons")
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    # Find available judges and strategies
    print("🔍 Scanning for available rankings...")
    judges_data = find_available_judges(base_path)
    
    if not judges_data:
        print("❌ No judges with both original and debiased rankings found!")
        return 1
    
    print(f"\n📊 Found {len(judges_data)} judges with ranking data")
    
    # Generate comparisons for each judge and strategy
    successful_comparisons = 0
    total_comparisons = sum(len(strategies) for strategies in judges_data.values())
    
    for judge_name, strategies in judges_data.items():
        print(f"\n{'='*50}")
        print(f"Processing {judge_name}")
        print(f"{'='*50}")
        
        for strategy in strategies:
            success = run_visualization_comparison(judge_name, strategy, base_path, output_base_path)
            if success:
                successful_comparisons += 1
    
    # Summary
    print(f"\n🎯 Summary:")
    print("=" * 70)
    print(f"Total comparisons attempted: {total_comparisons}")
    print(f"Successful comparisons: {successful_comparisons}")
    print(f"Failed comparisons: {total_comparisons - successful_comparisons}")
    
    if successful_comparisons > 0:
        print(f"\n📁 Visualizations saved to: {output_base_path}")
        print("\nGenerated comparison types for each successful run:")
        print("  - line_comparison_score.png (ranking transformation with CIs)")
        print("  - critical_difference_score.png (rank position changes and magnitude)")
        print("  - summary_comparison.png (correlation, range reduction, variance reduction)")
    
    return 0 if successful_comparisons > 0 else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
