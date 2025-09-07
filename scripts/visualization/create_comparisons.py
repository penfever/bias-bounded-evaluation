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

# Import data loader utilities
from data_loader import get_project_paths, load_judge_data_for_visualization

def _sanitize(name: str) -> str:
    # Normalize to a filesystem-friendly, compact folder name
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()

def run_visualization_comparison(judge_name: str, strategy: str, base_path: Path, 
                               output_base_path: Path) -> bool:
    """Run the visualization comparison for a specific judge and strategy using JSONL data."""
    
    print(f"\n📊 Creating visualizations for {judge_name} - {strategy} strategy...")
    
    # Judge directory path
    judge_dir = base_path / judge_name
    
    # Check if judge directory exists
    if not judge_dir.exists():
        print(f"  ❌ Judge directory not found: {judge_dir}")
        return False
    
    # Map old strategy names to new approach names
    strategy_to_approach = {
        "conservative": "combined_abb_conservative",
        "rms": "combined_abb_rms",
        "weighted": "combined_abb_weighted",
        "formatting_only": "abb_formatting_only",
        "montecarlo": "combined_abb_conservative"  # Use conservative as fallback for montecarlo
    }
    
    approach_name = strategy_to_approach.get(strategy, strategy)
    
    # Check if debiased data exists (try new naming first, then old)
    debiased_dir = judge_dir / f'base_debiased_{approach_name}'
    if not debiased_dir.exists():
        debiased_dir = judge_dir / 'base_debiased'
    
    if not debiased_dir.exists():
        print(f"  ❌ No debiased data found for {judge_name} with approach {approach_name}")
        return False
    
    # Create output directory
    safe_folder = f"{_sanitize(judge_name)}_{_sanitize(strategy)}_comparison"
    output_dir = output_base_path / safe_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use the visualization module directly with JSONL data
        vbt_path = Path(__file__).parent / "visualize_bias_transformation.py"
        if not vbt_path.exists():
            raise FileNotFoundError(f"visualize_bias_transformation.py not found at {vbt_path}")

        mod = SourceFileLoader("visualize_bias_transformation", str(vbt_path)).load_module()

        # Use the new JSONL-based visualization function
        result = mod.visualize_from_jsonl(judge_dir, output_dir)
        
        if result == 0:
            print(f"  ✅ Success! Visualizations saved to: {output_dir}")
            return True
        else:
            print(f"  ❌ Visualization failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Exception generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_available_judges(base_path: Path) -> Dict[str, List[str]]:
    """Find judges that have both original and debiased data available."""
    
    judges_data = {}
    
    judge_patterns = [
        "QwQ-32B-setting1",
        "DeepSeek-R1-32B-setting1",
        "DeepSeek-R1-32B-setting2", 
        "GPT-3.5-Turbo-0125-setting1",
        "GPT-4o-mini-0718-setting1"
    ]
    
    strategies = ["conservative", "rms", "weighted", "montecarlo", "formatting_only"]
    
    for judge in judge_patterns:
        judge_dir = base_path / judge
        if not judge_dir.exists():
            continue
            
        # Check for original data (base_processed)
        original_path = judge_dir / "base_processed"
        if not original_path.exists() or not list(original_path.glob("*.jsonl")):
            print(f"Warning: No original data found for {judge}")
            continue
            
        # Check for debiased data (any base_debiased* directory)
        debiased_dirs = list(judge_dir.glob('base_debiased*'))
        if debiased_dirs and any(list(d.glob("*.jsonl")) for d in debiased_dirs):
            # All strategies use the same debiased data
            judges_data[judge] = strategies
            print(f"✅ {judge}: {len(strategies)} strategies available ({', '.join(strategies)})")
        else:
            print(f"⚠️  {judge}: No debiased data found")
    
    return judges_data

def main():
    """Main function to create Combined A-BB comparison visualizations."""
    
    print("🚀 Creating Combined A-BB Comparison Visualizations")
    print("=" * 70)
    
    # Get project paths
    paths = get_project_paths()
    base_path = paths['data_base']
    output_base_path = paths['figures'] / 'combined_abb_comparisons'
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
