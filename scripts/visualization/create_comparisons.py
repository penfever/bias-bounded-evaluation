#!/usr/bin/env python3
"""
Create Combined A-BB comparison visualizations using the existing visualize_bias_transformation.py script.

This script generates the original-style comparison plots (line plots, critical difference plots, 
summary plots) for each Combined A-BB strategy by comparing original and debiased rankings.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Dict

def run_visualization_comparison(judge_name: str, strategy: str, base_path: Path, 
                               output_base_path: Path) -> bool:
    """Run the visualization comparison for a specific judge and strategy."""
    
    print(f"\n📊 Creating visualizations for {judge_name} - {strategy} strategy...")
    
    # Paths for original and debiased rankings
    original_path = base_path / judge_name / "tables" / "factor_scores_updated_cis"
    debiased_path = base_path / judge_name / f"tables_debiased_{strategy}" / "tables" / "factor_scores_updated_cis"
    
    # Check if paths exist
    if not original_path.exists():
        print(f"  ❌ Original path not found: {original_path}")
        return False
    
    if not debiased_path.exists():
        print(f"  ❌ Debiased path not found: {debiased_path}")
        return False
    
    # Create output directory
    output_dir = output_base_path / f"{judge_name}_{strategy}_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the visualization script
    script_path = "/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/scripts/visualize_bias_transformation.py"
    
    try:
        # Use bash to source environment and run python
        bash_cmd = f"""
        source ~/.zshrc && conda activate oumi && python "{script_path}" \
        --original "{original_path}" \
        --debiased "{debiased_path}" \
        --output-dir "{output_dir}"
        """
        
        print(f"  Running visualization comparison...")
        
        result = subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  ✅ Success! Visualizations saved to: {output_dir}")
            return True
        else:
            print(f"  ❌ Error running visualization script:")
            print(f"     stdout: {result.stdout}")
            print(f"     stderr: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"  ❌ Exception running visualization: {e}")
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