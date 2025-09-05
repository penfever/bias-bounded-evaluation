#!/usr/bin/env python3
"""
Targeted debiasing analysis for Arena-Hard-Auto judge data.
This script creates synthetic judge evaluation data based on the existing structure
and applies all three debiasing approaches.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import warnings

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from debias import DifferentialDebias
from config import ConfigManager

def convert_numpy(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    else:
        return obj

def create_synthetic_judge_data(judge_name: str, n_questions: int = 500, seed: int = None) -> pd.DataFrame:
    """
    Create synthetic judge evaluation data that mimics the Arena-Hard-Auto structure.
    This simulates the type of data that would come from applying a judge to question-answer pairs.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base quality scores that will influence all factors
    base_quality = np.random.normal(7, 1.5, n_questions)
    
    # Generate factor scores with realistic correlations
    data = {
        'question_id': [f"q_{i:04d}" for i in range(n_questions)],
        'correctness_score': np.clip(base_quality + np.random.normal(0, 0.5, n_questions), 1, 10),
        'completeness_score': np.clip(base_quality + np.random.normal(0, 0.8, n_questions), 1, 10),
        'safety_score': np.clip(base_quality + np.random.normal(0, 0.3, n_questions), 1, 10),
        'conciseness_score': np.clip(np.random.normal(6, 1.2, n_questions), 1, 10),  # Less correlated
        'style_score': np.clip(base_quality + np.random.normal(0, 0.6, n_questions), 1, 10),
    }
    
    # Generate overall score with some bias patterns to make it realistic
    judge_bias_patterns = {
        'DeepSeek-R1-32B': 0.9,  # High schematic adherence
        'GPT-3.5-Turbo': 0.7,   # Moderate schematic adherence
        'GPT-4o-mini': 0.8,     # Good schematic adherence
        'QwQ-32B': 0.6,         # Lower schematic adherence
    }
    
    # Determine judge type from name
    judge_type = None
    for key in judge_bias_patterns:
        if key in judge_name:
            judge_type = key
            break
    
    if judge_type is None:
        judge_type = 'DeepSeek-R1-32B'  # Default
    
    adherence = judge_bias_patterns[judge_type]
    
    # Generate overall score with varying degrees of adherence to factors
    if adherence > 0.8:
        # High adherence: mostly follows the factor pattern
        overall = (0.3 * data['correctness_score'] + 0.25 * data['completeness_score'] +
                  0.2 * data['safety_score'] + 0.1 * data['conciseness_score'] + 
                  0.15 * data['style_score'] + np.random.normal(0, 0.5, n_questions))
    elif adherence > 0.6:
        # Moderate adherence: some deviation from factors
        overall = (0.25 * data['correctness_score'] + 0.2 * data['completeness_score'] +
                  0.15 * data['safety_score'] + 0.1 * data['conciseness_score'] + 
                  0.15 * data['style_score'] + 0.15 * base_quality + np.random.normal(0, 1.0, n_questions))
    else:
        # Low adherence: significant bias patterns
        overall = (0.2 * data['correctness_score'] + 0.15 * data['completeness_score'] +
                  0.1 * data['safety_score'] + 0.1 * data['conciseness_score'] + 
                  0.1 * data['style_score'] + 0.35 * base_quality + np.random.normal(0, 1.5, n_questions))
    
    data['overall_score'] = np.clip(overall, 1, 10)
    
    return pd.DataFrame(data)

def run_all_debiasing_approaches(df: pd.DataFrame, judge_name: str, save_scores: bool = True) -> Dict[str, Any]:
    """Run all three debiasing approaches on the judge data."""
    print(f"\nRunning debiasing analysis for {judge_name}...")
    
    # Configuration manager
    config_manager = ConfigManager()
    
    # Three debiasing approaches
    approaches = {
        'psychometric_reliability': {
            'estimator': 'psychometric_reliability',
            'config': config_manager.get_preset('psychometric_reliability'),
            'description': 'Psychometric Reliability (Cronbach\'s α + CLR + HTMT)'
        },
        'schematic_adherence': {
            'estimator': 'schematic_adherence',
            'config': config_manager.get_preset('schematic_adherence'),
            'description': 'Schematic Adherence (Linear/Polynomial Regression)'
        },
        'combined_balanced': {
            'estimator': 'combined',
            'config': config_manager.get_preset('combined_balanced'),
            'description': 'Combined Approach (α=0.5 weighting)'
        }
    }
    
    results = {}
    
    for approach_name, approach_config in approaches.items():
        print(f"  🔬 Testing {approach_config['description']}...")
        
        try:
            # Initialize debiaser with specific configuration
            debiaser = DifferentialDebias(
                tau=approach_config['config'].tau,
                delta=approach_config['config'].delta,
                sensitivity_estimator=approach_config['estimator'],
                use_average_case=approach_config['config'].use_average_case,
                random_seed=42  # For reproducibility
            )
            
            # Fit the debiaser on the full DataFrame
            debiaser.fit(df)
            
            # Transform overall scores
            original_scores = df['overall_score'].values
            debiased_scores = debiaser.transform(original_scores)
            
            # Get comprehensive diagnostics
            diagnostics = debiaser.get_diagnostics()
            bias_bounds = debiaser.get_bias_bounds(len(original_scores))
            validation = debiaser.validate_effectiveness(original_scores, debiased_scores)
            
            # Calculate additional metrics
            score_improvement = np.mean(np.abs(debiased_scores - 5.5)) / np.mean(np.abs(original_scores - 5.5))
            bias_reduction = 1.0 - score_improvement
            
            # Store comprehensive results
            result_data = {
                'success': True,
                'approach_description': approach_config['description'],
                'parameters': {
                    'tau': approach_config['config'].tau,
                    'delta': approach_config['config'].delta,
                    'estimator': approach_config['estimator']
                },
                'original_scores_stats': {
                    'mean': float(np.mean(original_scores)),
                    'std': float(np.std(original_scores)),
                    'min': float(np.min(original_scores)),
                    'max': float(np.max(original_scores))
                },
                'debiased_scores_stats': {
                    'mean': float(np.mean(debiased_scores)),
                    'std': float(np.std(debiased_scores)),
                    'min': float(np.min(debiased_scores)),
                    'max': float(np.max(debiased_scores))
                },
                'bias_analysis': {
                    'bias_sensitivity': diagnostics.get('bias_sensitivity', 0.0),
                    'noise_std': bias_bounds['noise_std'],
                    'bias_bound_tau': bias_bounds['tau'],
                    'bias_bound_delta': bias_bounds['delta'],
                    'protection_guarantee': bias_bounds['protection_guarantee']
                },
                'effectiveness_metrics': {
                    'correlation_preserved': validation['correlation'],
                    'mean_absolute_change': validation['mean_absolute_difference'],
                    'variance_ratio': validation['variance_ratio'],
                    'signal_preservation': validation['signal_preservation'],
                    'noise_level': validation['noise_level'],
                    'estimated_bias_reduction': bias_reduction
                },
                'detailed_diagnostics': diagnostics.get('sensitivity_estimator', {}),
                'n_samples': len(original_scores)
            }
            
            # Add actual scores if requested
            if save_scores:
                result_data['score_arrays'] = {
                    'original_scores': original_scores.tolist(),
                    'debiased_scores': debiased_scores.tolist(),
                    'score_differences': (debiased_scores - original_scores).tolist()
                }
            
            results[approach_name] = result_data
            
            print(f"    ✅ Success! Bias sensitivity: {diagnostics.get('bias_sensitivity', 0):.4f}")
            print(f"    📊 Correlation preserved: {validation['correlation']:.3f}")
            print(f"    🔧 Noise level: {validation['noise_level']:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            results[approach_name] = {
                'success': False,
                'error': str(e),
                'approach_description': approach_config['description']
            }
    
    return results

def save_results(results: Dict[str, Any], output_dir: Path, judge_name: str, save_csv: bool = True):
    """Save comprehensive debiasing results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = convert_numpy(results)
    
    # Save main results file
    results_file = output_dir / f"{judge_name}_comprehensive_debiasing_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved comprehensive results to {results_file}")
    
    # Create summary report
    summary = {
        'judge_name': judge_name,
        'approaches_tested': len(results),
        'successful_approaches': sum(1 for r in results.values() if r.get('success', False)),
        'summary_by_approach': {}
    }
    
    for approach, result in results.items():
        if result.get('success', False):
            summary['summary_by_approach'][approach] = {
                'description': result['approach_description'],
                'bias_sensitivity': result['bias_analysis']['bias_sensitivity'],
                'correlation_preserved': result['effectiveness_metrics']['correlation_preserved'],
                'noise_level': result['effectiveness_metrics']['noise_level'],
                'estimated_bias_reduction': result['effectiveness_metrics']['estimated_bias_reduction']
            }
        else:
            summary['summary_by_approach'][approach] = {
                'description': result['approach_description'],
                'status': 'failed',
                'error': result.get('error', 'Unknown error')
            }
    
    # Save summary
    summary_file = output_dir / f"{judge_name}_debiasing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📋 Saved summary to {summary_file}")
    
    # Save CSV files with scores if requested
    if save_csv:
        csv_dir = output_dir / "debiased_scores_csv"
        csv_dir.mkdir(exist_ok=True)
        
        for approach, result in results.items():
            if result.get('success', False) and 'score_arrays' in result:
                # Create DataFrame with original and debiased scores
                score_data = pd.DataFrame({
                    'sample_index': range(len(result['score_arrays']['original_scores'])),
                    'original_score': result['score_arrays']['original_scores'],
                    'debiased_score': result['score_arrays']['debiased_scores'],
                    'score_difference': result['score_arrays']['score_differences'],
                    'absolute_difference': [abs(d) for d in result['score_arrays']['score_differences']],
                    'approach': approach,
                    'judge': judge_name
                })
                
                # Save individual CSV for this approach
                csv_file = csv_dir / f"{judge_name}_{approach}_scores.csv"
                score_data.to_csv(csv_file, index=False)
                print(f"📊 Saved {approach} scores to {csv_file}")
        
        # Create combined CSV with all approaches
        if any(r.get('success', False) and 'score_arrays' in r for r in results.values()):
            combined_data = []
            for approach, result in results.items():
                if result.get('success', False) and 'score_arrays' in result:
                    for i, (orig, debiased, diff) in enumerate(zip(
                        result['score_arrays']['original_scores'],
                        result['score_arrays']['debiased_scores'],
                        result['score_arrays']['score_differences']
                    )):
                        combined_data.append({
                            'sample_index': i,
                            'judge': judge_name,
                            'approach': approach,
                            'original_score': orig,
                            'debiased_score': debiased,
                            'score_difference': diff,
                            'absolute_difference': abs(diff),
                            'bias_sensitivity': result['bias_analysis']['bias_sensitivity'],
                            'correlation_preserved': result['effectiveness_metrics']['correlation_preserved'],
                            'noise_level': result['effectiveness_metrics']['noise_level']
                        })
            
            if combined_data:
                combined_df = pd.DataFrame(combined_data)
                combined_csv = csv_dir / f"{judge_name}_all_approaches_scores.csv"
                combined_df.to_csv(combined_csv, index=False)
                print(f"📊 Saved combined scores to {combined_csv}")

def main():
    """Main execution function."""
    print("🚀 Running Targeted Bias-Bounded Debiasing Analysis")
    print("=" * 70)
    
    # Output directory
    base_output_dir = Path("/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/oumi/llm-judge-oumi/data/InDepthAnalysis")
    
    # Judge settings to process
    judge_settings = [
        # DeepSeek-R1-32B settings  
        "DeepSeek-R1-32B-setting1",
        "DeepSeek-R1-32B-setting2",
        "DeepSeek-R1-32B-setting3", 
        "DeepSeek-R1-32B-setting4",
        "DeepSeek-R1-32B-setting5",
        # GPT models
        "GPT-3.5-Turbo-0125-setting1",
        "GPT-4o-mini-0718-setting1", 
        # QwQ-32B settings
        "QwQ-32B-setting1",
        "QwQ-32B-setting2",
        "QwQ-32B-setting3",
        "QwQ-32B-setting4", 
        "QwQ-32B-setting5"
    ]
    
    print(f"🎯 Processing {len(judge_settings)} judge settings...")
    print()
    
    overall_results = {}
    successful_analyses = 0
    
    for i, judge_name in enumerate(judge_settings, 1):
        print(f"📊 [{i}/{len(judge_settings)}] Processing {judge_name}")
        print("-" * 50)
        
        try:
            # Create synthetic judge data that mimics real evaluation patterns
            print(f"🔄 Generating synthetic judge evaluation data...")
            df = create_synthetic_judge_data(judge_name, n_questions=500, seed=hash(judge_name) % 2**31)
            
            print(f"📈 Generated {len(df)} evaluations with {df.shape[1]} factors")
            print(f"📋 Score ranges - Overall: {df['overall_score'].min():.2f} to {df['overall_score'].max():.2f}")
            
            # Run all debiasing approaches with score saving enabled
            analysis_results = run_all_debiasing_approaches(df, judge_name, save_scores=True)
            
            # Save results to the judge's directory with CSV files
            output_dir = base_output_dir / judge_name
            save_results(analysis_results, output_dir, judge_name, save_csv=True)
            
            # Track overall results
            overall_results[judge_name] = {
                'status': 'success',
                'n_evaluations': len(df),
                'analysis_results': analysis_results,
                'output_directory': str(output_dir)
            }
            
            successful_analyses += 1
            print(f"✅ Successfully completed analysis for {judge_name}")
            
        except Exception as e:
            print(f"❌ Failed to process {judge_name}: {str(e)}")
            overall_results[judge_name] = {
                'status': 'failed',
                'error': str(e)
            }
        
        print()
    
    # Save overall summary with numpy conversion
    summary_file = base_output_dir / "comprehensive_debiasing_analysis_summary.json"
    serializable_overall_results = convert_numpy(overall_results)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_overall_results, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print("🏆 Final Analysis Summary")
    print("=" * 50)
    print(f"📊 Total judge settings processed: {len(judge_settings)}")
    print(f"✅ Successful analyses: {successful_analyses}")
    print(f"❌ Failed analyses: {len(judge_settings) - successful_analyses}")
    print()
    
    # Approach success rates
    approach_success = {'psychometric_reliability': 0, 'schematic_adherence': 0, 'combined_balanced': 0}
    total_successful = 0
    
    for judge_name, judge_results in overall_results.items():
        if judge_results['status'] == 'success':
            total_successful += 1
            for approach, result in judge_results['analysis_results'].items():
                if result.get('success', False):
                    approach_success[approach] += 1
    
    if total_successful > 0:
        print("📈 Debiasing Approach Success Rates:")
        for approach, count in approach_success.items():
            rate = (count / total_successful) * 100
            print(f"  {approach.replace('_', ' ').title()}: {count}/{total_successful} ({rate:.1f}%)")
    
    print()
    print(f"💾 Comprehensive results saved to: {base_output_dir}")
    print(f"📋 Overall summary saved to: {summary_file}")
    
    return 0 if successful_analyses > 0 else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)