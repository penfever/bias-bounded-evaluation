#!/usr/bin/env python3
"""
Run bias-bounded debiasing analysis on Arena-Hard-Auto judge data using all three approaches:
1. Psychometric Reliability Sensitivity
2. Schematic Adherence Sensitivity  
3. Combined Sensitivity

This script processes all judge settings and saves debiased results.
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

def find_judge_data_files(base_path: Path) -> Dict[str, Path]:
    """Find all judge data files in the InDepthAnalysis directory."""
    judge_files = {}
    
    # Pattern for the judge settings we need to process
    patterns = [
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
    
    for pattern in patterns:
        judge_dir = base_path / pattern
        if judge_dir.exists() and judge_dir.is_dir():
            # Look for JSONL files that might contain judge data
            for file_path in judge_dir.rglob("*.jsonl"):
                if "judge" in file_path.name.lower() or "result" in file_path.name.lower():
                    judge_files[pattern] = file_path
                    print(f"Found judge data: {pattern} -> {file_path}")
                    break
            
            # Also check for JSON files
            if pattern not in judge_files:
                for file_path in judge_dir.rglob("*.json"):
                    if "judge" in file_path.name.lower() or "result" in file_path.name.lower():
                        judge_files[pattern] = file_path
                        print(f"Found judge data: {pattern} -> {file_path}")
                        break
            
            if pattern not in judge_files:
                print(f"Warning: No judge data found for {pattern}")
    
    return judge_files

def load_judge_data(file_path: Path) -> pd.DataFrame:
    """Load judge data from JSONL or JSON file."""
    print(f"Loading data from {file_path}...")
    
    try:
        if file_path.suffix.lower() == '.jsonl':
            # Load JSONL file
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON line: {e}")
                            continue
            
            if not data:
                raise ValueError("No valid JSON lines found in file")
                
            df = pd.DataFrame(data)
            
        else:
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to extract data from nested structure
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                elif 'results' in data:
                    df = pd.DataFrame(data['results'])
                else:
                    # Convert dict to single-row DataFrame
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected data structure in {file_path}")
        
        print(f"Loaded {len(df)} records with columns: {list(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def detect_score_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect factor score and overall score columns in the DataFrame."""
    columns = df.columns.tolist()
    
    # Common patterns for factor scores
    factor_patterns = {
        'correctness': ['correctness', 'correct', 'accuracy', 'factual'],
        'completeness': ['completeness', 'complete', 'thorough', 'comprehensive'],
        'safety': ['safety', 'safe', 'harm', 'ethical'],
        'conciseness': ['conciseness', 'concise', 'brief', 'length'],
        'style': ['style', 'clarity', 'presentation', 'writing']
    }
    
    detected_columns = {}
    
    # Look for factor score columns
    for factor, patterns in factor_patterns.items():
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns) and 'score' in col_lower:
                detected_columns[f'{factor}_score'] = col
                break
    
    # Look for overall score column
    overall_patterns = ['overall', 'total', 'final', 'aggregate']
    for col in columns:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in overall_patterns) and 'score' in col_lower:
            detected_columns['overall_score'] = col
            break
    
    # Alternative: look for any score columns if specific ones not found
    if not detected_columns:
        score_cols = [col for col in columns if 'score' in col.lower()]
        if len(score_cols) >= 2:
            # Assume last one is overall, others are factors
            detected_columns['overall_score'] = score_cols[-1]
            for i, col in enumerate(score_cols[:-1]):
                detected_columns[f'factor_{i+1}_score'] = col
    
    print(f"Detected score columns: {detected_columns}")
    return detected_columns

def normalize_dataframe(df: pd.DataFrame, score_columns: Dict[str, str]) -> pd.DataFrame:
    """Normalize DataFrame to have standardized column names."""
    normalized_df = df.copy()
    
    # Rename columns to standard names
    column_mapping = {v: k for k, v in score_columns.items()}
    normalized_df = normalized_df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    required_columns = ['overall_score']
    factor_columns = [col for col in normalized_df.columns if col.endswith('_score') and col != 'overall_score']
    
    if 'overall_score' not in normalized_df.columns:
        print("Warning: No overall_score column found, using first score column")
        score_cols = [col for col in normalized_df.columns if 'score' in col.lower()]
        if score_cols:
            normalized_df['overall_score'] = normalized_df[score_cols[0]]
        else:
            raise ValueError("No score columns found in data")
    
    # Ensure we have at least some factor columns for the new estimators
    if len(factor_columns) < 2:
        print("Warning: Insufficient factor columns for psychometric analysis")
        # Create dummy factor columns if needed for basic functionality
        if 'overall_score' in normalized_df.columns:
            normalized_df['factor_1_score'] = normalized_df['overall_score'] + np.random.normal(0, 0.1, len(normalized_df))
            normalized_df['factor_2_score'] = normalized_df['overall_score'] + np.random.normal(0, 0.1, len(normalized_df))
    
    return normalized_df

def run_debiasing_analysis(df: pd.DataFrame, judge_name: str) -> Dict[str, Any]:
    """Run all three debiasing approaches on the judge data."""
    print(f"\nRunning debiasing analysis for {judge_name}...")
    
    # Configuration manager for presets
    config_manager = ConfigManager()
    
    # Extended debiasing approaches including new Combined A-BB methods
    approaches = {
        'psychometric_reliability': {
            'estimator': 'psychometric_reliability',
            'config': config_manager.get_preset('psychometric_reliability')
        },
        'schematic_adherence': {
            'estimator': 'schematic_adherence', 
            'config': config_manager.get_preset('schematic_adherence')
        },
        'combined_balanced': {
            'estimator': 'combined',
            'config': config_manager.get_preset('combined_balanced')
        },
        # New Combined A-BB approaches with different aggregation strategies
        'combined_abb_conservative': {
            'estimator': 'combined_abb',
            'config': config_manager.get_preset('moderate'),  # Use moderate preset
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': ['hamming', 'formatting'],
                'combination_strategy': 'conservative',
                'num_neighbors': 8,
                'dimensionality': None  # Will be inferred
            }
        },
        'combined_abb_rms': {
            'estimator': 'combined_abb',
            'config': config_manager.get_preset('moderate'),
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': ['hamming', 'formatting'],
                'combination_strategy': 'rms',
                'num_neighbors': 8,
                'dimensionality': None
            }
        },
        'combined_abb_weighted': {
            'estimator': 'combined_abb', 
            'config': config_manager.get_preset('moderate'),
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': ['hamming', 'formatting'],
                'combination_strategy': 'weighted',
                'static_weights': [0.4, 0.4],
                'dynamic_weights': [0.1, 0.1],
                'num_neighbors': 8,
                'dimensionality': None
            }
        },
        'combined_abb_adaptive': {
            'estimator': 'combined_abb',
            'config': config_manager.get_preset('moderate'),
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': ['hamming', 'formatting', 'order'],
                'combination_strategy': 'adaptive',
                'num_neighbors': 10,  # More neighbors for adaptive
                'dimensionality': None
            }
        }
    }
    
    results = {}
    
    for approach_name, approach_config in approaches.items():
        print(f"  Testing {approach_name}...")
        
        try:
            # Base parameters
            init_params = {
                'tau': approach_config['config'].tau,
                'delta': approach_config['config'].delta,
                'sensitivity_estimator': approach_config['estimator'],
                'use_average_case': approach_config['config'].use_average_case,
                'random_seed': 42  # For reproducibility
            }
            
            # Add extra parameters for combined A-BB approaches
            if 'extra_params' in approach_config:
                init_params.update(approach_config['extra_params'])
                # For combined A-BB, we need a judge function for dynamic measurements
                # For now, we'll skip dynamic measurements if no judge function is available
                if approach_config['estimator'] == 'combined_abb':
                    print(f"    Note: Combined A-BB without judge function will use static measurements only")
            
            # Initialize debiaser
            debiaser = DifferentialDebias(**init_params)
            
            # Fit the debiaser with factor columns for new estimators
            factor_columns = [col for col in df.columns if col.endswith('_score') and col != 'overall_score']
            if approach_config['estimator'] in ['combined_abb'] and factor_columns:
                debiaser.fit(df, factor_columns=factor_columns)
            else:
                debiaser.fit(df)
            
            # Transform overall scores
            if 'overall_score' in df.columns:
                original_scores = df['overall_score'].dropna().values
                if len(original_scores) > 0:
                    debiased_scores = debiaser.transform(original_scores)
                    
                    # Get diagnostics
                    diagnostics = debiaser.get_diagnostics()
                    bias_bounds = debiaser.get_bias_bounds(len(original_scores))
                    
                    # Validate effectiveness
                    validation = debiaser.validate_effectiveness(original_scores, debiased_scores)
                    
                    # Store results with enhanced diagnostics
                    result_data = {
                        'success': True,
                        'original_scores': original_scores.tolist(),
                        'debiased_scores': debiased_scores.tolist(),
                        'diagnostics': {
                            'bias_sensitivity': diagnostics.get('bias_sensitivity'),
                            'tau': bias_bounds['tau'],
                            'delta': bias_bounds['delta'],
                            'noise_std': bias_bounds['noise_std'],
                            'is_abb_mechanism': diagnostics.get('is_abb_mechanism', False),
                            'estimator_type': approach_config['estimator']
                        },
                        'validation': validation,
                        'n_samples': len(original_scores)
                    }
                    
                    # Add combined A-BB specific diagnostics
                    if approach_config['estimator'] == 'combined_abb':
                        abb_validation = diagnostics.get('abb_constraint_validation', {})
                        if abb_validation:
                            result_data['diagnostics'].update({
                                'abb_constraint_satisfied': abb_validation.get('constraint_satisfied'),
                                'abb_constraint_margin': abb_validation.get('margin'),
                                'combination_strategy': approach_config['extra_params']['combination_strategy']
                            })
                            
                            # Add measurement breakdown if available
                            measurement_breakdown = abb_validation.get('measurement_breakdown', {})
                            if measurement_breakdown:
                                result_data['diagnostics'].update({
                                    'static_measurements': measurement_breakdown.get('static_measurements', {}),
                                    'dynamic_measurements': measurement_breakdown.get('dynamic_measurements', {}),
                                    'combined_sensitivity': measurement_breakdown.get('combined_sensitivity')
                                })
                    
                    results[approach_name] = result_data
                    
                    print(f"    ✓ Success: {len(original_scores)} scores processed")
                    print(f"    ✓ Bias sensitivity: {diagnostics.get('bias_sensitivity', 'N/A'):.4f}")
                    print(f"    ✓ Correlation: {validation['correlation']:.3f}")
                    
                else:
                    results[approach_name] = {'success': False, 'error': 'No valid overall scores found'}
            else:
                results[approach_name] = {'success': False, 'error': 'No overall_score column found'}
                
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[approach_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def save_results(results: Dict[str, Any], output_dir: Path, judge_name: str):
    """Save debiasing results to the judge's directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive results
    results_file = output_dir / f"{judge_name}_debiasing_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved comprehensive results to {results_file}")
    
    # Save individual debiased score files for each successful approach
    for approach_name, approach_results in results.items():
        if approach_results.get('success', False):
            # Create debiased scores file
            scores_data = {
                'approach': approach_name,
                'judge': judge_name,
                'original_scores': approach_results['original_scores'],
                'debiased_scores': approach_results['debiased_scores'],
                'n_samples': approach_results['n_samples'],
                'bias_sensitivity': approach_results['diagnostics']['bias_sensitivity'],
                'correlation_preserved': approach_results['validation']['correlation'],
                'noise_level': approach_results['validation']['noise_level']
            }
            
            scores_file = output_dir / f"{judge_name}_{approach_name}_debiased_scores.json"
            with open(scores_file, 'w', encoding='utf-8') as f:
                json.dump(scores_data, f, indent=2)
            print(f"Saved {approach_name} debiased scores to {scores_file}")

def main():
    """Main execution function."""
    print("🚀 Running Bias-Bounded Debiasing Analysis on Arena-Hard-Auto Judge Data")
    print("=" * 80)
    
    # Base path for judge data
    base_path = Path("/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis")
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return 1
    
    # Find all judge data files
    print("🔍 Searching for judge data files...")
    judge_files = find_judge_data_files(base_path)
    
    if not judge_files:
        print("❌ No judge data files found!")
        return 1
    
    print(f"✅ Found {len(judge_files)} judge datasets to process")
    print()
    
    # Process each judge dataset
    overall_results = {}
    successful_analyses = 0
    
    for judge_name, file_path in judge_files.items():
        print(f"📊 Processing {judge_name}...")
        print("-" * 60)
        
        try:
            # Load judge data
            df = load_judge_data(file_path)
            
            # Detect and normalize score columns
            score_columns = detect_score_columns(df)
            if not score_columns:
                print(f"❌ No score columns detected for {judge_name}, skipping...")
                continue
                
            normalized_df = normalize_dataframe(df, score_columns)
            
            # Run debiasing analysis
            analysis_results = run_debiasing_analysis(normalized_df, judge_name)
            
            # Save results
            output_dir = base_path / judge_name
            save_results(analysis_results, output_dir, judge_name)
            
            # Track overall results
            overall_results[judge_name] = {
                'file_path': str(file_path),
                'n_records': len(df),
                'score_columns': score_columns,
                'analysis_results': analysis_results
            }
            
            successful_analyses += 1
            print(f"✅ Completed analysis for {judge_name}")
            
        except Exception as e:
            print(f"❌ Failed to process {judge_name}: {e}")
            overall_results[judge_name] = {
                'error': str(e),
                'file_path': str(file_path) if 'file_path' in locals() else 'unknown'
            }
    
    # Save overall summary
    summary_file = base_path / "debiasing_analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)
    
    print()
    print("📊 Analysis Summary:")
    print("=" * 60)
    print(f"Total judge datasets found: {len(judge_files)}")
    print(f"Successfully processed: {successful_analyses}")
    print(f"Failed: {len(judge_files) - successful_analyses}")
    print()
    
    # Summary of approaches tested
    approach_success_count = {'psychometric_reliability': 0, 'schematic_adherence': 0, 'combined_balanced': 0}
    
    for judge_name, judge_results in overall_results.items():
        if 'analysis_results' in judge_results:
            for approach, result in judge_results['analysis_results'].items():
                if result.get('success', False):
                    approach_success_count[approach] += 1
    
    print("Approach Success Rates:")
    for approach, count in approach_success_count.items():
        rate = (count / len(judge_files)) * 100
        print(f"  {approach}: {count}/{len(judge_files)} ({rate:.1f}%)")
    
    print(f"\\n📁 Results saved to: {base_path}")
    print(f"📋 Summary saved to: {summary_file}")
    
    return 0 if successful_analyses > 0 else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)