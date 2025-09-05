#!/usr/bin/env python3
"""
Comprehensive Combined A-BB Analysis with All Judges

This script runs Combined A-BB debiasing analysis with both static and dynamic 
measurements across all available judges (API-based and GGUF).
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from debias import DifferentialDebias
from config import ConfigManager
from oumi_judge_interface import create_oumi_judge_function
from run_combined_abb_analysis import (
    load_and_prepare_score_data, 
    find_judge_data_directories,
    create_synthetic_judge_function
)


def get_available_judge_configs() -> Dict[str, str]:
    """Get all available judge configurations."""
    configs_dir = Path(__file__).parent / "judge_configs"
    
    if not configs_dir.exists():
        print("❌ Judge configs directory not found. Run oumi_judge_interface.py first.")
        return {}
    
    configs = {}
    for config_file in configs_dir.glob("*.yaml"):
        judge_name = config_file.stem
        configs[judge_name] = str(config_file)
    
    return configs


def get_dataset_to_judge_mapping() -> Dict[str, str]:
    """Map dataset names to appropriate judge configs for dynamic testing."""
    
    # Map dataset judges to our available judge configs
    mapping = {
        # QwQ dataset -> QwQ GGUF judge for consistency
        "QwQ-32B-setting1": "qwq-32b-gguf",
        
        # DeepSeek datasets -> DeepSeek GGUF judge
        "DeepSeek-R1-32B-setting1": "deepseek-r1-32b-gguf", 
        "DeepSeek-R1-32B-setting2": "deepseek-r1-32b-gguf",
        "DeepSeek-R1-32B-setting3": "deepseek-r1-32b-gguf",
        
        # GPT datasets -> appropriate API judges
        "GPT-3.5-Turbo-0125-setting1": "gpt-3.5-turbo",
        "GPT-4o-mini-0718-setting1": "gpt-4o-mini",
    }
    
    return mapping


def create_comprehensive_abb_approaches(judge_function: callable) -> Dict[str, Dict]:
    """Create comprehensive A-BB approaches with different generator combinations."""
    
    approaches = {
        'static_only_conservative': {
            'estimator': 'combined_abb',
            'tau': 3.5,  # Relaxed tau for conservative strategy
            'delta': 0.25,
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': [],  # No dynamic measurements
                'combination_strategy': 'conservative',
                'judge_function': judge_function
            }
        },
        
        'static_only_rms': {
            'estimator': 'combined_abb',
            'tau': 2.3,
            'delta': 0.13,
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': [],  # No dynamic measurements
                'combination_strategy': 'rms',
                'judge_function': judge_function
            }
        },
        
        'traditional_dynamic': {
            'estimator': 'combined_abb',
            'tau': 2.8,
            'delta': 0.18,
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': ['hamming', 'formatting'],  # Traditional generators
                'combination_strategy': 'conservative',
                'num_neighbors': 3,
                'judge_function': judge_function
            }
        },
        
        'context_aware_dynamic': {
            'estimator': 'combined_abb',
            'tau': 2.5,
            'delta': 0.15,
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': ['question_paraphrasing', 'answer_perturbation', 'model_obfuscation'],  # Context-aware generators
                'combination_strategy': 'rms',
                'num_neighbors': 3,
                'judge_function': judge_function
            }
        },
        
        'comprehensive_dynamic': {
            'estimator': 'combined_abb',
            'tau': 2.2,
            'delta': 0.12,
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': ['hamming', 'formatting', 'question_paraphrasing', 'answer_perturbation'],  # Both traditional and context-aware
                'combination_strategy': 'weighted',
                'static_weights': [0.6, 0.4],  # Weight psychometric higher
                'dynamic_weights': [0.2, 0.2, 0.3, 0.3],  # Weight context-aware higher
                'num_neighbors': 2,  # Reduced for comprehensive approach
                'judge_function': judge_function
            }
        }
    }
    
    return approaches


def run_judge_analysis(judge_name: str, 
                      judge_config_path: str,
                      dataset_name: str, 
                      df: pd.DataFrame,
                      use_synthetic: bool = False) -> Dict[str, Any]:
    """Run comprehensive A-BB analysis for a single judge."""
    
    print(f"🔧 Analyzing {judge_name} on {dataset_name} ({len(df)} samples)")
    
    # Create judge function
    if use_synthetic:
        print("  Using synthetic judge function")
        judge_function = create_synthetic_judge_function()
    else:
        print(f"  Using Oumi judge: {judge_config_path}")
        try:
            judge_function = create_oumi_judge_function(
                judge_config_path,
                cost_budget_usd=5.0,  # Conservative budget per judge
                cache_responses=True
            )
        except Exception as e:
            print(f"  ❌ Failed to create Oumi judge: {e}")
            print("  🔄 Falling back to synthetic judge")
            judge_function = create_synthetic_judge_function()
    
    # Get comprehensive approaches
    approaches = create_comprehensive_abb_approaches(judge_function)
    
    results = {}
    
    for approach_name, approach_config in approaches.items():
        print(f"    Testing {approach_name}...")
        
        try:
            # Initialize debiaser
            init_params = {
                'tau': approach_config.get('tau', 2.0),
                'delta': approach_config.get('delta', 0.1),
                'sensitivity_estimator': approach_config['estimator'],
                'use_average_case': approach_config.get('use_average_case', True),
                'random_seed': 42
            }
            
            # Add extra parameters
            init_params.update(approach_config['extra_params'])
            
            debiaser = DifferentialDebias(**init_params)
            
            # Fit with factor columns
            factor_columns = [col for col in df.columns if col.endswith('_score') and col != 'overall_score']
            if factor_columns:
                debiaser.fit(df, factor_columns=factor_columns)
            else:
                debiaser.fit(df)
            
            # Transform overall scores
            original_scores = df['overall_score'].values
            score_range = float(original_scores.max() - original_scores.min())
            debiased_scores = debiaser.transform(original_scores)
            
            # Get diagnostics
            diagnostics = debiaser.get_diagnostics()
            bias_bounds = debiaser.get_bias_bounds(len(original_scores))
            validation = debiaser.validate_effectiveness(original_scores, debiased_scores)
            
            # Store results
            result_data = {
                'success': True,
                'approach': approach_name,
                'judge': judge_name,
                'dataset': dataset_name,
                'n_samples': len(original_scores),
                'diagnostics': {
                    'bias_sensitivity': float(diagnostics.get('bias_sensitivity', 0)) if diagnostics.get('bias_sensitivity') is not None else 0,
                    'combined_sensitivity': float(diagnostics.get('abb_constraint_validation', {}).get('combined_sensitivity', 0)) if diagnostics.get('abb_constraint_validation', {}).get('combined_sensitivity') is not None else 0,
                    'tau': float(bias_bounds['tau']),
                    'delta': float(bias_bounds['delta']),
                    'noise_std': float(bias_bounds['noise_std']),
                    'is_abb_mechanism': bool(diagnostics.get('is_abb_mechanism', False)),
                    'abb_constraint_satisfied': bool(diagnostics.get('abb_constraint_validation', {}).get('constraint_satisfied', False)),
                    'abb_constraint_margin': float(diagnostics.get('abb_constraint_validation', {}).get('margin', 0)) if diagnostics.get('abb_constraint_validation', {}).get('margin') is not None else 0
                },
                'validation': {
                    'correlation': float(validation['correlation']),
                    'mean_absolute_difference': float(validation['mean_absolute_difference']),
                    'variance_ratio': float(validation['variance_ratio']),
                    'signal_preservation': float(validation['signal_preservation']),
                    'noise_level': float(validation['noise_level'])
                }
            }
            
            # Add measurement breakdown
            abb_validation = diagnostics.get('abb_constraint_validation', {})
            measurement_breakdown = abb_validation.get('measurement_breakdown', {})
            if measurement_breakdown:
                result_data['measurement_breakdown'] = {
                    'static_measurements': measurement_breakdown.get('static_measurements', {}),
                    'dynamic_measurements': measurement_breakdown.get('dynamic_measurements', {}),
                    'combination_strategy': measurement_breakdown.get('combination_strategy')
                }
            
            # Add judge cost information if using real judge
            if hasattr(judge_function, 'interface'):
                cost_summary = judge_function.interface.get_cost_summary()
                result_data['judge_costs'] = cost_summary
            
            results[approach_name] = result_data
            
            print(f"      ✅ Combined sensitivity: {result_data['diagnostics'].get('combined_sensitivity', 'N/A'):.4f}")
            print(f"      ✅ A-BB constraint: {result_data['diagnostics'].get('abb_constraint_satisfied', 'N/A')}")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results[approach_name] = {
                'success': False,
                'approach': approach_name,
                'judge': judge_name,
                'dataset': dataset_name,
                'error': str(e)
            }
    
    return results


def main():
    """Main execution function."""
    print("🚀 Comprehensive Combined A-BB Analysis Across All Judges")
    print("=" * 70)
    
    # Configuration
    USE_REAL_JUDGES = True  # Set to False for synthetic judges only
    MAX_SAMPLES_PER_DATASET = 50  # Limit samples for cost management
    
    if USE_REAL_JUDGES:
        print("⚠️  REAL JUDGE MODE: Will query multiple judges via Oumi")
        print("⚠️  This will incur significant API costs for cloud judges!")
        print("⚠️  GGUF judges require local model downloads!")
        
        # Ask for confirmation
        confirmation = input("Continue with real judge querying? (y/N): ").strip().lower()
        if confirmation not in ['y', 'yes']:
            print("Aborting. Set USE_REAL_JUDGES = False for synthetic analysis.")
            return 1
    
    # Get available judge configs
    judge_configs = get_available_judge_configs()
    if not judge_configs:
        print("❌ No judge configs found!")
        return 1
    
    print(f"📋 Available judges: {', '.join(judge_configs.keys())}")
    
    # Get dataset to judge mapping
    dataset_judge_mapping = get_dataset_to_judge_mapping()
    
    # Base path for score data
    base_path = Path("/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis")
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return 1
    
    # Find judge data directories
    print("🔍 Searching for judge data directories...")
    judge_dirs = find_judge_data_directories(base_path)
    
    if not judge_dirs:
        print("❌ No judge directories with base_processed data found!")
        return 1
    
    print(f"✅ Found {len(judge_dirs)} judge datasets to process")
    
    # Overall results storage
    comprehensive_results = {}
    successful_analyses = 0
    total_cost = 0.0
    
    # Process each dataset
    for dataset_name, dataset_dir in judge_dirs.items():
        print(f"\n📊 Processing Dataset: {dataset_name}")
        print("-" * 50)
        
        # Determine which judge to use for dynamic testing
        judge_name = dataset_judge_mapping.get(dataset_name)
        if not judge_name:
            print(f"  ⚠️  No judge mapping for {dataset_name}, skipping...")
            continue
        
        judge_config_path = judge_configs.get(judge_name)
        if not judge_config_path:
            print(f"  ⚠️  Judge config not found for {judge_name}, skipping...")
            continue
        
        try:
            # Load dataset
            df = load_and_prepare_score_data(dataset_name, base_path)
            
            # Limit samples for cost management
            if len(df) > MAX_SAMPLES_PER_DATASET:
                df = df.sample(n=MAX_SAMPLES_PER_DATASET, random_state=42)
                print(f"  📉 Limited to {MAX_SAMPLES_PER_DATASET} samples for cost management")
            
            print(f"  📈 Loaded {len(df)} evaluation contexts")
            
            # Run analysis with the appropriate judge
            judge_results = run_judge_analysis(
                judge_name=judge_name,
                judge_config_path=judge_config_path,
                dataset_name=dataset_name,
                df=df,
                use_synthetic=not USE_REAL_JUDGES
            )
            
            # Store results
            comprehensive_results[dataset_name] = {
                'dataset_path': str(dataset_dir),
                'judge_used': judge_name,
                'judge_config': judge_config_path,
                'n_samples': len(df),
                'approaches': judge_results
            }
            
            # Accumulate costs
            for approach_result in judge_results.values():
                if approach_result.get('success') and 'judge_costs' in approach_result:
                    cost = approach_result['judge_costs'].get('spent_usd', 0)
                    total_cost += cost
            
            successful_analyses += 1
            print(f"  ✅ Completed analysis for {dataset_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to process {dataset_name}: {e}")
            comprehensive_results[dataset_name] = {
                'dataset_path': str(dataset_dir),
                'error': str(e)
            }
    
    # Save comprehensive results
    output_file = base_path / "comprehensive_judge_abb_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 Comprehensive A-BB Analysis Results Summary")
    print("=" * 70)
    
    print(f"Total datasets processed: {len(judge_dirs)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {len(judge_dirs) - successful_analyses}")
    print(f"Total API cost: ${total_cost:.4f}")
    
    # Analyze approach success rates across all datasets
    approach_success = {}
    approach_sensitivities = {}
    
    for dataset_results in comprehensive_results.values():
        if 'approaches' in dataset_results:
            for approach_name, approach_result in dataset_results['approaches'].items():
                if approach_name not in approach_success:
                    approach_success[approach_name] = {'success': 0, 'total': 0}
                    approach_sensitivities[approach_name] = []
                
                approach_success[approach_name]['total'] += 1
                if approach_result.get('success', False):
                    approach_success[approach_name]['success'] += 1
                    
                    # Collect sensitivity values
                    sensitivity = approach_result.get('diagnostics', {}).get('combined_sensitivity', 0)
                    if sensitivity > 0:
                        approach_sensitivities[approach_name].append(sensitivity)
    
    print(f"\n📈 Approach Performance Summary:")
    for approach, counts in approach_success.items():
        rate = (counts['success'] / counts['total']) * 100 if counts['total'] > 0 else 0
        sensitivities = approach_sensitivities[approach]
        avg_sensitivity = np.mean(sensitivities) if sensitivities else 0
        
        print(f"  • {approach}:")
        print(f"    Success: {counts['success']}/{counts['total']} ({rate:.1f}%)")
        print(f"    Avg sensitivity: {avg_sensitivity:.4f}")
    
    # Comparison analysis
    if len(approach_sensitivities) > 1:
        print(f"\n🔍 Key Findings:")
        
        # Compare static vs dynamic approaches
        static_sensitivities = []
        traditional_dynamic_sensitivities = []
        context_aware_sensitivities = []
        
        for approach, sensitivities in approach_sensitivities.items():
            if 'static_only' in approach:
                static_sensitivities.extend(sensitivities)
            elif 'traditional_dynamic' in approach:
                traditional_dynamic_sensitivities.extend(sensitivities)
            elif 'context_aware' in approach or 'comprehensive' in approach:
                context_aware_sensitivities.extend(sensitivities)
        
        if static_sensitivities and context_aware_sensitivities:
            static_avg = np.mean(static_sensitivities)
            context_avg = np.mean(context_aware_sensitivities)
            improvement = (context_avg / static_avg - 1) * 100 if static_avg > 0 else 0
            
            print(f"  📊 Static-only sensitivity: {static_avg:.4f}")
            print(f"  🧠 Context-aware sensitivity: {context_avg:.4f}")
            print(f"  📈 Context-aware improvement: {improvement:+.1f}%")
    
    print(f"\n📁 Detailed results saved to: {output_file}")
    print(f"✅ Comprehensive analysis completed!")
    
    return 0 if successful_analyses > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)