#!/usr/bin/env python3
"""
Run Combined A-BB debiasing analysis on existing judge score data.

This script loads existing judge score data and applies the new Combined A-BB 
approaches with different aggregation strategies to demonstrate the enhanced
system-level bias measurement capabilities.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
import argparse

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from differential_debiasing.core.debias import DifferentialDebias
from differential_debiasing.core.config import ConfigManager
from differential_debiasing.core.sensitivity_profiles import load_judge_sensitivity_profile, get_formatting_sensitivity
from differential_debiasing.interfaces.oumi_interface import create_oumi_judge_function

def get_judge_config_path(judge_name: str) -> Path:
    """Get the correct config path for a judge based on the new directory structure."""
    # Get the base config directory (go up from scripts/analysis to root)
    base_config_dir = Path(__file__).parent.parent.parent / "configs" / "judges"
    
    # Map judge names to their correct subdirectories
    judge_mapping = {
        # OpenAI models
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo.yaml",
        "gpt-4o-mini": "openai/gpt-4o-mini.yaml",
        
        # Anthropic models  
        "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet.yaml",
        
        # Local GGUF models
        "qwq-32b-gguf": "local/qwq-32b-gguf.yaml",
        "deepseek-r1-32b-gguf": "local/deepseek-r1-32b-gguf.yaml",
    }
    
    if judge_name in judge_mapping:
        config_path = base_config_dir / judge_mapping[judge_name]
        if config_path.exists():
            return config_path
    
    # No fallback - fail fast if not found
    raise FileNotFoundError(f"Judge config not found for '{judge_name}'. Expected at one of: {list(judge_mapping.values())}")

def find_judge_data_directories(base_path: Path) -> Dict[str, Path]:
    """Find judge directories with base_processed data."""
    judge_dirs = {}
    
    # Pattern for the judge settings we want to process (diverse set of judges)
    patterns = [
        "QwQ-32B-setting1",
        "DeepSeek-R1-32B-setting1", 
        "DeepSeek-R1-32B-setting2",
        "GPT-3.5-Turbo-0125-setting1",
        "GPT-4o-mini-0718-setting1"
    ]
    
    for pattern in patterns:
        judge_dir = base_path / pattern
        if judge_dir.exists() and judge_dir.is_dir():
            # Check for base_processed directory with JSONL files
            base_processed_dir = judge_dir / "base_processed"
            if base_processed_dir.exists() and any(base_processed_dir.glob("*.jsonl")):
                judge_dirs[pattern] = judge_dir
                print(f"Found judge data: {pattern} -> {judge_dir}")
    
    return judge_dirs

def load_judge_evaluations(base_processed_dir: Path) -> Dict[str, List[Dict]]:
    """Load all judge evaluations from JSONL files."""
    print(f"Loading evaluations from {base_processed_dir}...")
    
    evaluations = {}
    
    for jsonl_file in base_processed_dir.glob("*.jsonl"):
        model_name = jsonl_file.stem
        model_evaluations = []
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            model_evaluations.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON line in {jsonl_file}: {e}")
                            continue
            
            if model_evaluations:
                evaluations[model_name] = model_evaluations
                print(f"  Loaded {len(model_evaluations)} evaluations for {model_name}")
        
        except Exception as e:
            print(f"Error loading {jsonl_file}: {e}")
            continue
    
    return evaluations

def get_score_mapping():
    """Get the score mapping dictionary."""
    return {
        '': 3,
        'A>>B': 1,
        'A>B': 2, 
        'A=B': 3,
        'B>A': 4,
        'B>>A': 5,
        'A<<B': 5,
        'A<B': 4,
        'B=A': 3,
        'B<A': 2,
        'B<<A': 1
    }

def extract_scores_from_evaluations(evaluations: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Extract all scores from evaluations using Arena-Hard-Auto Likert scale transformations."""
    # Score mapping for conversion from comparative judgments to 5-point Likert scale
    score_mapping = get_score_mapping()
    
    all_scores = []
    
    for model_name, model_evaluations in evaluations.items():
        for eval_data in model_evaluations:
            question_id = eval_data.get('question_id', '')
            games = eval_data.get('games', [])
            
            for game in games:
                # Initialize score data
                score_data = {
                    'model': model_name,
                    'question_id': question_id,
                }
                
                # Extract scores for each factor using the actual game data
                factor_fields = ['score', 'correctness_score', 'completeness_score', 
                               'safety_score', 'conciseness_score', 'style_score']
                
                for factor in factor_fields:
                    if factor in game:
                        # Get the raw judgment value
                        raw_judgment = game[factor]
                        
                        # Convert using the score mapping from Arena-Hard-Auto
                        if isinstance(raw_judgment, str) and raw_judgment.strip():
                            numeric_score = score_mapping.get(raw_judgment.strip(), 3)  # Default to 3 (tie)
                        else:
                            numeric_score = 3  # Default to neutral/tie
                        
                        score_data[factor] = float(numeric_score)
                    else:
                        # If factor not present, default to neutral score
                        score_data[factor] = 3.0
                
                # Ensure we have an overall score
                if 'overall_score' not in score_data:
                    score_data['overall_score'] = score_data.get('score', 3.0)
                
                all_scores.append(score_data)
    
    df = pd.DataFrame(all_scores)
    
    if len(df) > 0:
        print(f"Extracted {len(df)} real judge evaluations with Likert scale scores")
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Score ranges: {[(col, df[col].min(), df[col].max()) for col in df.columns if col.endswith('_score')]}")
    else:
        print("Warning: No scores extracted from evaluations")
    
    return df

def load_and_prepare_score_data(judge_name: str, base_path: Path) -> pd.DataFrame:
    """Load real judge evaluation data and prepare it for Combined A-BB analysis."""
    print(f"Loading real evaluation data for {judge_name}...")
    
    # Find the judge directory
    judge_dir = base_path / judge_name
    if not judge_dir.exists():
        raise ValueError(f"Judge directory not found: {judge_dir}")
    
    # Load evaluations from base_processed directory
    base_processed_dir = judge_dir / "base_processed"
    if not base_processed_dir.exists():
        raise ValueError(f"Base processed directory not found: {base_processed_dir}")
    
    # Load judge evaluations
    evaluations = load_judge_evaluations(base_processed_dir)
    if not evaluations:
        raise ValueError(f"No evaluations found in {base_processed_dir}")
    
    # Extract scores for debiasing analysis
    scores_df = extract_scores_from_evaluations(evaluations)
    if len(scores_df) < 10:
        raise ValueError(f"Not enough evaluations for debiasing ({len(scores_df)})")
    
    return scores_df

## Synthetic judge removed: script now exclusively uses real judges via Oumi

def check_sensitivity_profiles(real_judge_name: str) -> Dict[str, Any]:
    """
    Check for available sensitivity profiles for a judge.
    
    Parameters:
    -----------
    real_judge_name : str
        Name of the real judge to check
        
    Returns:
    --------
    Dict[str, Any] : Profile availability and values
    """
    # Look for profiles in the project root sensitivity_profiles directory
    profile_dir = Path(__file__).parent.parent.parent / "sensitivity_profiles"
    
    profile_info = {
        'profile_available': False,
        'formatting_sensitivity': None,
        'profile_dir': str(profile_dir)
    }
    
    try:
        profile = load_judge_sensitivity_profile(real_judge_name, profile_dir)
        if profile:
            formatting_sensitivity = profile.get_formatting_sensitivity()
            if formatting_sensitivity is not None:
                profile_info['profile_available'] = True
                profile_info['formatting_sensitivity'] = formatting_sensitivity
                print(f"✅ Found sensitivity profile for {real_judge_name}: formatting={formatting_sensitivity:.4f}")
            else:
                print(f"⚠️ Profile found for {real_judge_name} but formatting sensitivity is missing/failed")
        else:
            print(f"📋 No sensitivity profile found for {real_judge_name}")
    except Exception as e:
        print(f"⚠️ Error loading profile for {real_judge_name}: {e}")
    
    return profile_info


def run_combined_abb_analysis(df: pd.DataFrame, judge_name: str, 
                              real_judge_name: str) -> Dict[str, Any]:
    """Run Combined A-BB debiasing approaches on the judge data."""
    print(f"\nRunning Combined A-BB analysis for {judge_name}...")
    
    config_manager = ConfigManager()
    
    # Check for sensitivity profiles for the real judge
    profile_info = check_sensitivity_profiles(real_judge_name)
    
    # Create real judge function using Oumi interface with cost management
    print(f"Attempting to use real judge: {real_judge_name}")
    judge_config_path = get_judge_config_path(real_judge_name)
    if not judge_config_path.exists():
        raise FileNotFoundError(f"Judge config not found at {judge_config_path}")
    judge_function = create_oumi_judge_function(
        str(judge_config_path),
        cost_budget_usd=5.0,  # Conservative budget for testing
        cache_responses=True
    )
    print(f"✅ Successfully created {real_judge_name} judge function")
    
    # Determine dynamic generators based on profile availability
    # If we have formatting sensitivity profile, skip expensive formatting measurement
    if profile_info and profile_info['profile_available']:
        dynamic_generators = ['hamming']  # Only use fast hamming, skip slow formatting
        print(f"🚀 Using sensitivity profile - skipping expensive formatting measurement")
    else:
        dynamic_generators = ['hamming', 'formatting']  # Use both
        print(f"📏 No profile available - will measure hamming and formatting dynamically")

    # Define Combined A-BB approaches with different aggregation strategies  
    # Using relaxed tau/delta for conservative strategy to help with constraint satisfaction
    # Choose estimator type based on profile availability
    estimator_type = 'profile_enhanced_abb' if (profile_info and profile_info['profile_available']) else 'combined_abb'
    
    approaches = {
        'combined_abb_conservative': {
            'estimator': estimator_type,
            'tau': 3.5,  # Further relaxed tau for conservative strategy (was 3.0)
            'delta': 0.25,  # Further relaxed delta for conservative strategy (was 0.2)
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': dynamic_generators,
                'combination_strategy': 'conservative',
                'num_neighbors': 10,  # Efficient neighbor count for fast sensitivity measurement
                'target_samples': 10,  # Efficient sampling for dynamic scoring
                'dimensionality': None,
                'judge_function': judge_function,
                'sensitivity_profile': real_judge_name if profile_info and profile_info['profile_available'] else None,
                'profile_dir': str(Path(__file__).parent.parent.parent / "sensitivity_profiles")
            }
        },
        'combined_abb_rms': {
            'estimator': estimator_type,
            'tau': 2.3,  # Further relaxed tau for RMS strategy (was 2.2)
            'delta': 0.13,  # Further relaxed delta for RMS strategy (was 0.12)
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': dynamic_generators,
                'combination_strategy': 'rms',
                'num_neighbors': 10,  # Efficient neighbor count for fast sensitivity measurement
                'target_samples': 10,  # Efficient sampling for dynamic scoring
                'dimensionality': None,
                'judge_function': judge_function,
                'sensitivity_profile': real_judge_name if profile_info and profile_info['profile_available'] else None,
                'profile_dir': str(Path(__file__).parent.parent.parent / "sensitivity_profiles")
            }
        },
        'combined_abb_weighted': {
            'estimator': estimator_type,
            'tau': 2.2,  # Slightly relaxed tau for weighted strategy (was 2.0)
            'delta': 0.12,  # Slightly relaxed delta for weighted strategy (was 0.1)
            'use_average_case': True,
            'extra_params': {
                'static_estimators': ['psychometric_reliability', 'schematic_adherence'],
                'dynamic_generators': dynamic_generators,
                'combination_strategy': 'weighted',
                'static_weights': [0.6, 0.4],  # Weight psychometric higher
                'dynamic_weights': [0.7, 0.3], # Weight hamming higher
                'num_neighbors': 10,  # Efficient neighbor count for fast sensitivity measurement
                'target_samples': 10,  # Efficient sampling for dynamic scoring
                'dimensionality': None,
                'judge_function': judge_function
            }
        }
    }
    
    results = {}
    
    for approach_name, approach_config in approaches.items():
        print(f"  Testing {approach_name} strategy...")
        
        try:
            # Base parameters
            init_params = {
                'tau': approach_config.get('tau', 2.0),
                'delta': approach_config.get('delta', 0.1),
                'sensitivity_estimator': approach_config['estimator'],
                'use_average_case': approach_config.get('use_average_case', True),
                'random_seed': 42
            }
            
            # Add extra parameters
            init_params.update(approach_config['extra_params'])
            
            # Initialize debiaser
            debiaser = DifferentialDebias(**init_params)
            
            # Fit with factor columns
            factor_columns = [col for col in df.columns if col.endswith('_score') and col != 'overall_score']
            if factor_columns:
                debiaser.fit(df, factor_columns=factor_columns)
            else:
                debiaser.fit(df)
            
            # Transform overall scores with proper score range
            original_scores = df['overall_score'].values
            score_range = float(original_scores.max() - original_scores.min())
            debiased_scores = debiaser.transform(original_scores)
            
            # Get diagnostics
            diagnostics = debiaser.get_diagnostics()
            bias_bounds = debiaser.get_bias_bounds(len(original_scores))
            
            # Validate effectiveness
            validation = debiaser.validate_effectiveness(original_scores, debiased_scores)
            
            # Store comprehensive results (convert numpy types for JSON serialization)
            result_data = {
                'success': True,
                'approach': approach_name,
                'strategy': approach_config['extra_params']['combination_strategy'],
                'original_scores': [float(x) for x in original_scores.tolist()],
                'debiased_scores': [float(x) for x in debiased_scores.tolist()],
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
                },
                'n_samples': int(len(original_scores))
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
            
            print(f"    ✅ Success: {len(original_scores)} scores processed")
            # Handle combined sensitivity safely
            combined_sens = result_data['diagnostics'].get('combined_sensitivity', 'N/A')
            if isinstance(combined_sens, (int, float)):
                import numpy as np
                if np.isnan(combined_sens) or np.isinf(combined_sens):
                    print(f"    ⚠️ Combined sensitivity: {combined_sens} (invalid)")
                else:
                    print(f"    ✅ Combined sensitivity: {combined_sens:.4f}")
            else:
                print(f"    ✅ Combined sensitivity: {combined_sens}")
            
            print(f"    ✅ Correlation: {validation['correlation']:.3f}")
            print(f"    ✅ A-BB constraint satisfied: {result_data['diagnostics'].get('abb_constraint_satisfied', 'N/A')}")
            
            # Report individual dynamic measurements
            measurement_breakdown = result_data.get('measurement_breakdown')
            if measurement_breakdown:
                dynamic_measurements = measurement_breakdown.get('dynamic_measurements', {})
                
                for generator_name, measurement in dynamic_measurements.items():
                    if measurement and measurement.get('success', False):
                        sensitivity_val = measurement.get('sensitivity', 'N/A')
                        if isinstance(sensitivity_val, (int, float)):
                            print(f"    ✅ Dynamic {generator_name}: {sensitivity_val:.4f}")
                        else:
                            print(f"    ✅ Dynamic {generator_name}: {sensitivity_val}")
                    else:
                        error_msg = measurement.get('error', 'Unknown error') if measurement else 'No measurement data'
                        error_msg = str(error_msg)[:50]
                        print(f"    ❌ Dynamic {generator_name}: {error_msg}")
                        
                # Report static measurements  
                static_measurements = measurement_breakdown.get('static_measurements', {})
                for estimator_name, measurement in static_measurements.items():
                    if measurement and measurement.get('success', False):
                        sensitivity_val = measurement.get('sensitivity', 'N/A')
                        range_val = measurement.get('range', 'N/A')
                        if isinstance(sensitivity_val, (int, float)):
                            print(f"    ✅ Static {estimator_name}: {sensitivity_val:.4f} (range: {range_val})")
                        else:
                            print(f"    ✅ Static {estimator_name}: {sensitivity_val} (range: {range_val})")
                    else:
                        error_msg = measurement.get('error', 'Unknown error') if measurement else 'No measurement data'
                        error_msg = str(error_msg)[:50]
                        print(f"    ❌ Static {estimator_name}: {error_msg}")
            else:
                print(f"    ⚠️  No measurement breakdown available")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[approach_name] = {
                'success': False,
                'approach': approach_name,
                'error': str(e)
            }
    
    return results

def main(args):
    """Main execution function."""
    print("🚀 Running Combined A-BB Debiasing Analysis")
    print("=" * 60)
    
    # Map judge datasets to appropriate judge models
    JUDGE_MAPPING = {
        "QwQ-32B-setting1": "qwq-32b-gguf",
        "DeepSeek-R1-32B-setting1": "deepseek-r1-32b-gguf", 
        "DeepSeek-R1-32B-setting2": "deepseek-r1-32b-gguf",
        "DeepSeek-R1-32B-setting3": "deepseek-r1-32b-gguf",
        "GPT-3.5-Turbo-0125-setting1": "gpt-3.5-turbo",
        "GPT-4o-mini-0718-setting1": "gpt-4o-mini",
    }
    
    # Base path for score data (from CLI)
    base_path = Path(args.data_path)
    
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
    
    # Process each judge
    overall_results = {}
    successful_analyses = 0
    
    for judge_name, judge_dir in judge_dirs.items():
        print(f"\n📊 Processing {judge_name}...")
        print("-" * 60)
        
        try:
            # Load and prepare real evaluation data
            df = load_and_prepare_score_data(judge_name, base_path)
            
            # Determine which judge to use for this dataset
            if judge_name not in JUDGE_MAPPING:
                raise ValueError(f"No judge mapping found for dataset '{judge_name}'. Available mappings: {list(JUDGE_MAPPING.keys())}")
            
            judge_to_use = JUDGE_MAPPING[judge_name]
            print(f"  🔧 Using judge: {judge_to_use} for dataset: {judge_name}")
            
            # Run Combined A-BB analysis
            analysis_results = run_combined_abb_analysis(
                df, judge_name, real_judge_name=judge_to_use
            )
            
            # Store results
            overall_results[judge_name] = {
                'data_source': str(judge_dir),
                'n_samples': len(df),
                'approaches': analysis_results
            }
            
            successful_analyses += 1
            print(f"✅ Completed Combined A-BB analysis for {judge_name}")
            
        except Exception as e:
            print(f"❌ Failed to process {judge_name}: {e}")
            overall_results[judge_name] = {
                'error': str(e),
                'data_source': str(judge_dir)
            }
    
    # Save overall results
    output_file = base_path / "combined_abb_analysis_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print()
    print("📊 Combined A-BB Analysis Summary:")
    print("=" * 60)
    print(f"Total datasets found: {len(judge_dirs)}")
    print(f"Successfully processed: {successful_analyses}")
    print(f"Failed: {len(judge_dirs) - successful_analyses}")
    
    # Analyze approach success rates
    approach_success = {}
    for judge_results in overall_results.values():
        if 'approaches' in judge_results:
            for approach_name, approach_result in judge_results['approaches'].items():
                if approach_name not in approach_success:
                    approach_success[approach_name] = {'success': 0, 'total': 0}
                approach_success[approach_name]['total'] += 1
                if approach_result.get('success', False):
                    approach_success[approach_name]['success'] += 1
    
    print()
    print("Approach Success Rates:")
    for approach, counts in approach_success.items():
        rate = (counts['success'] / counts['total']) * 100 if counts['total'] > 0 else 0
        print(f"  {approach}: {counts['success']}/{counts['total']} ({rate:.1f}%)")
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return 0 if successful_analyses > 0 else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Combined A-BB Debiasing Analysis")
    parser.add_argument("--data-path", "-d", type=str, required=True,
                        help="Base path to Arena-Hard evaluation data (judge folders with base_processed)")
    parser.add_argument("--test", action="store_true", 
                        help="Run in test mode with limited samples (fast for debugging)")
    parser.add_argument("--test-samples", type=int, default=5,
                        help="Number of samples to use in test mode (default: 5)")
    
    args = parser.parse_args()
    
    # Set global test mode
    if args.test:
        print(f"🧪 Running in TEST MODE with {args.test_samples} samples")
        os.environ['TEST_MODE'] = 'true'
        os.environ['TEST_SAMPLES'] = str(args.test_samples)
    
    exit_code = main(args)
    exit(exit_code)
