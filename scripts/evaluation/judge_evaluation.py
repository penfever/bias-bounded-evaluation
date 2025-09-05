#!/usr/bin/env python3
"""
Judge Evaluation Debiasing Analysis for Arena-Hard-Auto.
This script applies bias-bounded debiasing to individual judge evaluations 
and generates new model rankings using the existing analysis pipeline.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
import glob
import subprocess
import shutil

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

def find_judge_datasets(base_path: Path) -> Dict[str, Path]:
    """Find all judge datasets with base_processed directories."""
    datasets = {}
    
    # Judge settings to process
    judge_patterns = [
        "DeepSeek-R1-32B-setting*",
        "GPT-3.5-Turbo-0125-setting*", 
        "GPT-4o-mini-0718-setting*",
        "QwQ-32B-setting*"
    ]
    
    for pattern in judge_patterns:
        for judge_dir in base_path.glob(pattern):
            if judge_dir.is_dir():
                base_processed_dir = judge_dir / "base_processed"
                if base_processed_dir.exists() and any(base_processed_dir.glob("*.jsonl")):
                    datasets[judge_dir.name] = judge_dir
                    print(f"Found judge dataset: {judge_dir.name}")
    
    return datasets

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

def get_reverse_score_mapping():
    """Get reverse mapping from numeric scores to comparative judgments."""
    # Create reverse mapping from numeric to judgment strings
    # For ties at score boundaries, prefer simpler forms
    reverse_mapping = {
        1.0: 'A>>B',
        2.0: 'A>B',
        3.0: 'A=B',
        4.0: 'B>A', 
        5.0: 'B>>A'
    }
    return reverse_mapping

def numeric_to_judgment(score: float) -> str:
    """Convert a numeric score back to comparative judgment string with rounding."""
    # Round to nearest integer, then map back
    rounded_score = max(1.0, min(5.0, round(score)))  # Clamp to [1,5] range
    reverse_mapping = get_reverse_score_mapping()
    return reverse_mapping[rounded_score]

def extract_scores_from_evaluations(evaluations: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Extract all scores from evaluations using Arena-Hard-Auto Likert scale transformations."""
    # Score mapping for conversion from comparative judgments to 5-point Likert scale
    # From arena-hard-auto/factor_analysis.py
    score_mapping = get_score_mapping()
    
    all_scores = []
    
    for model_name, model_evaluations in evaluations.items():
        for eval_data in model_evaluations:
            question_id = eval_data.get('question_id', '')
            games = eval_data.get('games', [])
            
            for game in games:
                # Extract judgment text to look for score patterns
                judgment = game.get('judgment', '')
                
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

def apply_debiasing_to_evaluations(evaluations: Dict[str, List[Dict]], 
                                  judge_name: str, save_raw: bool = True) -> Dict[str, Any]:
    """Apply debiasing to all judge evaluations."""
    print(f"\nApplying debiasing to {judge_name} evaluations...")
    
    # Extract scores for debiasing analysis
    scores_df = extract_scores_from_evaluations(evaluations)
    
    if len(scores_df) < 10:
        raise ValueError(f"Not enough evaluations for debiasing ({len(scores_df)})")
    
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
            # Initialize debiaser
            debiaser = DifferentialDebias(
                tau=approach_config['config'].tau,
                delta=approach_config['config'].delta,
                sensitivity_estimator=approach_config['estimator'],
                use_average_case=approach_config['config'].use_average_case,
                random_seed=42
            )
            
            # Fit the debiaser on all evaluation data
            debiaser.fit(scores_df)
            
            # Transform individual factor scores
            transformed_evaluations = {}
            
            for model_name, model_evaluations in evaluations.items():
                transformed_model_evaluations = []
                
                for eval_data in model_evaluations:
                    transformed_eval = json.loads(json.dumps(eval_data))  # Deep copy
                    
                    # Transform scores in games
                    if 'games' in transformed_eval:
                        for game in transformed_eval['games']:
                            # Extract current scores for this evaluation
                            current_scores = []
                            factor_names = []
                            
                            for factor in ['correctness_score', 'completeness_score', 'safety_score', 
                                          'conciseness_score', 'style_score']:
                                if factor in game:
                                    try:
                                        score_val = game[factor]
                                        score_mapping = get_score_mapping()
                                        # Convert judgment string to numeric using score mapping
                                        if isinstance(score_val, str) and score_val.strip():
                                            numeric_score = score_mapping.get(score_val.strip(), 3)  # Default to 3 (tie)
                                            current_scores.append(float(numeric_score))
                                            factor_names.append(factor)
                                        elif isinstance(score_val, (int, float)):
                                            current_scores.append(float(score_val))
                                            factor_names.append(factor)
                                    except (ValueError, TypeError):
                                        pass
                            
                            # Transform the scores if we have any
                            if current_scores:
                                current_scores_array = np.array(current_scores)
                                # Ensure scores are in 1-5 range (Likert scale)
                                current_scores_array = np.clip(current_scores_array, 1.0, 5.0)
                                transformed_scores = debiaser.transform(current_scores_array)
                                
                                # Update the game with transformed scores AND replace original judgments
                                for i, factor_name in enumerate(factor_names):
                                    debiased_score = float(transformed_scores[i])
                                    # Store numeric debiased score for reference
                                    game[f"{factor_name}_debiased"] = debiased_score
                                    # Replace original comparative judgment with debiased version
                                    game[factor_name] = numeric_to_judgment(debiased_score)
                            
                            # Transform overall score if present
                            if 'score' in game:
                                try:
                                    score_val = game['score']
                                    score_mapping = get_score_mapping()
                                    original_numeric = None
                                    
                                    # Convert judgment string to numeric using score mapping
                                    if isinstance(score_val, str) and score_val.strip():
                                        original_numeric = float(score_mapping.get(score_val.strip(), 3))
                                    elif isinstance(score_val, (int, float)):
                                        original_numeric = float(score_val)
                                    
                                    if original_numeric is not None:
                                        original_score = np.clip(original_numeric, 1.0, 5.0)
                                        transformed_score = debiaser.transform(np.array([original_score]))[0]
                                        debiased_score = float(transformed_score)
                                        game['score_debiased'] = debiased_score
                                        # Replace original comparative judgment with debiased version
                                        game['score'] = numeric_to_judgment(debiased_score)
                                except (ValueError, TypeError):
                                    pass
                    
                    transformed_model_evaluations.append(transformed_eval)
                
                transformed_evaluations[model_name] = transformed_model_evaluations
            
            # Get diagnostics
            diagnostics = debiaser.get_diagnostics()
            bias_bounds = debiaser.get_bias_bounds(len(scores_df))
            
            # Calculate validation metrics on overall scores
            original_overall = scores_df['overall_score'].values
            transformed_overall = debiaser.transform(original_overall)
            validation = debiaser.validate_effectiveness(original_overall, transformed_overall)
            
            # Store results
            result_data = {
                'success': True,
                'approach_description': approach_config['description'],
                'parameters': {
                    'tau': approach_config['config'].tau,
                    'delta': approach_config['config'].delta,
                    'estimator': approach_config['estimator']
                },
                'evaluation_stats': {
                    'n_evaluations': len(scores_df),
                    'n_models': len(evaluations),
                    'original_score_range': [float(original_overall.min()), float(original_overall.max())],
                    'transformed_score_range': [float(transformed_overall.min()), float(transformed_overall.max())]
                },
                'bias_analysis': {
                    'bias_sensitivity': diagnostics.get('bias_sensitivity', 0.0),
                    'noise_std': bias_bounds['noise_std'],
                    'bias_bound_tau': bias_bounds['tau'],
                    'bias_bound_delta': bias_bounds['delta'],
                    'protection_guarantee': bias_bounds['protection_guarantee']
                },
                'effectiveness_metrics': {
                    'score_correlation_preserved': validation['correlation'],
                    'mean_absolute_score_change': validation['mean_absolute_difference'],
                    'variance_ratio': validation['variance_ratio'],
                    'signal_preservation': validation['signal_preservation'],
                    'noise_level': validation['noise_level']
                },
                'detailed_diagnostics': diagnostics.get('sensitivity_estimator', {}),
                'transformed_evaluations': transformed_evaluations
            }
            
            # Add raw scores if requested
            if save_raw:
                result_data['raw_score_analysis'] = {
                    'original_scores': original_overall.tolist(),
                    'transformed_scores': transformed_overall.tolist(),
                    'score_differences': (transformed_overall - original_overall).tolist()
                }
            
            results[approach_name] = result_data
            
            print(f"    ✅ Success! Bias sensitivity: {diagnostics.get('bias_sensitivity', 0):.4f}")
            print(f"    📊 Score correlation preserved: {validation['correlation']:.3f}")
            print(f"    🔧 Noise level: {validation['noise_level']:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            results[approach_name] = {
                'success': False,
                'error': str(e),
                'approach_description': approach_config['description']
            }
    
    return results

def save_debiased_evaluations(results: Dict[str, Any], judge_dir: Path, judge_name: str):
    """Save debiased evaluations as new JSONL files."""
    print(f"\nSaving debiased evaluations for {judge_name}...")
    
    for approach_name, result in results.items():
        if result.get('success', False) and 'transformed_evaluations' in result:
            # Create new directory for debiased data
            debiased_dir = judge_dir / f"base_processed_debiased_{approach_name}"
            debiased_dir.mkdir(exist_ok=True)
            
            transformed_evaluations = result['transformed_evaluations']
            
            # Save each model's transformed evaluations
            for model_name, model_evaluations in transformed_evaluations.items():
                output_file = debiased_dir / f"{model_name}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for eval_data in model_evaluations:
                        f.write(json.dumps(eval_data, ensure_ascii=False) + '\n')
                
                print(f"  📝 Saved {len(model_evaluations)} evaluations for {model_name} ({approach_name})")
    
    # Save analysis results
    results_dir = judge_dir / "debiasing_results"
    results_dir.mkdir(exist_ok=True)
    
    # Remove large transformed_evaluations from results for JSON saving
    json_results = {}
    for approach_name, result in results.items():
        json_result = result.copy()
        if 'transformed_evaluations' in json_result:
            del json_result['transformed_evaluations']  # Too large for JSON
        json_results[approach_name] = json_result
    
    results_file = results_dir / f"{judge_name}_evaluation_debiasing_results.json"
    serializable_results = convert_numpy(json_results)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved analysis results to {results_file}")

def run_analysis_pipeline(judge_dir: Path, approach_name: str) -> bool:
    """Run the flexible analysis pipeline on debiased data."""
    print(f"Running analysis pipeline for {approach_name}...")
    
    debiased_dir = judge_dir / f"base_processed_debiased_{approach_name}"
    if not debiased_dir.exists():
        print(f"  ❌ Debiased directory not found: {debiased_dir}")
        return False
    
    # Arena-hard-auto directory (using local copy in examples)
    arena_hard_auto_dir = Path(__file__).parent.parent.parent / "examples" / "arena-hard-auto"
    
    if not arena_hard_auto_dir.exists():
        print(f"  ❌ Arena-hard-auto directory not found: {arena_hard_auto_dir}")
        return False
    
    try:
        judge_name = judge_dir.name
        output_dir = judge_dir / f"tables_debiased_{approach_name}"
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        print(f"  📂 Input directory: {debiased_dir}")
        print(f"  📂 Output directory: {output_dir}")
        print(f"  📁 Processing {len(list(debiased_dir.glob('*.jsonl')))} JSONL files")
        
        # Run the fast rankings-only analysis pipeline
        run_analysis_script = arena_hard_auto_dir / "run_analysis_rankings_only.sh"
        
        if run_analysis_script.exists():
            # Make script executable
            subprocess.run(f'chmod +x "{run_analysis_script}"', shell=True, check=True)
            
            # Execute the rankings-only analysis script
            cmd = (f'cd "{arena_hard_auto_dir}" && '
                   f'source ~/.zshrc && conda activate oumi && '
                   f'bash run_analysis_rankings_only.sh '
                   f'--input-dir "{debiased_dir}" '
                   f'--output-dir "{output_dir}" '
                   f'--judge-name "{judge_name}" '
                   f'--suffix "{approach_name}"')
            
            print(f"  🔧 Running: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)  # 10 min timeout (much faster now)
            
            if result.returncode == 0:
                print(f"  ✅ Analysis pipeline completed successfully")
                print(f"  📊 Results saved to {output_dir}")
                return True
            else:
                print(f"  ❌ Analysis pipeline failed:")
                print(f"    stdout: {result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout}")
                print(f"    stderr: {result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr}")
        else:
            print(f"  ❌ run_analysis_flexible.sh not found: {run_analysis_script}")
        
    except Exception as e:
        print(f"  ❌ Error running analysis pipeline: {e}")
    
    return False

def generate_ranking_comparisons(judge_dir: Path, judge_name: str):
    """Generate CSV comparisons between original and debiased rankings."""
    print(f"\nGenerating ranking comparisons for {judge_name}...")
    
    # Find original ranking file
    original_tables_dir = judge_dir / "tables" / "factor_scores_updated_cis"
    if not original_tables_dir.exists():
        original_tables_dir = judge_dir / "tables" / "factor_scores_original_cis"
    
    original_ranking_file = None
    if original_tables_dir.exists():
        for csv_file in original_tables_dir.glob("*score_factor*.csv"):
            original_ranking_file = csv_file
            break
    
    if not original_ranking_file:
        print(f"  ⚠️ No original ranking file found")
        return
    
    try:
        # Load original rankings
        original_df = pd.read_csv(original_ranking_file)
        original_df = original_df.sort_values('score', ascending=False).reset_index(drop=True)
        original_df['original_rank'] = original_df.index + 1
        
        comparisons_dir = judge_dir / "ranking_comparisons"
        comparisons_dir.mkdir(exist_ok=True)
        
        # Save original ranking
        original_output = comparisons_dir / f"{judge_name}_original_ranking.csv"
        original_df[['model', 'score', 'original_rank']].to_csv(original_output, index=False)
        print(f"  📊 Saved original ranking: {original_output}")
        
        # Process each debiased approach
        approaches = ['psychometric_reliability', 'schematic_adherence', 'combined_balanced']
        
        all_comparisons = []
        
        for approach in approaches:
            debiased_tables_dir = judge_dir / f"tables_debiased_{approach}" / "factor_scores_updated_cis"
            if not debiased_tables_dir.exists():
                debiased_tables_dir = judge_dir / f"tables_debiased_{approach}" / "factor_scores_original_cis"
            
            if debiased_tables_dir.exists():
                debiased_ranking_file = None
                for csv_file in debiased_tables_dir.glob("*score_factor*.csv"):
                    debiased_ranking_file = csv_file
                    break
                
                if debiased_ranking_file:
                    try:
                        debiased_df = pd.read_csv(debiased_ranking_file)
                        debiased_df = debiased_df.sort_values('score', ascending=False).reset_index(drop=True)
                        debiased_df['debiased_rank'] = debiased_df.index + 1
                        
                        # Merge with original rankings
                        comparison_df = pd.merge(
                            original_df[['model', 'score', 'original_rank']],
                            debiased_df[['model', 'score', 'debiased_rank']],
                            on='model',
                            suffixes=('_original', '_debiased')
                        )
                        
                        comparison_df['rank_change'] = comparison_df['original_rank'] - comparison_df['debiased_rank']
                        comparison_df['score_change'] = comparison_df['score_debiased'] - comparison_df['score_original']
                        comparison_df['approach'] = approach
                        
                        # Save individual comparison
                        comparison_output = comparisons_dir / f"{judge_name}_{approach}_ranking_comparison.csv"
                        comparison_df.to_csv(comparison_output, index=False)
                        print(f"  📊 Saved {approach} comparison: {comparison_output}")
                        
                        all_comparisons.append(comparison_df)
                        
                    except Exception as e:
                        print(f"  ⚠️ Error processing {approach}: {e}")
        
        # Create combined comparison
        if all_comparisons:
            combined_df = pd.concat(all_comparisons, ignore_index=True)
            combined_output = comparisons_dir / f"{judge_name}_all_approaches_ranking_comparison.csv"
            combined_df.to_csv(combined_output, index=False)
            print(f"  📊 Saved combined comparison: {combined_output}")
            
            # Generate summary statistics
            summary_stats = []
            for approach in combined_df['approach'].unique():
                approach_df = combined_df[combined_df['approach'] == approach]
                stats = {
                    'approach': approach,
                    'mean_rank_change': approach_df['rank_change'].mean(),
                    'std_rank_change': approach_df['rank_change'].std(),
                    'max_rank_change': approach_df['rank_change'].abs().max(),
                    'mean_score_change': approach_df['score_change'].mean(),
                    'std_score_change': approach_df['score_change'].std(),
                    'models_improved': (approach_df['rank_change'] > 0).sum(),
                    'models_declined': (approach_df['rank_change'] < 0).sum(),
                    'models_unchanged': (approach_df['rank_change'] == 0).sum()
                }
                summary_stats.append(stats)
            
            summary_df = pd.DataFrame(summary_stats)
            summary_output = comparisons_dir / f"{judge_name}_ranking_comparison_summary.csv"
            summary_df.to_csv(summary_output, index=False)
            print(f"  📋 Saved summary statistics: {summary_output}")
            
    except Exception as e:
        print(f"  ❌ Error generating comparisons: {e}")

def main():
    """Main execution function."""
    print("🚀 Running Judge Evaluation Debiasing Analysis")
    print("=" * 70)
    
    # Base path for judge data
    base_path = Path("/Users/benfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Projects/oumi/llm-judge-oumi/data/InDepthAnalysis")
    
    if not base_path.exists():
        print(f"❌ Error: Base path does not exist: {base_path}")
        return 1
    
    # Find all judge datasets
    print("🔍 Searching for judge datasets...")
    datasets = find_judge_datasets(base_path)
    
    if not datasets:
        print("❌ No judge datasets found!")
        return 1
    
    print(f"✅ Found {len(datasets)} judge datasets")
    print()
    
    # Process each judge dataset
    overall_results = {}
    successful_analyses = 0
    
    for i, (judge_name, judge_dir) in enumerate(datasets.items(), 1):
        print(f"📊 [{i}/{len(datasets)}] Processing {judge_name}")
        print("-" * 60)
        
        try:
            # Load judge evaluations
            base_processed_dir = judge_dir / "base_processed"
            evaluations = load_judge_evaluations(base_processed_dir)
            
            if not evaluations:
                print(f"  ⚠️ No evaluations found in {base_processed_dir}")
                continue
            
            # Apply debiasing
            debiasing_results = apply_debiasing_to_evaluations(evaluations, judge_name)
            
            # Save debiased evaluations
            save_debiased_evaluations(debiasing_results, judge_dir, judge_name)
            
            # Run analysis pipeline for each successful approach
            successful_approaches = []
            for approach_name, result in debiasing_results.items():
                if result.get('success', False):
                    print(f"\n  🔄 Running analysis pipeline for {approach_name}...")
                    if run_analysis_pipeline(judge_dir, approach_name):
                        successful_approaches.append(approach_name)
            
            # Generate ranking comparisons
            if successful_approaches:
                generate_ranking_comparisons(judge_dir, judge_name)
            
            # Track overall results
            overall_results[judge_name] = {
                'status': 'success',
                'n_evaluations': sum(len(evals) for evals in evaluations.values()),
                'n_models': len(evaluations),
                'successful_approaches': successful_approaches,
                'debiasing_results': {k: {kk: vv for kk, vv in v.items() if kk != 'transformed_evaluations'} 
                                     for k, v in debiasing_results.items()}
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
    
    # Save overall summary
    summary_file = base_path / "judge_evaluation_debiasing_analysis_summary.json"
    serializable_results = convert_numpy(overall_results)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    # Print final summary
    print("🏆 Final Analysis Summary")
    print("=" * 50)
    print(f"📊 Total judge datasets: {len(datasets)}")
    print(f"✅ Successful analyses: {successful_analyses}")
    print(f"❌ Failed analyses: {len(datasets) - successful_analyses}")
    
    if successful_analyses > 0:
        print("\n📈 Results Summary:")
        for judge_name, results in overall_results.items():
            if results['status'] == 'success':
                print(f"  {judge_name}:")
                print(f"    - {results['n_evaluations']} evaluations across {results['n_models']} models")
                print(f"    - Successful approaches: {results['successful_approaches']}")
    
    print(f"\n💾 Results saved to: {base_path}")
    print(f"📋 Overall summary saved to: {summary_file}")
    
    return 0 if successful_analyses > 0 else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)