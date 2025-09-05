#!/usr/bin/env python3
"""
Measure and cache judge sensitivity profiles for efficient bias-bounded evaluation.

This script pre-computes expensive sensitivity measurements (particularly formatting sensitivity)
that can be reused across multiple datasets. This separates measurement from mechanism application,
improving performance and flexibility.
"""

import sys
import difflib
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
import argparse
from datetime import datetime, timezone

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from differential_debiasing.core.debias import DifferentialDebias
from differential_debiasing.core.config import ConfigManager
from differential_debiasing.interfaces.oumi_interface import create_oumi_judge_function
from differential_debiasing.sensitivity.abb_sensitivity import ABBSensitivity
from differential_debiasing.neighbors import FormattingNeighborGenerator


def get_judge_config_path(judge_name: str) -> Path:
    """Get the correct config path for a judge based on the directory structure."""
    base_config_dir = Path(__file__).parent.parent.parent / "configs" / "judges"
    
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
    
    raise FileNotFoundError(f"Judge config not found for '{judge_name}'. Expected at one of: {list(judge_mapping.values())}")


def find_sample_data(base_path: Path, max_samples: int = 50) -> pd.DataFrame:
    """
    Find sample evaluation data for sensitivity measurement.
    
    Parameters:
    -----------
    base_path : Path
        Base path to search for evaluation data
    max_samples : int
        Maximum number of samples to use for measurement
        
    Returns:
    --------
    pd.DataFrame : Sample data with question_id, model columns
    """
    print(f"🔍 Searching for sample data in {base_path}...")
    
    # Look for existing judge directories with base_processed data
    sample_data = []
    
    for judge_dir in base_path.glob("*-setting*"):
        if judge_dir.is_dir():
            base_processed_dir = judge_dir / "base_processed"
            if base_processed_dir.exists():
                print(f"  Found data source: {judge_dir.name}")
                
                # Load a few samples from JSONL files
                for jsonl_file in list(base_processed_dir.glob("*.jsonl"))[:3]:  # Limit to 3 models
                    model_name = jsonl_file.stem
                    
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            sample_count = 0
                            for line in f:
                                if sample_count >= max_samples // 3:  # Distribute samples across models
                                    break
                                    
                                line = line.strip()
                                if line:
                                    try:
                                        data = json.loads(line)
                                        question_id = data.get('question_id', f'q_{sample_count}')
                                        sample_data.append({
                                            'question_id': question_id,
                                            'model': model_name,
                                            'source_dir': str(base_processed_dir)
                                        })
                                        sample_count += 1
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        print(f"    ⚠️ Error reading {jsonl_file}: {e}")
                        continue
                
                # Break after finding first valid source
                if sample_data:
                    break
    
    if not sample_data:
        raise ValueError(f"No evaluation data found in {base_path}")
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Found {len(df)} sample evaluations across {df['model'].nunique()} models")
    return df


def measure_formatting_sensitivity(judge_name: str, 
                                 sample_data: pd.DataFrame,
                                 num_neighbors: int = 10,
                                 target_samples: int = 20,
                                 cost_budget_usd: float = 10.0) -> Dict[str, Any]:
    """
    Measure formatting sensitivity for a specific judge.
    
    Parameters:
    -----------
    judge_name : str
        Name of the judge to measure
    sample_data : pd.DataFrame
        Sample data with question_id, model, source_dir columns
    num_neighbors : int
        Number of formatting neighbors to generate
    target_samples : int
        Number of samples to use for measurement
    cost_budget_usd : float
        Budget for API calls (if using cloud judge)
        
    Returns:
    --------
    Dict[str, Any] : Measurement results with sensitivity value and metadata
    """
    print(f"📏 Measuring formatting sensitivity for {judge_name}...")
    
    try:
        # Create judge function
        judge_config_path = get_judge_config_path(judge_name)
        judge_function = create_oumi_judge_function(
            str(judge_config_path),
            cost_budget_usd=cost_budget_usd,
            cache_responses=True
        )
        print(f"✅ Created judge function for {judge_name}")
        
        # Sample subset for measurement
        measurement_samples = sample_data.sample(n=min(target_samples, len(sample_data)), random_state=42)
        print(f"🎲 Using {len(measurement_samples)} samples for measurement")
        
        # Create formatting neighbor generator
        formatting_generator = FormattingNeighborGenerator(
            text_fields=['answer_a', 'answer_b'],
            formatting_types=['whitespace', 'capitalization', 'punctuation'],
            single_field=True,
            random_seed=42
        )
        
        # Create ABB sensitivity estimator
        sensitivity_estimator = ABBSensitivity(
            judge_function=judge_function,
            neighbor_generator=formatting_generator,
            num_neighbors=num_neighbors,
            target_samples=len(measurement_samples),
            random_seed=42
        )
        
        # Get base processed directory from first sample
        base_processed_dir = Path(measurement_samples.iloc[0]['source_dir'])
        
        print(f"🔧 Measuring sensitivity with {num_neighbors} neighbors on {len(measurement_samples)} samples...")
        
        # Fit the estimator
        sensitivity_estimator.fit(measurement_samples)
        
        # Get sensitivity estimate
        sensitivity_value = sensitivity_estimator.estimate()
        
        # Get diagnostic information
        diagnostics = sensitivity_estimator.get_diagnostics()
        
        # Save samples by reusing neighbors and scores already computed during A-BB
        formatting_samples = []
        try:
            from differential_debiasing.interfaces.arena_hard_utils import SCORE_TO_PAIRWISE
            neighbors_exps = sensitivity_estimator.get_sampled_neighbors()  # List[List[Dict]]
            neighbor_scores = sensitivity_estimator.get_neighbor_judgments()  # List[np.ndarray]
            original_scores = getattr(sensitivity_estimator, '_original_judgments', None)
            if neighbors_exps and neighbor_scores and original_scores is not None:
                # Take up to 5 experiments
                for exp_idx, (exp_contexts, exp_scores) in enumerate(zip(neighbors_exps[:5], neighbor_scores[:5])):
                    # Find the perturbed sample in this experiment
                    perturbed_ctx = next((c for c in exp_contexts if isinstance(c, dict) and c.get('is_formatting_neighbor')), None)
                    if not perturbed_ctx:
                        continue
                    idx = int(perturbed_ctx.get('perturbed_sample_index', -1))
                    if idx < 0 or idx >= len(original_scores) or idx >= len(exp_scores):
                        continue
                    orig_score = float(np.asarray(original_scores).flatten()[idx])
                    neigh_score = float(np.asarray(exp_scores).flatten()[idx])
                    # Build sample entry
                    # Determine which field actually changed and diff that field
                    original_a = perturbed_ctx.get('original_answer_a', '') or ''
                    original_b = perturbed_ctx.get('original_answer_b', '') or ''
                    original_q = perturbed_ctx.get('original_question', '') or ''
                    pert_a = perturbed_ctx.get('answer_a', '') or ''
                    pert_b = perturbed_ctx.get('answer_b', '') or ''
                    pert_q = perturbed_ctx.get('question', '') or ''

                    if pert_a != original_a:
                        changed_field = 'answer_a'
                        original_text = original_a
                        neighbor_text = pert_a
                    elif pert_b != original_b:
                        changed_field = 'answer_b'
                        original_text = original_b
                        neighbor_text = pert_b
                    elif pert_q != original_q:
                        changed_field = 'question'
                        original_text = original_q
                        neighbor_text = pert_q
                    else:
                        changed_field = 'unknown'
                        original_text = original_a
                        neighbor_text = pert_a
                    diff_lines = list(difflib.unified_diff(
                        original_text.splitlines(),
                        neighbor_text.splitlines(),
                        fromfile='original', tofile='neighbor', lineterm=''
                    ))
                    diff_text = "\n".join(diff_lines)

                    entry = {
                        'question_id': perturbed_ctx.get('question_id'),
                        'model': perturbed_ctx.get('model'),
                        'changed_field': changed_field,
                        'original_response': original_text,
                        'neighbors': [neighbor_text],
                        'diff': diff_text,
                        'original_judgment': {
                            'pairwise': SCORE_TO_PAIRWISE.get(orig_score),
                            'score': orig_score
                        },
                        'neighbor_judgments': [{
                            'pairwise': SCORE_TO_PAIRWISE.get(neigh_score),
                            'score': neigh_score
                        }]
                    }
                    formatting_samples.append(entry)
            else:
                print("⚠️ Warning: No neighbor experiments available to log samples.")
        except Exception as e:
            print(f"⚠️ Warning: Could not collect formatting samples: {e}")
            formatting_samples = []
        
        # Calculate confidence interval (simple bootstrap estimate)
        neighbor_differences = sensitivity_estimator.get_neighbor_differences()
        if neighbor_differences:
            # Simple confidence interval based on std of neighbor differences
            mean_diff = np.mean(neighbor_differences)
            std_diff = np.std(neighbor_differences)
            confidence_interval = [
                max(0, sensitivity_value - 1.96 * std_diff / np.sqrt(len(neighbor_differences))),
                sensitivity_value + 1.96 * std_diff / np.sqrt(len(neighbor_differences))
            ]
        else:
            confidence_interval = [sensitivity_value, sensitivity_value]
        
        # Gather cost information if available
        cost_info = {}
        if hasattr(judge_function, 'interface'):
            try:
                cost_summary = judge_function.interface.get_cost_summary()
                cost_info = cost_summary
            except:
                pass
        
        measurement_result = {
            'value': float(sensitivity_value),
            'confidence_interval': [float(ci) for ci in confidence_interval],
            'samples_used': int(len(measurement_samples)),
            'neighbors_generated': int(num_neighbors),
            'measurement_date': datetime.now(timezone.utc).isoformat(),
            'neighbor_differences': [float(d) for d in neighbor_differences],
            'mean_neighbor_difference': float(np.mean(neighbor_differences)) if neighbor_differences else 0.0,
            'std_neighbor_difference': float(np.std(neighbor_differences)) if neighbor_differences else 0.0,
            'cost_info': cost_info,
            'diagnostics': diagnostics,
            'formatting_samples': formatting_samples
        }
        
        print(f"✅ Formatting sensitivity measured: {sensitivity_value:.4f} (95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}])")
        
        return measurement_result
        
    except Exception as e:
        print(f"❌ Failed to measure formatting sensitivity for {judge_name}: {e}")
        return {
            'error': str(e),
            'measurement_date': datetime.now(timezone.utc).isoformat()
        }


def save_sensitivity_profile(judge_name: str, 
                           formatting_result: Dict[str, Any],
                           output_dir: Path) -> Path:
    """
    Save sensitivity profile to JSON file.
    
    Parameters:
    -----------
    judge_name : str
        Name of the judge
    formatting_result : Dict[str, Any]
        Formatting sensitivity measurement results
    output_dir : Path
        Directory to save profile files
        
    Returns:
    --------
    Path : Path to saved profile file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract formatting samples for separate file
    formatting_samples = formatting_result.pop('formatting_samples', [])
    
    profile_data = {
        'judge_name': judge_name,
        'profile_version': '1.0',
        'created_date': datetime.now(timezone.utc).isoformat(),
        'formatting_sensitivity': formatting_result,
        'measurement_metadata': {
            'script_version': '1.0',
            'measurement_approach': 'arena_hard_content_perturbation',
            'neighbor_types': ['whitespace', 'capitalization', 'punctuation'],
            'notes': 'Formatting sensitivity measured using Arena-Hard content perturbations'
        }
    }
    
    # Save main profile
    profile_file = output_dir / f"{judge_name}_sensitivity_profile.json"
    with open(profile_file, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)
    
    # Save formatting samples to separate file for inspection
    if formatting_samples:
        samples_file = output_dir / f"{judge_name}_formatting_samples.json"
        samples_data = {
            'judge_name': judge_name,
            'created_date': datetime.now(timezone.utc).isoformat(),
            'description': 'Sample formatting neighbor transformations for manual inspection',
            'samples': formatting_samples
        }
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
        print(f"📋 Formatting samples saved to: {samples_file}")
    
    print(f"💾 Sensitivity profile saved to: {profile_file}")
    return profile_file


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Measure and cache judge sensitivity profiles")
    parser.add_argument("--judge", "-j", type=str, required=True,
                        help="Judge name to measure (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet')")
    parser.add_argument("--data-path", "-d", type=str, 
                        default="/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis",
                        help="Base path to evaluation data")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="sensitivity_profiles",
                        help="Directory to save sensitivity profiles")
    parser.add_argument("--samples", "-s", type=int, default=20,
                        help="Number of samples to use for measurement")
    parser.add_argument("--neighbors", "-n", type=int, default=10,
                        help="Number of neighbors to generate")
    parser.add_argument("--budget", "-b", type=float, default=10.0,
                        help="Cost budget in USD for API calls")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be measured without actually running")
    
    args = parser.parse_args()
    
    print("🎯 Judge Sensitivity Measurement")
    print("=" * 50)
    print(f"Judge: {args.judge}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples: {args.samples}")
    print(f"Neighbors: {args.neighbors}")
    print(f"Budget: ${args.budget}")
    
    if args.dry_run:
        print("\n🧪 DRY RUN MODE - No measurements will be performed")
    
    # Validate inputs
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data path does not exist: {data_path}")
        return 1
    
    output_dir = Path(args.output_dir)
    
    # Check if profile already exists
    existing_profile = output_dir / f"{args.judge}_sensitivity_profile.json"
    if existing_profile.exists():
        print(f"⚠️  Sensitivity profile already exists: {existing_profile}")
        response = input("Overwrite existing profile? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return 0
    
    try:
        # Find sample data
        sample_data = find_sample_data(data_path, max_samples=args.samples * 3)
        
        if args.dry_run:
            print(f"\n🧪 Would measure formatting sensitivity for {args.judge}")
            print(f"   Using {len(sample_data)} samples from {sample_data['model'].nunique()} models")
            print(f"   Would generate {args.neighbors} neighbors per measurement")
            print(f"   Profile would be saved to: {output_dir / f'{args.judge}_sensitivity_profile.json'}")
            return 0
        
        # Measure formatting sensitivity
        formatting_result = measure_formatting_sensitivity(
            judge_name=args.judge,
            sample_data=sample_data,
            num_neighbors=args.neighbors,
            target_samples=args.samples,
            cost_budget_usd=args.budget
        )
        
        # Save sensitivity profile
        profile_file = save_sensitivity_profile(
            judge_name=args.judge,
            formatting_result=formatting_result,
            output_dir=output_dir
        )
        
        # Print summary
        print(f"\n📊 Measurement Summary:")
        print("=" * 30)
        if 'error' not in formatting_result:
            print(f"✅ Formatting sensitivity: {formatting_result['value']:.4f}")
            ci = formatting_result['confidence_interval']
            print(f"   95% Confidence interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"   Samples used: {formatting_result['samples_used']}")
            print(f"   Neighbors generated: {formatting_result['neighbors_generated']}")
        else:
            print(f"❌ Measurement failed: {formatting_result['error']}")
            return 1
        
        print(f"\n💾 Profile saved to: {profile_file}")
        print("✅ Measurement completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during measurement: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
