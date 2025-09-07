#!/usr/bin/env python3
"""
Measure and cache judge sensitivity profiles for efficient bias-bounded evaluation.

This script pre-computes expensive sensitivity measurements (particularly formatting sensitivity)
that can be reused across multiple datasets. This separates measurement from mechanism application,
improving performance and flexibility.
"""

import sys
import os
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
from differential_debiasing.interfaces.arena_hard_utils import find_sample_data as find_arena_sample_data


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


# (find_sample_data moved to differential_debiasing.interfaces.arena_hard_utils)


def measure_formatting_sensitivity(judge_name: str,
                                 sample_data: pd.DataFrame,
                                 num_neighbors: int = 40,
                                 target_samples: int = 20,
                                 cost_budget_usd: float = 10.0,
                                 prefer_existing_scores: bool = False,
                                 disable_transforms: bool = False,
                                 oumi_retries: int = 3) -> Dict[str, Any]:
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
            cache_responses=True,
            # Control reuse of cached baseline scores via CLI flag
            prefer_existing_scores=prefer_existing_scores,
            n_retries=oumi_retries
        )
        print(f"✅ Created judge function for {judge_name}")
        
        # Sample subset for measurement
        measurement_samples = sample_data.sample(n=min(target_samples, len(sample_data)), random_state=42)
        print(f"🎲 Using {len(measurement_samples)} samples for measurement")
        
        # Create formatting neighbor generator
        # Prefer a lightweight rephrase model via OUMI_REPHRASE_CONFIG; otherwise reuse the judge engine
        rephrase_cfg = os.getenv('OUMI_REPHRASE_CONFIG')
        shared_engine = None
        try:
            # Force-create engine now so we can reuse it if needed
            if hasattr(judge_function, 'interface'):
                shared_engine = judge_function.interface._get_inference_engine()
        except Exception:
            shared_engine = None
        if rephrase_cfg:
            print(f"🧠 Using rephrase engine from OUMI_REPHRASE_CONFIG: {rephrase_cfg}")
        else:
            if shared_engine is not None:
                print("♻️ Reusing judge engine for rephrasing to avoid double-loading the model")
            else:
                print("⚠️ No OUMI_REPHRASE_CONFIG set and judge engine unavailable; rephrasing may instantiate a new model")

        formatting_generator = FormattingNeighborGenerator(
            text_fields=['answer_a', 'answer_b'],
            formatting_types=['rephrasing'],
            single_field=True,
            disable_transforms=disable_transforms,
            random_seed=42,
            # If a specific rephrase config is provided, use it; else share the judge engine
            oumi_config_path=(rephrase_cfg if rephrase_cfg else None),
            oumi_shared_engine=(None if rephrase_cfg else shared_engine),
            oumi_retries=oumi_retries,
            rephrase_temperature=0.3
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
        
        # Get sensitivity estimate (RMS in the dataset's numeric score space, e.g., 1..5 Likert)
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
        
        # Capture baseline judgments from the estimator
        baseline_scores = []
        try:
            baseline_scores = [float(x) for x in np.asarray(sensitivity_estimator._original_judgments).flatten()]
        except Exception:
            baseline_scores = []

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
            'formatting_samples': formatting_samples,
            'baseline_scores': baseline_scores,
            # Explicit representation metadata for clarity in downstream reporting
            'representation': {
                'type': 'likert_numeric',
                'min_score': 1.0,
                'max_score': 5.0
            },
            # Tag whether transforms were disabled (this denotes intrinsic jitter runs)
            'disable_transforms': bool(disable_transforms)
        }
        
        label = "intrinsic jitter (RMS)" if disable_transforms else "formatting sensitivity (RMS)"
        print(f"✅ {label} measured: {sensitivity_value:.4f} (95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}])")
        
        return measurement_result
        
    except Exception as e:
        print(f"❌ Failed to measure formatting sensitivity for {judge_name}: {e}")
        return {
            'error': str(e),
            'measurement_date': datetime.now(timezone.utc).isoformat()
        }


def _infer_dataset_id(sample_data: pd.DataFrame) -> str:
    """Infer a dataset identifier from Arena-Hard sample_data source_dir.

    Uses the parent directory name of the base_processed folder, e.g.,
    <...>/GPT-4o-mini-0718-setting1/base_processed -> GPT-4o-mini-0718-setting1
    Falls back to 'unknown_dataset' if not resolvable.
    """
    try:
        if 'source_dir' in sample_data.columns:
            parents = [Path(p).resolve().parent.name for p in sample_data['source_dir'].dropna().unique().tolist()]
            parents = [p for p in parents if p]
            if parents:
                # If multiple, pick the most common
                from collections import Counter
                return Counter(parents).most_common(1)[0][0]
    except Exception:
        pass
    return 'unknown_dataset'


def save_sensitivity_profile(judge_name: str,
                           measurement_result: Dict[str, Any],
                           output_dir: Path,
                           sample_data: Optional[pd.DataFrame] = None) -> Path:
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
    formatting_samples = measurement_result.pop('formatting_samples', [])

    # Determine dataset id if available
    dataset_id = _infer_dataset_id(sample_data) if sample_data is not None else 'unknown_dataset'

    # Decide which section to populate based on disable_transforms flag
    is_intrinsic = bool(measurement_result.get('disable_transforms', False))
    profile_key = 'intrinsic_sensitivity' if is_intrinsic else 'formatting_sensitivity'
    profile_version = '1.1'
    
    profile_data = {
        'judge_name': judge_name,
        'profile_version': profile_version,
        'created_date': datetime.now(timezone.utc).isoformat(),
        profile_key: measurement_result,
        'measurement_metadata': {
            'script_version': '1.1',
            'measurement_approach': 'intrinsic_jitter' if is_intrinsic else 'arena_hard_content_perturbation',
            'neighbor_types': ['none'] if is_intrinsic else ['rephrasing'],
            'notes': 'Intrinsic jitter measured with transforms disabled' if is_intrinsic else 'Formatting sensitivity measured using Arena-Hard content perturbations',
            'dataset_id': dataset_id
        }
    }
    
    # Save main profile
    # If intrinsic, organize under dataset-specific subdirectory to ensure dataset+judge pairing
    if is_intrinsic and dataset_id != 'unknown_dataset':
        output_dir = output_dir / dataset_id
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_file = output_dir / f"{judge_name}_intrinsic_profile.json"
    else:
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
    parser.add_argument("--neighbors", "-n", type=int, default=40,
                        help="Number of neighbors to generate")
    parser.add_argument("--budget", "-b", type=float, default=10.0,
                        help="Cost budget in USD for API calls")
    parser.add_argument("--oumi-retries", type=int, default=3,
                        help="Number of retries for Oumi inference (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be measured without actually running")
    parser.add_argument("--reuse-baseline", action="store_true",
                        help="Reuse precomputed Arena-Hard baseline judgments instead of recomputing")
    parser.add_argument("--disable-transforms", action="store_true",
                        help="Disable all formatting transforms (neighbors identical to baseline)")
    
    args = parser.parse_args()
    
    print("🎯 Judge Sensitivity Measurement")
    print("=" * 50)
    print(f"Judge: {args.judge}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Samples: {args.samples}")
    print(f"Neighbors: {args.neighbors}")
    print(f"Budget: ${args.budget}")
    print(f"Oumi retries: {args.oumi_retries}")
    print(f"Reuse baseline: {'Yes' if args.reuse_baseline else 'No'}")
    print(f"Disable transforms: {'Yes' if args.disable_transforms else 'No'}")
    
    if args.dry_run:
        print("\n🧪 DRY RUN MODE - No measurements will be performed")
    
    # Validate inputs
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"❌ Data path does not exist: {data_path}")
        return 1
    
    output_dir = Path(args.output_dir)
    
    # Check if profile already exists (account for intrinsic vs formatting and dataset scoping)
    # Compute a quick dataset id guess for file naming
    try:
        temp_samples = find_arena_sample_data(Path(args.data_path), judge_name=args.judge, max_samples=5)
        dataset_id = _infer_dataset_id(temp_samples)
    except Exception:
        dataset_id = 'unknown_dataset'
    if args.disable_transforms and dataset_id != 'unknown_dataset':
        candidate = output_dir / dataset_id / f"{args.judge}_intrinsic_profile.json"
    else:
        candidate = output_dir / f"{args.judge}_sensitivity_profile.json"
    if candidate.exists():
        print(f"⚠️  Sensitivity profile already exists: {candidate}")
        response = input("Overwrite existing profile? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return 0
    
    try:
        # Find sample data
        sample_data = find_arena_sample_data(data_path, judge_name=args.judge, max_samples=args.samples * 3)
        
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
            cost_budget_usd=args.budget,
            prefer_existing_scores=args.reuse_baseline,
            disable_transforms=args.disable_transforms,
            oumi_retries=args.oumi_retries
        )
        
        # Compute Hamming-1 sensitivity using baseline scores only (no extra judge calls)
        try:
            baseline_scores = np.asarray(formatting_result.get('baseline_scores', [])).flatten()
            if baseline_scores.size == 0:
                raise ValueError("baseline_scores missing")
            df_scores = pd.DataFrame({'score': baseline_scores})
            hamming_estimator = ABBSensitivity(
                judge_function=None,
                neighbor_generator='hamming',
                num_neighbors=args.neighbors,
                target_samples=len(df_scores),
                random_seed=42
            )
            hamming_estimator.fit(df_scores)
            hamming_value = float(hamming_estimator.estimate())
            print(f"📏 Hamming-1 sensitivity: {hamming_value:.4f}")
        except Exception as e:
            print(f"⚠️  Could not compute Hamming-1 sensitivity from baseline scores: {e}")
            hamming_value = None

        # Augment results with combined average if available
        if hamming_value is not None:
            formatting_result['hamming_sensitivity'] = {
                'value': hamming_value,
                'neighbors_generated': int(args.neighbors)
            }
            formatting_result['combined_average_sensitivity'] = float((formatting_result['value'] + hamming_value) / 2.0)

        # Save sensitivity profile
        profile_file = save_sensitivity_profile(
            judge_name=args.judge,
            measurement_result=formatting_result,
            output_dir=output_dir,
            sample_data=sample_data
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
            if 'hamming_sensitivity' in formatting_result:
                print(f"   Hamming-1 sensitivity: {formatting_result['hamming_sensitivity']['value']:.4f}")
                print(f"   Combined average (fmt + ham)/2: {formatting_result['combined_average_sensitivity']:.4f}")
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
