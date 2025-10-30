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
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from differential_debiasing.core.debias import DifferentialDebias
from differential_debiasing.core.sensitivity_profiles import collect_profile_info
from differential_debiasing.core.utils import (
    round_nested,
    calculate_effective_alpha,
    estimate_s_mu_norm,
    build_abb_precheck,
)
from differential_debiasing.interfaces import (
    find_judge_data_directories,
    load_and_prepare_score_data,
    get_dataset_judge_mapping,
    determine_score_scale,
)
from differential_debiasing.sensitivity import estimate_schematic_context_sensitivity
# Note: Oumi judge interface is imported lazily only when needed


APPROACH_CONFIG_PATH = Path(__file__).with_name("combined_abb_approaches.json")
try:
    with open(APPROACH_CONFIG_PATH, "r", encoding="utf-8") as _fh:
        DEFAULT_APPROACH_CONFIGS = json.load(_fh)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Approach configuration file missing: {APPROACH_CONFIG_PATH}. "
        "Please restore the configuration before running the analysis."
    ) from exc


def _combine_sensitivity(
    approach_name: str,
    component_values: Dict[str, float],
    extra_params: Dict[str, Any],
) -> float:
    """Derive the combined sensitivity value for a given approach."""
    if not component_values:
        return 0.0

    values = {k: float(v) for k, v in component_values.items()}
    arr = np.array(list(values.values()), dtype=float)

    if approach_name == "abb_formatting_only":
        return float(values.get("formatting", arr[0]))

    if approach_name == "combined_abb_conservative":
        return float(arr.max())

    if approach_name == "combined_abb_rms":
        return float(np.sqrt(np.mean(np.square(arr))))

    weights = extra_params.get("component_weights") if isinstance(extra_params, dict) else None
    if weights:
        weight_values: List[float] = []
        component_array: List[float] = []
        for name, val in values.items():
            weight = float(weights.get(name, 0.0))
            if weight > 0.0:
                weight_values.append(weight)
                component_array.append(val)
        if weight_values:
            weight_arr = np.array(weight_values, dtype=float)
            comp_arr = np.array(component_array, dtype=float)
            total = weight_arr.sum()
            if total > 0.0:
                normalized = weight_arr / total
                return float(np.dot(normalized, comp_arr))

    return float(arr.mean())


def _summarize_results(overall_results: Dict[str, Any]) -> Dict[str, Any]:
    """Condense full analysis outputs for the JSON summary."""
    summary: Dict[str, Any] = {}
    for judge_name, judge_data in overall_results.items():
        entry: Dict[str, Any] = {}
        if isinstance(judge_data, dict):
            if 'data_source' in judge_data:
                entry['data_source'] = judge_data['data_source']
            if 'n_samples' in judge_data:
                entry['n_samples'] = judge_data['n_samples']
            if 'error' in judge_data:
                entry['error'] = judge_data['error']

            approaches = judge_data.get('approaches')
            if isinstance(approaches, dict):
                summary_approaches: Dict[str, Any] = {}
                for name, data in approaches.items():
                    if isinstance(data, dict):
                        summary_approaches[name] = {
                            k: v
                            for k, v in data.items()
                            if k not in ['original_scores', 'debiased_scores', 'model_question_data']
                        }
                    else:
                        summary_approaches[name] = data
                entry['approaches'] = summary_approaches

        summary[judge_name] = entry
    return summary

def run_combined_abb_analysis(
    df: pd.DataFrame,
    judge_name: str,
    real_judge_name: str,
    strict_abb: bool = False,
    enable_shrinkage: bool = False,
    target_tau: Optional[float] = None,
    target_delta: Optional[float] = None,
    shrink_alpha: Optional[float] = None,
    shrink_center: str = "mean",
    abb_dimensionality: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Combined A-BB debiasing approaches on the judge data."""
    print(f"\nRunning Combined A-BB analysis for {judge_name}...")
    # Check for sensitivity profiles for the real judge, and dataset-scoped intrinsic jitter
    profile_info = collect_profile_info(real_judge_name, judge_name, verbose=True)
    
    ctx = profile_info.get('context_adjusted_rms') if profile_info else None
    if ctx is None:
        # If formatting is missing entirely, we cannot proceed safely
        raise RuntimeError(
            f"Formatting sensitivity profile missing for judge '{real_judge_name}'. Please generate it first."
        )
    print(f"🚀 Using Context-Adjusted RMS for formatting: {ctx:.4f}")

    # Establish a single, consistent score range from input data and reuse it
    score_min, score_max, score_range = determine_score_scale(df)
    print(f"   ↪︎ Using unified score scale: min={score_min:.3f}, max={score_max:.3f}, range={score_range:.3f}")

    # Compute context-aware schematic sensitivity (psychometric path deprecated)
    factor_columns = [col for col in df.columns if col.endswith('_score') and col != 'overall_score']

    # Schematic
    schematic_raw = 0.0
    schem_ctx = 0.0
    try:
        schematic_raw, schem_ctx = estimate_schematic_context_sensitivity(
            df,
            factor_columns=factor_columns or None,
            score_range=score_range,
            intrinsic_sensitivity=profile_info.get('intrinsic_sensitivity'),
        )
    except Exception as e:
        print(f"Warning: Schematic adherence estimation failed: {e}")

    print(f"   ↪︎ context-aware static: schematic={schem_ctx:.4f}")

    approaches = copy.deepcopy(DEFAULT_APPROACH_CONFIGS)
    
    results = {}
    
    # Progress bar over strategies
    try:
        from tqdm import tqdm as _tqdm
        _approach_iter = _tqdm(list(approaches.items()), total=len(approaches), desc="Combined A-BB strategies")
    except Exception:
        _approach_iter = approaches.items()
    
    for approach_name, approach_config in _approach_iter:
        print(f"  Testing {approach_name} strategy...")
        
        try:
            # Base parameters
            # If a global target_tau is provided, use it for all strategies
            effective_tau = float(target_tau) if target_tau is not None else float(approach_config.get('tau', 2.0))
            effective_delta = (
                float(target_delta)
                if target_delta is not None
                else float(approach_config.get('delta', 0.1))
            )

            init_params = {
                'tau': effective_tau,
                'delta': effective_delta,
                'sensitivity_estimator': approach_config['estimator'],
                'use_average_case': approach_config.get('use_average_case', True),
                'random_seed': 42,
            }

            if abb_dimensionality is not None:
                init_params['dimensionality'] = int(abb_dimensionality)

            # Shrinkage controls (from function parameters)
            if bool(enable_shrinkage):
                if shrink_alpha is not None:
                    init_params['shrink_alpha'] = float(shrink_alpha)
                if target_tau is not None:
                    init_params['target_tau'] = float(target_tau)
                init_params['shrink_center'] = str(shrink_center or 'mean')
            
            # Add extra parameters
            extra_params = approach_config.setdefault('extra_params', {})
            extra_params.setdefault('fixed_sensitivity_value', None)
            init_params.update(extra_params)

            # Compute combined fixed sensitivity for this approach
            component_values = {
                'formatting': ctx if ctx > 0 else 1e-6,
                'schematic': schem_ctx if schem_ctx > 0 else 1e-6,
            }
            combined_fixed = _combine_sensitivity(approach_name, component_values, extra_params)

            # No-op: if combined sensitivity is non-positive, skip mechanism and return original
            original_scores = df['overall_score'].values
            if combined_fixed <= 0.0:
                print(f"    ℹ️ No-op: combined sensitivity is {combined_fixed:.4f}; skipping debiaser")
                debiased_scores = original_scores.copy()
                diagnostics = {
                    'fitted': True,
                    'is_abb_mechanism': False,
                    'bias_sensitivity': combined_fixed
                }
                bias_bounds = {
                    'tau': init_params['tau'],
                    'delta': init_params['delta'],
                    'noise_std': 0.0
                }
                validation = {
                    'correlation': 1.0,
                    'mean_absolute_difference': 0.0,
                    'variance_ratio': 1.0,
                    'noise_level': 0.0
                }
            else:
                # A-BB precheck using shared normalization helpers
                delta_val = float(approach_config.get('delta', init_params.get('delta', 0.1)))
                tau_val = float(effective_tau)
                shrink_enabled = bool(enable_shrinkage)
                s_mu_norm = estimate_s_mu_norm(
                    len(df),
                    shrink_center=str(shrink_center),
                ) if shrink_enabled else 0.0
                alpha_hat = 1.0
                if shrink_enabled:
                    alpha_hat = calculate_effective_alpha(
                        bias_sensitivity=combined_fixed,
                        score_range=score_range,
                        tau=tau_val,
                        delta=delta_val,
                        shrink_alpha=float(shrink_alpha) if shrink_alpha is not None else None,
                        target_tau=float(target_tau) if target_tau is not None else None,
                    )
                precheck = build_abb_precheck(
                    tau=tau_val,
                    delta=delta_val,
                    sensitivity=combined_fixed,
                    score_range=score_range,
                    alpha=alpha_hat,
                    s_mu_norm=s_mu_norm,
                )
                normalized_tau = precheck.get('normalized_tau')
                normalized_sensitivity = precheck.get('normalized_sensitivity')
                tau_display = (
                    f"{float(normalized_tau):.4f}" if normalized_tau is not None else f"{precheck['tau']:.3f}"
                )
                sens_display = (
                    f"{float(normalized_sensitivity):.4f}"
                    if normalized_sensitivity is not None
                    else f"{precheck['effective_sensitivity']:.4f}"
                )
                print(
                    f"    A-BB precheck (normalized): tau={tau_display} (raw {precheck['tau']:.3f}), "
                    f"delta={delta_val:.3f}, α̂={alpha_hat:.3f}, Sμ̂={precheck['s_mu_norm']:.4f}, "
                    f"A={precheck['A_component']:.4f}, B={precheck['B_component']:.4f}, Δ̂eff={sens_display}, "
                    f"threshold={precheck['constraint_threshold']:.4f}, margin={precheck['margin']:.4f} "
                    f"(range={score_range:.3f})"
                )
                if strict_abb and float(precheck.get('margin', 0.0)) <= 0.0:
                    msg = (
                        f"Strict mode: A-BB precheck failed (tau={precheck['tau']:.6f} <= "
                        f"threshold={precheck['constraint_threshold']:.6f}); skipping strategy"
                    )
                    print(f"    ❌ {msg}")
                    results[approach_name] = {
                        'success': False,
                        'approach': approach_name,
                        'error': msg,
                        'diagnostics': {
                            'is_abb_mechanism': False,
                            'bias_sensitivity': combined_fixed
                        }
                    }
                    continue

                init_params['fixed_sensitivity_value'] = combined_fixed
                
                # Initialize debiaser
                debiaser = DifferentialDebias(**init_params)
                
                # Fit with factor columns
                factor_columns = [col for col in df.columns if col.endswith('_score') and col != 'overall_score']
                if factor_columns:
                    debiaser.fit(df, factor_columns=factor_columns)
                else:
                    debiaser.fit(df)

                # After fit, print actual shrinkage alpha and S_mu norm bound from diagnostics if available
                try:
                    diag = debiaser.get_diagnostics() or {}
                    sh = diag.get('shrinkage', {})
                    if sh:
                        print(f"    ↪︎ shrinkage: α={sh.get('alpha'):.3f}, Sμ_bound={diag.get('abb_constraint_validation', {}).get('normalized_sensitivity', None) if False else (getattr(debiaser, '_s_mu_norm', 0.0)):.4f}")
                except Exception:
                    pass

                # Align debiaser's internal scale to the unified Likert range for consistent bounds/diagnostics
                try:
                    debiaser._score_min = score_min
                    debiaser._score_max = score_max
                    debiaser._original_range = score_range
                except Exception:
                    pass

                # Transform overall scores using the unified scale
                debiased_scores = debiaser.transform(original_scores, score_min=score_min, score_max=score_max)
                
                # Additionally debias each factor column and collect per-sample fields
                debiased_factors = {}
                for factor_col in factor_columns:
                    try:
                        factor_vals = df[factor_col].values
                        debiased_factor_vals = debiaser.transform(factor_vals, score_min=score_min, score_max=score_max)
                        # Store with requested naming: score_debiased_<factorname without _score>
                        factor_base = factor_col.replace('_score', '')
                        key_name = f"score_debiased_{factor_base}"
                        debiased_factors[key_name] = [float(x) for x in debiased_factor_vals.tolist()]
                    except Exception as _e:
                        # Skip on failure; continue with others
                        continue
                
                # Get bias bounds
                bias_bounds = debiaser.get_bias_bounds(len(original_scores))
                # Validate effectiveness
                validation = debiaser.validate_effectiveness(original_scores, debiased_scores)
                # Collect diagnostics for consistent reporting downstream
                try:
                    diagnostics = debiaser.get_diagnostics() or {}
                except Exception:
                    diagnostics = {}
            
            
            
            # Store comprehensive results (convert numpy types for JSON serialization)
            result_data = {
                'success': True,
                'approach': approach_name,
                'strategy': approach_name,
                'original_scores': [float(x) for x in original_scores.tolist()],
                'debiased_scores': [float(x) for x in debiased_scores.tolist()],
                # Package per-sample outputs as dicts including per-factor debiased fields
                'model_question_data': [
                    dict(
                        model=str(m),
                        question_id=str(q),
                        game_index=int(g),
                        score_debiased=float(s),
                        **{k: float(v[idx]) if isinstance(v, list) else float(v) 
                           for k, v in debiased_factors.items() if idx < len(v)}
                    )
                    for idx, (m, q, g, s) in enumerate(zip(
                        df['model'].tolist(),
                        df['question_id'].tolist(),
                        df['game_index'].tolist(),
                        debiased_scores.tolist()
                    ))
                ],
                'diagnostics': {
                    'bias_sensitivity': float(diagnostics.get('bias_sensitivity', combined_fixed)) if isinstance(diagnostics, dict) else combined_fixed,
                    'combined_sensitivity': float(diagnostics.get('abb_constraint_validation', {}).get('combined_sensitivity', combined_fixed)) if isinstance(diagnostics, dict) else combined_fixed,
                    # Context-adjusted formatting RMS used (if available)
                    'context_adjusted_formatting_rms': float(profile_info['context_adjusted_rms']) if profile_info.get('context_adjusted_rms') is not None else None,
                    'context_adjust_intrinsic_rms': float(profile_info['intrinsic_sensitivity']) if profile_info.get('intrinsic_sensitivity') is not None else None,
                    'formatting_profile_rms': float(profile_info['formatting_sensitivity']) if profile_info.get('formatting_sensitivity') is not None else None,
                    # Save context-aware static sensitivities used in combination
                    'schematic_context_rms': float(schem_ctx),
                    'schematic_raw_rms': float(schematic_raw),
                    # hamming_profile_rms omitted in this script
                    'intrinsic_profile_rms': float(profile_info['intrinsic_sensitivity']) if profile_info.get('intrinsic_sensitivity') is not None else None,
                    'tau': float(bias_bounds['tau']),
                    'delta': float(bias_bounds['delta']),
                    'noise_std': float(bias_bounds['noise_std']),
                    'is_abb_mechanism': bool(diagnostics.get('is_abb_mechanism', False)) if isinstance(diagnostics, dict) else False,
                    # Preserve None when unavailable; do not force False default
                    'abb_constraint_satisfied': (
                        bool(diagnostics.get('abb_constraint_validation', {}).get('constraint_satisfied'))
                        if isinstance(diagnostics, dict) and diagnostics.get('abb_constraint_validation') is not None
                        else None
                    ),
                    'abb_constraint_margin': float(diagnostics.get('abb_constraint_validation', {}).get('margin', 0)) if isinstance(diagnostics, dict) and diagnostics.get('abb_constraint_validation', {}).get('margin') is not None else 0
                },
                'validation': {
                    'correlation': float(validation['correlation']),
                    'mean_absolute_difference': float(validation['mean_absolute_difference']),
                    'variance_ratio': float(validation['variance_ratio']),
                    'noise_level': float(validation['noise_level'])
                },
                'n_samples': int(len(original_scores))
            }
            
            # Add measurement breakdown
            abb_validation = diagnostics.get('abb_constraint_validation', {}) if isinstance(diagnostics, dict) else {}
            measurement_breakdown = abb_validation.get('measurement_breakdown', {})
            if measurement_breakdown:
                result_data['measurement_breakdown'] = {
                    'static_measurements': measurement_breakdown.get('static_measurements', {}),
                    'dynamic_measurements': measurement_breakdown.get('dynamic_measurements', {}),
                    'combination_strategy': measurement_breakdown.get('combination_strategy')
                }
            
            results[approach_name] = result_data
            
            print(f"    ✅ Success: {len(original_scores)} scores processed")
            # Handle combined sensitivity safely
            combined_sens = result_data['diagnostics'].get('combined_sensitivity', 'N/A')
            if isinstance(combined_sens, (int, float)):
                if np.isnan(combined_sens) or np.isinf(combined_sens):
                    print(f"    ⚠️ Combined sensitivity: {combined_sens} (invalid)")
                else:
                    print(f"    ✅ Combined sensitivity: {combined_sens:.4f}")
            else:
                print(f"    ✅ Combined sensitivity: {combined_sens}")
            
            print(f"    ✅ Correlation: {validation['correlation']:.3f}")
            
            # Report individual measurements (robust to simple float values)
            measurement_breakdown = result_data.get('measurement_breakdown')
            if measurement_breakdown:
                dynamic_measurements = measurement_breakdown.get('dynamic_measurements', {})
                for generator_name, measurement in dynamic_measurements.items():
                    if measurement is None:
                        print(f"    ❌ Dynamic {generator_name}: No measurement data")
                        continue
                    # Support both simple float values and structured dicts
                    if isinstance(measurement, (int, float, np.floating)):
                        print(f"    ✅ Dynamic {generator_name}: {float(measurement):.4f}")
                    elif isinstance(measurement, dict):
                        if measurement.get('success', False):
                            sensitivity_val = measurement.get('sensitivity', 'N/A')
                            if isinstance(sensitivity_val, (int, float, np.floating)):
                                print(f"    ✅ Dynamic {generator_name}: {float(sensitivity_val):.4f}")
                            else:
                                print(f"    ✅ Dynamic {generator_name}: {sensitivity_val}")
                        else:
                            error_msg = measurement.get('error', 'Unknown error')
                            error_msg = str(error_msg)[:50]
                            print(f"    ❌ Dynamic {generator_name}: {error_msg}")
                    else:
                        print(f"    ⚠️ Dynamic {generator_name}: Unrecognized format ({type(measurement).__name__})")

                # Report static measurements (may also be floats)
                static_measurements = measurement_breakdown.get('static_measurements', {})
                for estimator_name, measurement in static_measurements.items():
                    if measurement is None:
                        print(f"    ❌ Static {estimator_name}: No measurement data")
                        continue
                    if isinstance(measurement, (int, float, np.floating)):
                        print(f"    ✅ Static {estimator_name}: {float(measurement):.4f}")
                    elif isinstance(measurement, dict):
                        if measurement.get('success', False):
                            sensitivity_val = measurement.get('sensitivity', 'N/A')
                            if isinstance(sensitivity_val, (int, float, np.floating)):
                                print(f"    ✅ Static {estimator_name}: {float(sensitivity_val):.4f}")
                            else:
                                print(f"    ✅ Static {estimator_name}: {sensitivity_val}")
                        else:
                            error_msg = measurement.get('error', 'Unknown error')
                            error_msg = str(error_msg)[:50]
                            print(f"    ❌ Static {estimator_name}: {error_msg}")
                    else:
                        print(f"    ⚠️ Static {estimator_name}: Unrecognized format ({type(measurement).__name__})")
            # Suppress warning when no measurement breakdown is present (fixed/no-dynamic path)
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[approach_name] = {
                'success': False,
                'approach': approach_name,
                'error': str(e)
            }
    
    return results

def save_debiased_scores(base_path: Path, judge_name: str, approach_results: Dict[str, Any]):
    """Save debiased scores to individual model JSONL files in per-setting directories."""
    
    # For each successful approach, create directory and save files
    for approach_name, approach_data in approach_results.items():
        if approach_data.get('success', False):
            # Create directory for this setting and approach
            # e.g., "sos-addl-data/InDepthAnalysis/DeepSeek-R1-32B-setting1/base_debiased_combined_abb_conservative"
            debiased_dir = base_path / judge_name / f"base_debiased_{approach_name}"
            debiased_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract model-question-score data (support dict-based with per-factor fields)
            model_question_data = approach_data.get('model_question_data', [])
            
            # Group by model
            from collections import defaultdict
            model_scores = defaultdict(list)
            for item in model_question_data:
                if isinstance(item, dict):
                    out = {
                        "question_id": item.get("question_id"),
                        "game": int(item.get("game_index", item.get("game", 0))),
                        "score_debiased": float(item.get("score_debiased"))
                    }
                    # Include any per-factor debiased fields present
                    for k, v in item.items():
                        if k.startswith("score_debiased_") and k != "score_debiased":
                            try:
                                out[k] = float(v)
                            except Exception:
                                continue
                    model_scores[str(item.get("model"))].append(out)
                else:
                    # Backward-compat: handle tuple format (model, question_id, game_idx, score)
                    try:
                        model, question_id, game_idx, score = item
                        model_scores[model].append({
                            "question_id": question_id,
                            "game": int(game_idx),
                            "score_debiased": float(score)
                        })
                    except Exception:
                        continue
            
            # Save each model's scores to a JSONL file
            for model_name, scores in model_scores.items():
                output_file = debiased_dir / f"{model_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for score_entry in scores:
                        json.dump(score_entry, f, ensure_ascii=False)
                        f.write('\n')
                
                print(f"  📝 Saved {len(scores)} debiased scores for {model_name} to {output_file}")
            
            # Save approach metadata
            metadata_file = debiased_dir / f"debiasing_metadata_{approach_name}.json"
            metadata = {
                'approach': approach_name,
                'judge': judge_name,
                'n_samples': approach_data.get('n_samples'),
                'diagnostics': approach_data.get('diagnostics'),
                'validation': approach_data.get('validation')
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"  📝 Saved metadata for {approach_name} to {metadata_file}")

def main(args):
    """Main execution function."""
    print("🚀 Running Combined A-BB Debiasing Analysis")
    print("=" * 60)

    dataset_judge_map = get_dataset_judge_mapping()
    
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
            if judge_name not in dataset_judge_map:
                raise ValueError(
                    f"No judge mapping found for dataset '{judge_name}'. "
                    f"Available mappings: {list(dataset_judge_map.keys())}"
                )

            judge_to_use = dataset_judge_map[judge_name]
            print(f"  🔧 Using judge: {judge_to_use} for dataset: {judge_name}")
            
            # Run Combined A-BB analysis
            analysis_results = run_combined_abb_analysis(
                df,
                judge_name,
                real_judge_name=judge_to_use,
                strict_abb=bool(getattr(args, 'strict_abb', False)),
                enable_shrinkage=bool(getattr(args, 'enable_shrinkage', False)),
                target_tau=getattr(args, 'target_tau', None),
                target_delta=getattr(args, 'target_delta', None),
                shrink_alpha=getattr(args, 'shrink_alpha', None),
                shrink_center=str(getattr(args, 'shrink_center', 'mean')),
                abb_dimensionality=getattr(args, 'abb_dimensionality', None),
            )
            
            # Save debiased scores to individual model JSONL files
            save_debiased_scores(base_path, judge_name, analysis_results)
            
            # Store results (keep for summary/compatibility)
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
    
    # Save overall results summary (without individual scores)
    output_file = base_path / "combined_abb_analysis_results.json"
    
    # Create summary without individual scores
    summary_results = _summarize_results(overall_results)
    rounded_results = round_nested(summary_results, sig=3)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rounded_results, f, indent=2, ensure_ascii=False)
    
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
    parser.add_argument("--strict-abb", action="store_true",
                        help="Pre-check A-BB constraint with normalized sensitivity and skip strategy when margin <= 0")
    parser.add_argument("--enable-shrinkage", action="store_true",
                        help="Enable shrinkage (contractive) mapping before Gaussian noise")
    parser.add_argument("--target-tau", type=float, default=None,
                        help="Target tau (original score units) to calibrate shrinkage alpha if shrink-alpha not provided")
    parser.add_argument("--target-delta", type=float, default=None,
                        help="Override delta for all strategies (must be in (0,1))")
    parser.add_argument("--shrink-alpha", type=float, default=None,
                        help="Explicit shrinkage alpha in (0,1]; overrides target-tau calibration if provided")
    parser.add_argument("--shrink-center", type=str, default="mean",
                        help="Shrinkage center: mean|median|holdout_mean|profile_mean|ema(0.9)|zero|one")
    parser.add_argument("--abb-dimensionality", type=int, default=None,
                        help="Override effective dimensionality used for A-BB noise (defaults to sample count)")
    
    args = parser.parse_args()
    
    # Set global test mode
    if args.test:
        print(f"🧪 Running in TEST MODE with {args.test_samples} samples")
        os.environ['TEST_MODE'] = 'true'
        os.environ['TEST_SAMPLES'] = str(args.test_samples)
    
    exit_code = main(args)
    exit(exit_code)
