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
from differential_debiasing.core.sensitivity_profiles import collect_profile_info
from differential_debiasing.core.utils import (
    context_adjusted_rms,
    compute_abb_constraint_validation,
    round_nested,
)
from differential_debiasing.interfaces import (
    get_judge_config_path,
    find_judge_data_directories,
    load_and_prepare_score_data,
)
from differential_debiasing.sensitivity.psychometric_reliability import PsychometricReliabilitySensitivity
from differential_debiasing.sensitivity.schematic_adherence import SchematicAdherenceSensitivity
# Note: Oumi judge interface is imported lazily only when needed

def run_combined_abb_analysis(df: pd.DataFrame, judge_name: str,
                              real_judge_name: str,
                              strict_abb: bool = False,
                              model_question_mapping: Optional[Dict[str, List[Dict]]] = None,
                              enable_shrinkage: bool = False,
                              target_tau: Optional[float] = None,
                              shrink_alpha: Optional[float] = None,
                              shrink_center: str = "mean") -> Dict[str, Any]:
    """Run Combined A-BB debiasing approaches on the judge data."""
    print(f"\nRunning Combined A-BB analysis for {judge_name}...")
    
    config_manager = ConfigManager()
    
    # Check for sensitivity profiles for the real judge, and dataset-scoped intrinsic jitter
    profile_info = collect_profile_info(real_judge_name, judge_name, verbose=True)
    
    # Judge function not needed in this script (no dynamic generators)
    judge_function = None
    
    # Determine dynamic generators based on profile availability
    profile_override = None
    ctx = profile_info.get('context_adjusted_rms') if profile_info else None
    if ctx is None:
        # If formatting is missing entirely, we cannot proceed safely
        raise RuntimeError(
            f"Formatting sensitivity profile missing for judge '{real_judge_name}'. Please generate it first."
        )
    # No dynamic generators in this script (formatting covered by profile; omit hamming as trivial here)
    dynamic_generators = []
    # Build a composed profile override including only formatting (context-adjusted RMS)
    profile_override = {'value': float(ctx)}
    print(f"🚀 Using Context-Adjusted RMS for formatting: {ctx:.4f}")

    # Establish a single, consistent score range from input data and reuse it
    # Compute across all *_score columns to capture the maximum observed range
    score_cols_all = [col for col in df.columns if col.endswith('_score')]
    if score_cols_all:
        global_min = float(pd.concat([df[c] for c in score_cols_all], axis=0).min())
        global_max = float(pd.concat([df[c] for c in score_cols_all], axis=0).max())
        score_min = global_min
        score_max = global_max
        score_range = float(score_max - score_min)
        if score_range <= 0:
            # Fallback to 1-5 Likert range if degenerate
            score_min, score_max, score_range = 1.0, 5.0, 4.0
    else:
        # Fallback if schema unexpected
        score_min, score_max, score_range = 1.0, 5.0, 4.0
    print(f"   ↪︎ Using unified score scale: min={score_min:.3f}, max={score_max:.3f}, range={score_range:.3f}")

    # Compute context-aware static sensitivities (psychometric and schematic)
    factor_columns = [col for col in df.columns if col.endswith('_score') and col != 'overall_score']

    # Psychometric
    try:
        psych_est = PsychometricReliabilitySensitivity(factor_columns=factor_columns, robust=True)
        psych_est.fit(df)
        psych_raw = float(psych_est.estimate(score_range))
    except Exception as e:
        print(f"Warning: Psychometric reliability estimation failed: {e}")
        psych_raw = 0.0
    psych_ctx = context_adjusted_rms(psych_raw, float(profile_info['intrinsic_sensitivity'])) if profile_info.get('intrinsic_sensitivity') is not None else psych_raw

    # Schematic
    try:
        schem_est = SchematicAdherenceSensitivity(factor_columns=factor_columns, target_column='overall_score')
        schem_est.fit(df)
        schem_raw = float(schem_est.estimate(score_range))
    except Exception as e:
        print(f"Warning: Schematic adherence estimation failed: {e}")
        schem_raw = 0.0
    schem_ctx = context_adjusted_rms(schem_raw, float(profile_info['intrinsic_sensitivity'])) if profile_info.get('intrinsic_sensitivity') is not None else schem_raw

    print(f"   ↪︎ context-aware static: psych={psych_ctx:.4f}, schematic={schem_ctx:.4f}")

    # Create real judge function using Oumi only if dynamic generators are needed
    if len(dynamic_generators) > 0:
        from differential_debiasing.interfaces.oumi_interface import create_oumi_judge_function
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

    # Define Combined A-BB approaches with different aggregation strategies
    # Set smaller tau values (1.0–1.5 range) per request
    # Choose estimator type based on profile availability
    # Use profile-enhanced estimator when we have any profile override or judge-level profile
    # We will pass a fixed combined sensitivity for each strategy
    estimator_type = 'fixed'
    
    approaches = {
        'combined_abb_conservative': {
            'estimator': estimator_type,
            'tau': 1.2,
            'delta': 0.15,
            'use_average_case': True,
            'extra_params': {
                # Fixed estimator parameters (set later per-strategy before init)
                'fixed_sensitivity_value': None,
            }
        },
        'combined_abb_rms': {
            'estimator': estimator_type,
            'tau': 1.2,
            'delta': 0.1,
            'use_average_case': True,
            'extra_params': {
                'fixed_sensitivity_value': None,
            }
        },
        'combined_abb_weighted': {
            'estimator': estimator_type,
            'tau': 1.2,
            'delta': 0.1,
            'use_average_case': True,
            'extra_params': {
                'fixed_sensitivity_value': None,
            }
        },
        'abb_formatting_only': {
            'estimator': estimator_type,
            'tau': 1.2,
            'delta': 0.05,
            'use_average_case': True,
            'extra_params': {
                'fixed_sensitivity_value': None,
            }
        },
        'combined_abb_montecarlo': {
            'estimator': estimator_type,
            'tau': 1.2,
            'delta': 0.1,
            'use_average_case': True,
            'extra_params': {
                'fixed_sensitivity_value': None,
                'mc_weights': {
                    'formatting': 1.0/3.0,
                    'psychometric': 1.0/3.0,
                    'schematic': 1.0/3.0,
                }
            }
        }
    }
    
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
            init_params = {
                'tau': effective_tau,
                'delta': approach_config.get('delta', 0.1),
                'sensitivity_estimator': approach_config['estimator'],
                'use_average_case': approach_config.get('use_average_case', True),
                'random_seed': 42
            }

            # Shrinkage controls (from function parameters)
            if bool(enable_shrinkage):
                if shrink_alpha is not None:
                    init_params['shrink_alpha'] = float(shrink_alpha)
                if target_tau is not None:
                    init_params['target_tau'] = float(target_tau)
                init_params['shrink_center'] = str(shrink_center or 'mean')
            
            # Add extra parameters
            init_params.update(approach_config['extra_params'])

            # Compute combined fixed sensitivity for this approach
            # Add a slight perturbation to avoid exact zeros leading to no-op (TODO: revisit handling of zero ctx)
            _eps = 1e-3
            _fmt = ctx if ctx > 0 else _eps
            _psy = psych_ctx if psych_ctx > 0 else _eps
            _sch = schem_ctx if schem_ctx > 0 else _eps

            if approach_name == 'combined_abb_conservative':
                combined_fixed = float(max(_fmt, _psy, _sch))
            elif approach_name == 'combined_abb_rms':
                import numpy as _np
                combined_fixed = float(_np.sqrt(_np.mean(_np.square([_fmt, _psy, _sch]))))
            elif approach_name == 'combined_abb_montecarlo':
                import numpy as _np
                w = approach_config['extra_params'].get('mc_weights', {}) or {}
                w_fmt = float(w.get('formatting', 1.0/3.0))
                w_psy = float(w.get('psychometric', 1.0/3.0))
                w_sch = float(w.get('schematic', 1.0/3.0))
                total_w = w_fmt + w_psy + w_sch
                if total_w <= 0:
                    w_fmt = w_psy = w_sch = 1.0/3.0
                    total_w = 1.0
                w_fmt /= total_w
                w_psy /= total_w
                w_sch /= total_w
                v_fmt = _fmt * _fmt
                v_psy = _psy * _psy
                v_sch = _sch * _sch
                combined_fixed = float(_np.sqrt(w_fmt * v_fmt + w_psy * v_psy + w_sch * v_sch))
                print(f"    MC weights: formatting={w_fmt:.3f}, psychometric={w_psy:.3f}, schematic={w_sch:.3f}")
            elif approach_name == 'abb_formatting_only':
                combined_fixed = float(_fmt)
            else:  # weighted: default equal weights
                combined_fixed = float((_fmt + _psy + _sch) / 3.0)

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
                    'signal_preservation': 1.0,
                    'noise_level': 0.0
                }
            else:
                # A-BB precheck using normalized sensitivity with optional shrinkage estimate
                delta_val = float(approach_config.get('delta', init_params.get('delta', 0.1)))
                tau_val = float(effective_tau)
                # Estimate alpha and S_mu bound
                s_mu_norm = 0.0
                if enable_shrinkage and str(shrink_center).lower() in ("mean", "median"):
                    n_samples = int(len(df))
                    d_eff = n_samples
                    s_mu_norm = float(min(1.0, (d_eff ** 0.5) / max(n_samples, 1)))
                delta_factor = float(np.sqrt(2.0 / delta_val))
                delta_hat_raw = float(combined_fixed / score_range) if score_range > 0 else float(combined_fixed)
                tau_norm = float(tau_val / score_range) if score_range > 0 else float(tau_val)
                A = delta_hat_raw * delta_factor
                B = float(s_mu_norm) * delta_factor
                if enable_shrinkage:
                    if shrink_alpha is not None:
                        alpha_hat = float(shrink_alpha)
                    elif target_tau is not None and score_range > 0:
                        target_tau_norm = float(target_tau / score_range)
                        if A > B:
                            alpha_hat = float(max(0.0, min(1.0, (target_tau_norm - B) / (A - B))))
                        else:
                            alpha_hat = 1.0 if target_tau_norm > B else 0.0
                    else:
                        alpha_hat = 1.0
                else:
                    alpha_hat = 1.0
                eff_sens = alpha_hat * combined_fixed
                precheck = compute_abb_constraint_validation(
                    tau=tau_val, delta=delta_val, sensitivity=eff_sens, score_range=score_range
                )
                delta_hat = precheck.get('normalized_sensitivity')
                tau_norm_report = precheck.get('normalized_tau', tau_norm)
                threshold = precheck.get('constraint_threshold')
                margin = precheck.get('margin')
                print(
                    f"    A-BB precheck (normalized): tau={tau_norm_report:.4f} (raw {tau_val:.3f}), "
                    f"delta={delta_val:.3f}, α̂={alpha_hat:.3f}, Sμ̂={s_mu_norm:.4f}, "
                    f"A={A:.4f}, B={B:.4f}, Δ̂eff={delta_hat:.4f}, "
                    f"threshold={threshold:.4f}, margin={margin:.4f} (range={score_range:.3f})"
                )
                if strict_abb and float(margin) <= 0.0:
                    msg = (
                        f"Strict mode: A-BB precheck failed (tau={tau_val:.6f} <= threshold={threshold:.6f}); skipping strategy"
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
                    'psychometric_context_rms': float(psych_ctx),
                    'schematic_context_rms': float(schem_ctx),
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
                    'signal_preservation': float(validation['signal_preservation']),
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
            if approach_name == 'combined_abb_montecarlo':
                w = approach_config['extra_params'].get('mc_weights', {}) or {}
                result_data.setdefault('measurement_breakdown', {})
                result_data['measurement_breakdown'].update({
                    'monte_carlo_weights': {
                        'formatting': float(w.get('formatting', 1.0/3.0)),
                        'psychometric': float(w.get('psychometric', 1.0/3.0)),
                        'schematic': float(w.get('schematic', 1.0/3.0)),
                    }
                })
            
            # Add judge cost information if using real judge
            if hasattr(judge_function, 'interface'):
                cost_summary = judge_function.interface.get_cost_summary()
                result_data['judge_costs'] = cost_summary
            
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
                df,
                judge_name,
                real_judge_name=judge_to_use,
                strict_abb=bool(getattr(args, 'strict_abb', False)),
                enable_shrinkage=bool(getattr(args, 'enable_shrinkage', False)),
                target_tau=getattr(args, 'target_tau', None),
                shrink_alpha=getattr(args, 'shrink_alpha', None),
                shrink_center=str(getattr(args, 'shrink_center', 'mean')),
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
    summary_results = {}
    for judge_name, judge_data in overall_results.items():
        summary_results[judge_name] = {
            'data_source': judge_data.get('data_source'),
            'n_samples': judge_data.get('n_samples')
        }
        
        if 'error' in judge_data:
            summary_results[judge_name]['error'] = judge_data['error']
        
        if 'approaches' in judge_data:
            summary_approaches = {}
            for approach_name, approach_data in judge_data['approaches'].items():
                if isinstance(approach_data, dict):
                    # Remove large data arrays
                    summary_approach = {k: v for k, v in approach_data.items() 
                                     if k not in ['original_scores', 'debiased_scores', 'model_question_data']}
                    summary_approaches[approach_name] = summary_approach
                else:
                    summary_approaches[approach_name] = approach_data
            summary_results[judge_name]['approaches'] = summary_approaches
    
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
    parser.add_argument("--shrink-alpha", type=float, default=None,
                        help="Explicit shrinkage alpha in (0,1]; overrides target-tau calibration if provided")
    parser.add_argument("--shrink-center", type=str, default="mean",
                        help="Shrinkage center: mean|median|holdout_mean|profile_mean|ema(0.9)|zero|one")
    
    args = parser.parse_args()
    
    # Set global test mode
    if args.test:
        print(f"🧪 Running in TEST MODE with {args.test_samples} samples")
        os.environ['TEST_MODE'] = 'true'
        os.environ['TEST_SAMPLES'] = str(args.test_samples)
    
    exit_code = main(args)
    exit(exit_code)
