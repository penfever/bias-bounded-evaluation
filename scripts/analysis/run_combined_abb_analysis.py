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
from differential_debiasing.core.sensitivity_profiles import (
    load_judge_sensitivity_profile,
    get_formatting_sensitivity,
    load_intrinsic_sensitivity_profile,
    SensitivityProfileManager,
)
from differential_debiasing.core.utils import context_adjusted_rms, compute_abb_constraint_validation

# --- Local helpers -----------------------------------------------------------
def _round_sig(x: float, sig: int = 3) -> float:
    try:
        import math
        if x == 0 or not math.isfinite(float(x)):
            return float(x)
        return float(round(x, sig - int(math.floor(math.log10(abs(x)))) - 1))
    except Exception:
        return float(x)

def _round_nested(obj, sig: int = 3):
    # Recursively round floats in nested structures to sig significant digits
    import numbers
    if isinstance(obj, float):
        return _round_sig(obj, sig)
    if isinstance(obj, list):
        return [_round_nested(v, sig) for v in obj]
    if isinstance(obj, dict):
        return {k: _round_nested(v, sig) for k, v in obj.items()}
    # Leave ints, bools, None, strings as-is
    return obj
from differential_debiasing.sensitivity.psychometric_reliability import PsychometricReliabilitySensitivity
from differential_debiasing.sensitivity.schematic_adherence import SchematicAdherenceSensitivity
# Note: Oumi judge interface is imported lazily only when needed

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

def check_sensitivity_profiles(real_judge_name: str, dataset_id: str) -> Dict[str, Any]:
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
        'hamming_sensitivity': None,
        'combined_average_sensitivity': None,
        'intrinsic_sensitivity': None,
        'context_adjusted_rms': None,
        'profile_dir': str(profile_dir)
    }
    
    try:
        profile = load_judge_sensitivity_profile(real_judge_name, profile_dir)
        if profile:
            formatting_sensitivity = profile.get_formatting_sensitivity()
            if formatting_sensitivity is not None:
                profile_info['formatting_sensitivity'] = formatting_sensitivity
                profile_info['profile_available'] = True
                print(f"✅ Found formatting profile for {real_judge_name}: {formatting_sensitivity:.4f}")
            else:
                print(f"⚠️ Profile found for {real_judge_name} but formatting sensitivity is missing/failed")

            # Optional values: hamming and combined average, if present in the profile
            try:
                ham = profile.get_hamming_sensitivity() if hasattr(profile, 'get_hamming_sensitivity') else None
                if ham is not None:
                    profile_info['hamming_sensitivity'] = float(ham)
                    # Intentionally no print: hamming is not used in this script
            except Exception:
                pass
            try:
                cav = profile.get_combined_average_sensitivity() if hasattr(profile, 'get_combined_average_sensitivity') else None
                if cav is not None:
                    profile_info['combined_average_sensitivity'] = float(cav)
            except Exception:
                pass

        # Try dataset-scoped intrinsic jitter profile
        mgr = SensitivityProfileManager(profile_dir)
        intrinsic = mgr.get_intrinsic_sensitivity_for_dataset(real_judge_name, dataset_id)
        if intrinsic is None:
            # Fallback: use any intrinsic profile available for this judge
            fallback = mgr.find_any_intrinsic_profile_for_judge(real_judge_name)
            if fallback is not None:
                fb_dataset, fb_prof = fallback
                intrinsic = fb_prof.get_intrinsic_sensitivity()
                if intrinsic is not None:
                    print(f"✅ Using fallback intrinsic from dataset={fb_dataset} for judge={real_judge_name}: {intrinsic:.4f}")
        if intrinsic is not None:
            profile_info['intrinsic_sensitivity'] = intrinsic
            print(f"✅ Found intrinsic jitter for ({dataset_id}, {real_judge_name}): {intrinsic:.4f}")
        else:
            print(f"📋 No intrinsic jitter profile found for dataset={dataset_id}, judge={real_judge_name}")

        # Derive context-adjusted RMS; if intrinsic missing, proceed with formatting as-is
        if profile_info['formatting_sensitivity'] is not None:
            tot = float(profile_info['formatting_sensitivity'])
            if profile_info['intrinsic_sensitivity'] is not None:
                intr = float(profile_info['intrinsic_sensitivity'])
                print(f"   ↪︎ context_adjusted_rms inputs (check_sensitivity_profiles): formatting_total={tot:.4f}, intrinsic={intr:.4f}")
                ctx = context_adjusted_rms(tot, intr)
            else:
                ctx = tot
                print(f"ℹ️ Intrinsic missing; using formatting RMS as context-adjusted value: {ctx:.4f}")
            profile_info['context_adjusted_rms'] = ctx
            print(f"🎯 Context-adjusted RMS for ({dataset_id}, {real_judge_name}): {ctx:.4f}")
        else:
            print(f"📋 No formatting sensitivity profile found for {real_judge_name}")
    except Exception as e:
        print(f"⚠️ Error loading profile for {real_judge_name}: {e}")
    
    return profile_info


def run_combined_abb_analysis(df: pd.DataFrame, judge_name: str,
                              real_judge_name: str,
                              strict_abb: bool = False,
                              model_question_mapping: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, Any]:
    """Run Combined A-BB debiasing approaches on the judge data."""
    print(f"\nRunning Combined A-BB analysis for {judge_name}...")
    
    config_manager = ConfigManager()
    
    # Check for sensitivity profiles for the real judge, and dataset-scoped intrinsic jitter
    profile_info = check_sensitivity_profiles(real_judge_name, dataset_id=judge_name)
    
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
            'tau': 2.5,
            'delta': 0.15,
            'use_average_case': True,
            'extra_params': {
                # Fixed estimator parameters (set later per-strategy before init)
                'fixed_sensitivity_value': None,
            }
        },
        'combined_abb_rms': {
            'estimator': estimator_type,
            'tau': 2.25,
            'delta': 0.1,
            'use_average_case': True,
            'extra_params': {
                'fixed_sensitivity_value': None,
            }
        },
        'combined_abb_weighted': {
            'estimator': estimator_type,
            'tau': 2.0,
            'delta': 0.1,
            'use_average_case': True,
            'extra_params': {
                'fixed_sensitivity_value': None,
            }
        },
        'combined_abb_montecarlo': {
            'estimator': estimator_type,
            'tau': 1.5,
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
            init_params = {
                'tau': approach_config.get('tau', 2.0),
                'delta': approach_config.get('delta', 0.1),
                'sensitivity_estimator': approach_config['estimator'],
                'use_average_case': approach_config.get('use_average_case', True),
                'random_seed': 42
            }
            
            # Add extra parameters
            init_params.update(approach_config['extra_params'])

            # Compute combined fixed sensitivity for this approach
            if approach_name == 'combined_abb_conservative':
                combined_fixed = float(max(ctx, psych_ctx, schem_ctx))
            elif approach_name == 'combined_abb_rms':
                import numpy as _np
                combined_fixed = float(_np.sqrt(_np.mean(_np.square([ctx, psych_ctx, schem_ctx]))))
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
                v_fmt = ctx * ctx
                v_psy = psych_ctx * psych_ctx
                v_sch = schem_ctx * schem_ctx
                combined_fixed = float(_np.sqrt(w_fmt * v_fmt + w_psy * v_psy + w_sch * v_sch))
                print(f"    MC weights: formatting={w_fmt:.3f}, psychometric={w_psy:.3f}, schematic={w_sch:.3f}")
            else:  # weighted: default equal weights
                combined_fixed = float((ctx + psych_ctx + schem_ctx) / 3.0)

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
                # A-BB precheck using normalized sensitivity (no mechanism yet): Δ̂ = Δ / range
                delta_val = float(approach_config.get('delta', init_params.get('delta', 0.1)))
                tau_val = float(approach_config.get('tau', init_params.get('tau', 2.0)))
                precheck = compute_abb_constraint_validation(
                    tau=tau_val, delta=delta_val, sensitivity=combined_fixed, score_range=score_range
                )
                delta_hat = precheck.get('normalized_sensitivity')
                threshold = precheck.get('constraint_threshold')
                margin = precheck.get('margin')
                print(f"    A-BB precheck: tau={tau_val:.3f}, delta={delta_val:.3f}, Δ̂={delta_hat:.4f}, threshold={threshold:.4f}, margin={margin:.4f} (range={score_range:.3f})")
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

                # Align debiaser's internal scale to the unified Likert range for consistent bounds/diagnostics
                try:
                    debiaser._score_min = score_min
                    debiaser._score_max = score_max
                    debiaser._original_range = score_range
                except Exception:
                    pass

                # Transform overall scores using the unified scale
                debiased_scores = debiaser.transform(original_scores, score_min=score_min, score_max=score_max)
                
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
                'model_question_data': list(zip(df['model'].tolist(), df['question_id'].tolist(), debiased_scores.tolist())),
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
                import numpy as np
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
            # e.g., "sos-addl-data/InDepthAnalysis/DeepSeek-R1-32B-setting1/base_debiased"
            debiased_dir = base_path / judge_name / "base_debiased"
            debiased_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract model-question-score data
            model_question_data = approach_data.get('model_question_data', [])
            
            # Group by model
            from collections import defaultdict
            model_scores = defaultdict(list)
            for model, question_id, score in model_question_data:
                model_scores[model].append({
                    "question_id": question_id,
                    "score_debiased": float(score)
                })
            
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
            
            # For now, only save the first successful approach
            # (you can modify this to save all approaches in separate subdirectories if needed)
            break

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
                df, judge_name, real_judge_name=judge_to_use, strict_abb=bool(getattr(args, 'strict_abb', False))
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
    
    rounded_results = _round_nested(summary_results, sig=3)
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
    
    args = parser.parse_args()
    
    # Set global test mode
    if args.test:
        print(f"🧪 Running in TEST MODE with {args.test_samples} samples")
        os.environ['TEST_MODE'] = 'true'
        os.environ['TEST_SAMPLES'] = str(args.test_samples)
    
    exit_code = main(args)
    exit(exit_code)
