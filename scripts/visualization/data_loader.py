#!/usr/bin/env python3
"""
Unified data loader for visualization scripts.

Handles loading both original evaluation data (base_processed) and debiased scores (base_debiased)
from the new JSONL format.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from differential_debiasing.interfaces.arena_hard_utils import (
    convert_scores_to_win_rates,
    bootstrap_to_win_rate_ci,
)
from differential_debiasing.interfaces import get_arena_score_mapping


def get_project_paths() -> Dict[str, Path]:
    """Discover project paths from current location."""
    # Start from this file's location
    current_file = Path(__file__).resolve()
    
    # Navigate up to find the project root (bias-bounded-evaluation)
    project_root = current_file.parent
    while project_root.name != "bias-bounded-evaluation" and project_root.parent != project_root:
        project_root = project_root.parent
    
    if project_root.name != "bias-bounded-evaluation":
        # Fallback: assume we're in scripts/visualization
        project_root = current_file.parent.parent.parent
    
    # Define standard paths
    # The actual project structure has sos-addl-data at the same level as bias-bounded-evaluation
    actual_data_base = project_root.parent / 'sos-addl-data' / 'InDepthAnalysis'
    
    paths = {
        'project_root': project_root,
        'figures': project_root / 'figures',
        'tables': project_root / 'tables',
        'data_base': actual_data_base,
        'scripts': project_root / 'scripts',
        'visualization': project_root / 'scripts' / 'visualization'
    }
    
    # Create directories if they don't exist
    paths['figures'].mkdir(parents=True, exist_ok=True)
    paths['tables'].mkdir(parents=True, exist_ok=True)
    
    return paths


def _parse_ci_str(ci: str) -> Optional[Tuple[float, float]]:
    try:
        import re
        nums = re.findall(r'[-+]?\d*\.?\d+', str(ci))
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    except Exception:
        pass
    return None


def detect_baseline_model(judge_dir: Union[str, Path]) -> Optional[str]:
    """Detect baseline model name for a setting by scanning original CIS CSVs.

    Heuristic: in factor_scores_original_cis/*.csv, one model has zero CI (both bounds zero).
    We scan files and return the most frequent such model.
    """
    judge_dir = Path(judge_dir)
    cis_dir = judge_dir / 'tables' / 'factor_scores_original_cis'
    if not cis_dir.exists():
        return None
    counts: Dict[str, int] = {}
    try:
        for csv_path in cis_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            model_col = 'model' if 'model' in df.columns else None
            if model_col is None:
                # try common variants
                for cand in ['model_id', 'name', 'Model']:
                    if cand in df.columns:
                        df = df.rename(columns={cand: 'model'})
                        model_col = 'model'
                        break
            if model_col is None or 'CI' not in df.columns and not ({'rating_q025','rating_q975'} <= set(df.columns)):
                continue
            baseline_candidates = []
            if 'CI' in df.columns:
                for _, row in df.iterrows():
                    parsed = _parse_ci_str(row['CI'])
                    if parsed is None:
                        continue
                    lo, hi = parsed
                    if abs(lo) < 1e-3 and abs(hi) < 1e-3:
                        baseline_candidates.append(str(row['model']))
            if not baseline_candidates and {'rating_q025','rating_q975'} <= set(df.columns):
                for _, row in df.iterrows():
                    try:
                        if float(row['rating_q025']) == float(row['rating_q975']):
                            # Optional: also check equals score if exists
                            if 'score' in df.columns:
                                if abs(float(row['rating_q025']) - float(row['score'])) > 1e-9:
                                    continue
                            baseline_candidates.append(str(row['model']))
                    except Exception:
                        continue
            for m in baseline_candidates:
                counts[m] = counts.get(m, 0) + 1
    except Exception:
        return None
    if not counts:
        return None
    # Return the most frequent candidate
    return max(counts.items(), key=lambda kv: kv[1])[0]


def load_original_evaluations(base_processed_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all judge evaluations from JSONL files in base_processed directory.
    
    Returns DataFrame with columns: model, question_id, game_index, score, [factor_scores...]
    """
    base_processed_dir = Path(base_processed_dir)
    if not base_processed_dir.exists():
        raise ValueError(f"Base processed directory not found: {base_processed_dir}")
    
    score_mapping = get_arena_score_mapping()
    all_scores = []
    
    # Process each JSONL file
    for jsonl_file in base_processed_dir.glob("*.jsonl"):
        model_name = jsonl_file.stem
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        question_id = data.get('question_id', '')
                        games = data.get('games', [])
                        
                        # Process each game
                        for game_idx, game in enumerate(games):
                            score_data = {
                                'model': model_name,
                                'question_id': question_id,
                                'game_index': game_idx,
                            }
                            
                            # Extract scores for each factor
                            factor_fields = ['score', 'correctness_score', 'completeness_score', 
                                           'safety_score', 'conciseness_score', 'style_score']
                            
                            for factor in factor_fields:
                                if factor in game:
                                    raw_judgment = game[factor]
                                    if isinstance(raw_judgment, str) and raw_judgment.strip():
                                        numeric_score = score_mapping.get(raw_judgment.strip(), 3)
                                    else:
                                        numeric_score = 3
                                    score_data[factor] = float(numeric_score)
                                else:
                                    score_data[factor] = 3.0
                            
                            # Ensure we have an overall score
                            if 'overall_score' not in score_data:
                                score_data['overall_score'] = score_data.get('score', 3.0)
                            
                            all_scores.append(score_data)
                    
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line in {jsonl_file}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error loading {jsonl_file}: {e}")
            continue
    
    df = pd.DataFrame(all_scores)
    
    if len(df) > 0:
        print(f"Loaded {len(df)} original evaluations from {len(set(df['model']))} models")
    else:
        print("Warning: No original evaluations loaded")
    
    return df


def load_debiased_scores(base_debiased_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all debiased scores from JSONL files in base_debiased directory.
    
    Returns DataFrame with columns: model, question_id, game, score_debiased
    """
    base_debiased_dir = Path(base_debiased_dir)
    if not base_debiased_dir.exists():
        raise ValueError(f"Base debiased directory not found: {base_debiased_dir}")
    
    all_scores = []
    
    # Process each JSONL file
    for jsonl_file in base_debiased_dir.glob("*.jsonl"):
        model_name = jsonl_file.stem
        
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        score_data = {
                            'model': model_name,
                            'question_id': data['question_id'],
                            'game': data['game'],
                        }
                        # Always include overall debiased score if present
                        if 'score_debiased' in data:
                            score_data['score_debiased'] = float(data['score_debiased'])
                        # Include any per-factor debiased fields
                        for k, v in data.items():
                            if isinstance(k, str) and k.startswith('score_debiased_'):
                                try:
                                    score_data[k] = float(v)
                                except Exception:
                                    continue
                        all_scores.append(score_data)
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"Warning: Skipping invalid line in {jsonl_file}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error loading {jsonl_file}: {e}")
            continue
    
    df = pd.DataFrame(all_scores)
    
    if len(df) > 0:
        print(f"Loaded {len(df)} debiased scores from {len(set(df['model']))} models")
    else:
        print("Warning: No debiased scores loaded")
    
    return df


def merge_original_and_debiased(original_df: pd.DataFrame, debiased_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge original and debiased DataFrames on (model, question_id, game).
    
    Handles column name differences (game_index vs game) gracefully.
    """
    # Ensure consistent column naming
    if 'game_index' in original_df.columns and 'game' not in original_df.columns:
        original_df = original_df.rename(columns={'game_index': 'game'})
    
    # Merge on model, question_id, and game
    merge_keys = ['model', 'question_id', 'game']
    
    # Determine which debiased score columns are available (overall + per-factor)
    debiased_cols = [c for c in debiased_df.columns if c == 'score_debiased' or c.startswith('score_debiased_')]
    # Keep all original columns and add all debiased score columns
    merged_df = pd.merge(
        original_df,
        debiased_df[merge_keys + debiased_cols],
        on=merge_keys,
        how='left'
    )
    
    # Report merge statistics
    n_original = len(original_df)
    n_matched = merged_df['score_debiased'].notna().sum()
    print(f"Merged {n_matched}/{n_original} records ({n_matched/n_original*100:.1f}% match rate)")
    
    return merged_df


def aggregate_scores_by_model(df: pd.DataFrame, 
                             score_cols: Optional[List[str]] = None,
                             confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Aggregate scores by model, computing mean and confidence intervals.
    
    Parameters:
    -----------
    df : DataFrame with individual evaluation scores
    score_cols : List of score columns to aggregate (default: all numeric columns ending with 'score')
    confidence_level : Confidence level for intervals (default: 0.95)
    
    Returns:
    --------
    DataFrame with model-level aggregated scores and confidence intervals
    """
    if score_cols is None:
        # Find all numeric columns for original factors and debiased (overall + per-factor)
        score_cols = [col for col in df.columns 
                     if (
                         col.endswith('score')  # original factors and overall
                         or col == 'score_debiased'  # overall debiased
                         or col.startswith('score_debiased_')  # per-factor debiased
                     ) and pd.api.types.is_numeric_dtype(df[col])]
    
    # Group by model
    grouped = df.groupby('model')
    
    # Compute statistics
    agg_dict = {}
    for col in score_cols:
        agg_dict[col] = ['mean', 'std', 'count', 'sem']
    
    model_stats = grouped.agg(agg_dict)
    
    # Flatten column names
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    
    # Add confidence intervals
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    result_df = pd.DataFrame(index=model_stats.index)
    result_df['model'] = model_stats.index
    
    for col in score_cols:
        result_df[col] = model_stats[f'{col}_mean']
        sem = model_stats[f'{col}_sem']
        ci_lower = model_stats[f'{col}_mean'] - z_score * sem
        ci_upper = model_stats[f'{col}_mean'] + z_score * sem
        
        # Format CI as string for compatibility with existing visualization code
        # Visualization expects deltas from mean, not absolute bounds
        result_df[f'{col}_CI'] = result_df.apply(
            lambda row: f"({ci_lower.loc[row.name] - model_stats[f'{col}_mean'].loc[row.name]:.2f}, +{ci_upper.loc[row.name] - model_stats[f'{col}_mean'].loc[row.name]:.2f})", 
            axis=1
        )
        
        # Also store numeric CI bounds
        result_df[f'{col}_CI_lower'] = ci_lower
        result_df[f'{col}_CI_upper'] = ci_upper
        result_df[f'{col}_std'] = model_stats[f'{col}_std']
        result_df[f'{col}_count'] = model_stats[f'{col}_count']
    
    result_df = result_df.reset_index(drop=True)
    
    return result_df


def create_ranking_dataframe(model_scores: pd.DataFrame, 
                           score_col: str = 'score',
                           approach_name: str = 'approach') -> pd.DataFrame:
    """
    Create a ranking DataFrame in the format expected by visualization scripts.
    
    Parameters:
    -----------
    model_scores : DataFrame with model-level aggregated scores
    score_col : Name of the score column to use for ranking
    approach_name : Name of the approach/strategy
    
    Returns:
    --------
    DataFrame with columns: model, score, CI, approach, [original_score if available]
    """
    ranking_df = pd.DataFrame()
    
    ranking_df['model'] = model_scores['model']
    ranking_df['score'] = model_scores[score_col]
    
    # Add CI if available
    ci_col = f'{score_col}_CI'
    if ci_col in model_scores.columns:
        ranking_df['CI'] = model_scores[ci_col]
    else:
        ranking_df['CI'] = ['(0.00, +0.00)'] * len(model_scores)
    
    ranking_df['approach'] = approach_name
    
    # Add original score if this is debiased data
    if score_col.startswith('score_debiased'):
        # If per-factor debiased, map back to its original factor column
        if score_col.startswith('score_debiased_'):
            factor_base = score_col.replace('score_debiased_', '')
            original_col_candidates = [f'{factor_base}_score', factor_base]
            for cand in original_col_candidates:
                if cand in model_scores.columns:
                    ranking_df['original_score'] = model_scores[cand]
                    break
        elif 'overall_score' in model_scores.columns:
            ranking_df['original_score'] = model_scores['overall_score']
        elif 'score' in model_scores.columns:
            ranking_df['original_score'] = model_scores['score']
    
    # Sort by score (descending)
    ranking_df = ranking_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return ranking_df


def load_judge_data_for_visualization(judge_dir: Union[str, Path], 
                                    approach: Optional[str] = None,
                                    convert_to_win_rates: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load all data for a judge, ready for visualization.
    
    Parameters:
    -----------
    judge_dir : Path to judge directory containing base_processed and base_debiased
    approach : Specific approach to load (if None, loads first available)
    
    Returns:
    --------
    Dictionary with:
    - 'original': DataFrame with original rankings
    - 'debiased': DataFrame with debiased rankings
    - 'merged': DataFrame with both original and debiased scores
    """
    judge_dir = Path(judge_dir)
    
    # Load original evaluations
    base_processed_dir = judge_dir / 'base_processed'
    original_df = load_original_evaluations(base_processed_dir)
    
    # Detect baseline model dynamically for this setting
    baseline_model = detect_baseline_model(judge_dir) or 'gpt-4-0314'
    
    # Load debiased scores
    # Find debiased directories based on approach
    if approach:
        # Look for specific approach directory
        base_debiased_dir = judge_dir / f'base_debiased_{approach}'
        if not base_debiased_dir.exists():
            # Fallback to old naming convention
            base_debiased_dir = judge_dir / 'base_debiased'
    else:
        # Look for any base_debiased_* directory
        debiased_dirs = list(judge_dir.glob('base_debiased_*'))
        if debiased_dirs:
            base_debiased_dir = debiased_dirs[0]
            # Extract approach name from directory
            approach = base_debiased_dir.name.replace('base_debiased_', '')
        else:
            # Fallback to old naming convention
            base_debiased_dir = judge_dir / 'base_debiased'
    
    if not base_debiased_dir.exists():
        print(f"Warning: No debiased data found at {base_debiased_dir}")
        # Return original data only
        original_agg = aggregate_scores_by_model(original_df)
        
        # Convert scores to win rates if requested
        if convert_to_win_rates:
            score_columns = [col for col in original_agg.columns 
                            if col.endswith('_score')]
            for col in score_columns:
                if col in original_agg.columns:
                    # Skip non-numeric columns
                    if not pd.api.types.is_numeric_dtype(original_agg[col]):
                        print(f"Warning: Skipping non-numeric column {col}")
                        continue
                        
                    # Build a temp DF with score and CI bounds if present
                    cols = ['model', col]
                    ci_lower_col = f'{col}_CI_lower'
                    ci_upper_col = f'{col}_CI_upper'
                    has_bounds = ci_lower_col in original_agg.columns and ci_upper_col in original_agg.columns
                    if has_bounds:
                        cols += [ci_lower_col, ci_upper_col]
                    temp_df = original_agg[cols].copy()
                    
                    # Convert score to win rates
                    # Preserve raw central scores for fixed-context CI conversion
                    raw_center = dict(zip(temp_df['model'], temp_df[col].values))
                    temp_df = convert_scores_to_win_rates(temp_df, score_column=col, baseline_model=baseline_model)
                    original_agg[col] = temp_df[col]
                    
                    # Convert CI if bounds available
                    if has_bounds:
                        temp_df = bootstrap_to_win_rate_ci(temp_df, score_column=col, baseline_model=baseline_model, reference_raw_scores=raw_center)
                        ci_col = f'{col}_CI'
                        original_agg[ci_col] = temp_df[ci_col]
                        original_agg[ci_lower_col] = temp_df[ci_lower_col]
                        original_agg[ci_upper_col] = temp_df[ci_upper_col]
        
        return {
            'original': create_ranking_dataframe(original_agg, 'overall_score', 'original'),
            'debiased': pd.DataFrame(),
            'merged': original_df
        }
    
    debiased_df = load_debiased_scores(base_debiased_dir)
    
    # Merge data
    merged_df = merge_original_and_debiased(original_df, debiased_df)
    
    # Aggregate by model
    aggregated_df = aggregate_scores_by_model(merged_df)
    
    # Convert scores to win rates if requested
    if convert_to_win_rates:
        # Convert all score columns to win rates in both aggregated and merged dataframes
        
        # First convert in aggregated_df
        score_columns = [col for col in aggregated_df.columns 
                        if col.endswith('_score') or col == 'score_debiased' or col.startswith('score_debiased_')]
        
        for col in score_columns:
            if col in aggregated_df.columns:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(aggregated_df[col]):
                    print(f"Warning: Skipping non-numeric column {col}")
                    continue
                    
                # Create temporary DataFrame for conversion
                temp_df = aggregated_df[['model', col]].copy()
                # Preserve raw central scores for fixed-context CI conversion
                raw_center = dict(zip(temp_df['model'], temp_df[col].values))
                
                # Also include CI bounds if available
                ci_lower_col = f'{col}_CI_lower'
                ci_upper_col = f'{col}_CI_upper'
                if ci_lower_col in aggregated_df.columns and ci_upper_col in aggregated_df.columns:
                    temp_df[ci_lower_col] = aggregated_df[ci_lower_col]
                    temp_df[ci_upper_col] = aggregated_df[ci_upper_col]
                
                # Convert scores to win rates
                temp_df = convert_scores_to_win_rates(temp_df, score_column=col, baseline_model=baseline_model)
                aggregated_df[col] = temp_df[col]
                
                # Convert CIs if bounds are available: transform lower/upper through the same mapping
                if ci_lower_col in temp_df.columns and ci_upper_col in temp_df.columns:
                    ci_col = f'{col}_CI'
                    # Use Arena-Hard utility to convert CI bounds to win-rate scale
                    temp_with_ci = bootstrap_to_win_rate_ci(temp_df, score_column=col, baseline_model=baseline_model, reference_raw_scores=raw_center)
                    # Copy back CI string and numeric bounds
                    aggregated_df[ci_col] = temp_with_ci[ci_col]
                    if ci_lower_col in aggregated_df.columns and ci_lower_col in temp_with_ci.columns:
                        aggregated_df[ci_lower_col] = temp_with_ci[ci_lower_col]
                    if ci_upper_col in aggregated_df.columns and ci_upper_col in temp_with_ci.columns:
                        aggregated_df[ci_upper_col] = temp_with_ci[ci_upper_col]
        
        # Also convert in merged_df for visualization consistency
        score_columns_merged = [col for col in merged_df.columns 
                               if col.endswith('_score') or col == 'score_debiased' or col.startswith('score_debiased_')]
        
        for col in score_columns_merged:
            if col in merged_df.columns:
                # Skip non-numeric columns
                if not pd.api.types.is_numeric_dtype(merged_df[col]):
                    print(f"Warning: Skipping non-numeric column {col} in merged_df")
                    continue
                    
                # Create temporary DataFrame with model means for each score
                model_means = merged_df.groupby('model')[col].mean().reset_index()
                model_means_converted = convert_scores_to_win_rates(model_means, score_column=col, baseline_model=baseline_model)
                
                # Map the converted scores back to the merged dataframe
                score_mapping = dict(zip(model_means_converted['model'], model_means_converted[col]))
                merged_df[col] = merged_df['model'].map(score_mapping)
    
    # Create ranking DataFrames
    original_ranking = create_ranking_dataframe(aggregated_df, 'overall_score', 'original')
    debiased_ranking = create_ranking_dataframe(aggregated_df, 'score_debiased', approach or 'debiased')
    
    # Also build factor-wise debiased rankings if available (prefer per-factor debiased columns)
    factor_cols = ['correctness_score', 'completeness_score', 'safety_score', 
                   'conciseness_score', 'style_score']
    factor_rankings: Dict[str, pd.DataFrame] = {}
    for fcol in factor_cols:
        if fcol in aggregated_df.columns:
            factor_base = fcol.replace('_score', '')
            debiased_factor_col = f'score_debiased_{factor_base}'
            # Prefer per-factor debiased if present; otherwise fallback to overall debiased
            used_debiased_col = debiased_factor_col if debiased_factor_col in aggregated_df.columns else 'score_debiased'
            factor_rankings[fcol] = create_ranking_dataframe(aggregated_df, used_debiased_col, approach or 'debiased')
            # Ensure original_score column reflects the factor's original
            if 'original_score' not in factor_rankings[fcol].columns and fcol in aggregated_df.columns:
                factor_rankings[fcol]['original_score'] = aggregated_df[fcol]

    return {
        'original': original_ranking,
        'debiased': debiased_ranking,
        'merged': merged_df,
        'aggregated': aggregated_df,
        'factor_rankings': factor_rankings
    }


def load_all_approaches_data(judge_dir: Union[str, Path]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load debiased data for all available approaches for a judge.
    
    Parameters:
    -----------
    judge_dir : Path to judge directory containing base_processed and base_debiased_*
    
    Returns:
    --------
    Dictionary mapping approach names to data dictionaries (as returned by load_judge_data_for_visualization)
    """
    judge_dir = Path(judge_dir)
    
    # Find all debiased directories
    debiased_dirs = list(judge_dir.glob('base_debiased_*'))
    
    # Also check for old naming convention
    old_style_dir = judge_dir / 'base_debiased'
    if old_style_dir.exists() and old_style_dir not in debiased_dirs:
        debiased_dirs.append(old_style_dir)
    
    results = {}
    
    for debiased_dir in debiased_dirs:
        # Extract approach name
        if debiased_dir.name == 'base_debiased':
            approach = 'default'
        else:
            approach = debiased_dir.name.replace('base_debiased_', '')
        
        # Load data for this approach
        try:
            data = load_judge_data_for_visualization(judge_dir, approach, convert_to_win_rates=True)
            if not data['debiased'].empty:
                results[approach] = data
        except Exception as e:
            print(f"Error loading approach {approach}: {e}")
    
    return results
