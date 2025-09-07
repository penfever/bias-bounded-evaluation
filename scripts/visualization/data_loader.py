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


def get_score_mapping() -> Dict[str, int]:
    """Get the score mapping dictionary for Arena-Hard-Auto Likert scale."""
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


def load_original_evaluations(base_processed_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load all judge evaluations from JSONL files in base_processed directory.
    
    Returns DataFrame with columns: model, question_id, game_index, score, [factor_scores...]
    """
    base_processed_dir = Path(base_processed_dir)
    if not base_processed_dir.exists():
        raise ValueError(f"Base processed directory not found: {base_processed_dir}")
    
    score_mapping = get_score_mapping()
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
                            'score_debiased': float(data['score_debiased'])
                        }
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
    
    # Keep all original columns and add debiased score
    merged_df = pd.merge(
        original_df,
        debiased_df[merge_keys + ['score_debiased']],
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
        # Find all numeric columns ending with 'score' or 'score_debiased'
        score_cols = [col for col in df.columns 
                     if (col.endswith('score') or col.endswith('score_debiased')) 
                     and pd.api.types.is_numeric_dtype(df[col])]
    
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
        result_df[f'{col}_CI'] = result_df.apply(
            lambda row: f"({ci_lower.loc[row.name]:.2f}, {ci_upper.loc[row.name]:.2f})", 
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
    if 'score_debiased' in score_col and 'overall_score' in model_scores.columns:
        ranking_df['original_score'] = model_scores['overall_score']
    elif 'score_debiased' in score_col and 'score' in model_scores.columns:
        ranking_df['original_score'] = model_scores['score']
    
    # Sort by score (descending)
    ranking_df = ranking_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return ranking_df


def load_judge_data_for_visualization(judge_dir: Union[str, Path], 
                                    approach: Optional[str] = None) -> Dict[str, pd.DataFrame]:
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
    
    # Create ranking DataFrames
    original_ranking = create_ranking_dataframe(aggregated_df, 'overall_score', 'original')
    debiased_ranking = create_ranking_dataframe(aggregated_df, 'score_debiased', 
                                               approach or 'debiased')
    
    return {
        'original': original_ranking,
        'debiased': debiased_ranking,
        'merged': merged_df,
        'aggregated': aggregated_df
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
            data = load_judge_data_for_visualization(judge_dir, approach)
            if not data['debiased'].empty:
                results[approach] = data
        except Exception as e:
            print(f"Error loading approach {approach}: {e}")
    
    return results