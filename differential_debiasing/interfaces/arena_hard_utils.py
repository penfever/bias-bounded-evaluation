"""
Utilities for working with Arena-Hard(-Auto) data:

- Parse prompts into (question, answer_a, answer_b)
- Extract verdict tokens and map to scores
- Load per-question judgment records from base_processed directories
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Iterable, Union, List
from pathlib import Path

# Canonical mapping from pairwise tokens to numeric scores
PAIRWISE_TO_SCORE: Dict[str, float] = {
    'A>>B': 1.0,
    'A>B': 2.0,
    'A=B': 3.0,
    'B>A': 4.0,
    'B>>A': 5.0,
}

SCORE_TO_PAIRWISE: Dict[float, str] = {v: k for k, v in PAIRWISE_TO_SCORE.items()}


def verdict_token_to_score(token: str) -> Optional[float]:
    token = token.strip()
    return PAIRWISE_TO_SCORE.get(token)


def extract_verdict_token(record: Dict[str, Any]) -> Optional[str]:
    """Extract a pairwise verdict token from Arena-Hard(-Auto) style records.

    Supports both top-level and nested-in-`games` schemas found in base_processed files.
    Tries these in order for each candidate container (record, then each game):
    - Direct field 'score' with a token like 'A>>B', 'A>B', 'A=B', 'B>A', 'B>>A'
    - Pattern [[A>>B]] style inside 'judgment' text
    - Loose token occurrences in 'judgment'
    """

    def _from_container(container: Dict[str, Any]) -> Optional[str]:
        score_field = container.get('score')
        if isinstance(score_field, str):
            token = score_field.strip()
            if token in PAIRWISE_TO_SCORE:
                return token

        # Numeric fallback (some files store debiased numeric only)
        num = container.get('score_debiased')
        if isinstance(num, (int, float)):
            token_from_num = SCORE_TO_PAIRWISE.get(float(num))
            if token_from_num:
                return token_from_num

        judgment = container.get('judgment')
        if isinstance(judgment, str):
            # Prefer explicit [[A>>B]] bracket form
            m = re.search(r"\[\[(A>>B|B>>A|A>B|B>A|A=B)\]\]", judgment)
            if m:
                return m.group(1)
            # Fallback: loose tokens (avoid matching inside words)
            m2 = re.search(r"\b(A>>B|B>>A|A>B|B>A|A=B)\b", judgment)
            if m2:
                return m2.group(1)
        return None

    # Check top-level first
    token = _from_container(record)
    if token:
        return token

    # Then check nested games (some files store judgments per game)
    games = record.get('games')
    if isinstance(games, Iterable):
        for g in games:
            if isinstance(g, dict):
                token = _from_container(g)
                if token:
                    return token

    return None


def parse_arena_hard_prompt(user_prompt: str) -> Tuple[str, str, str]:
    """Parse Arena-Hard-Auto prompt to extract question and two answers.

    Supports standard markers with robust regex and helpful errors.
    """
    # Normalize line endings just in case
    text = user_prompt

    # Question
    qm = re.search(r"<\|User Prompt\|>\s*\n(.*?)\n\s*<\|The Start of Assistant A\'s Answer\|>", text, re.DOTALL)
    if not qm:
        raise ValueError("Could not extract question from Arena-Hard prompt (missing User Prompt block)")
    question = qm.group(1).strip()

    # Assistant A
    am = re.search(r"<\|The Start of Assistant A\'s Answer\|>\s*\n(.*?)\n\s*<\|The End of Assistant A\'s Answer\|>", text, re.DOTALL)
    if not am:
        raise ValueError("Could not extract Assistant A's answer from Arena-Hard prompt")
    answer_a = am.group(1).strip()

    # Assistant B
    bm = re.search(r"<\|The Start of Assistant B\'s Answer\|>\s*\n(.*?)\n\s*<\|The End of Assistant B\'s Answer\|>", text, re.DOTALL)
    if not bm:
        raise ValueError("Could not extract Assistant B's answer from Arena-Hard prompt")
    answer_b = bm.group(1).strip()

    return question, answer_a, answer_b


# Precompiled regex for extracting final pairwise verdict pattern
ARENA_PAIRWISE_RE = re.compile(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]")


def extract_pairwise_from_judge_response(response: str) -> Optional[str]:
    """Extract the final pairwise verdict token from a judge response.

    Returns the last occurrence of one of the allowed tokens within [[...]] if present.
    """
    if not isinstance(response, str) or not response:
        return None
    matches = ARENA_PAIRWISE_RE.findall(response)
    return matches[-1] if matches else None


def build_pairwise_judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    """Build a complete judge prompt (system + pairwise content) for Arena-Hard style judging."""
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the responses "
        "provided by two AI assistants to the user prompt displayed below. You will be given assistant A's "
        "answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
        "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\n"
        "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\n"
        "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
        "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n"
        "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
        "1. Assistant A is significantly better: [[A>>B]]\n"
        "2. Assistant A is slightly better: [[A>B]]\n"
        "3. Tie, relatively the same: [[A=B]]\n"
        "4. Assistant B is slightly better: [[B>A]]\n"
        "5. Assistant B is significantly better: [[B>>A]]\n\n"
        "IMPORTANT: You must end your response with exactly one of the bracketed patterns above (e.g., [[A>B]]). Do not include any text after the final verdict pattern.\n\n"
        "Example output: \"My final verdict is tie: [[A=B]]\"."
    )

    template = (
        "<|User Prompt|>\n{question}\n\n"
        "<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n"
        "<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"
    )

    formatted = template.format(question=question, answer_a=answer_a, answer_b=answer_b)
    return system_prompt + "\n\n" + formatted


def load_judgment_data(base_dir: Union[str, Path], question_id: str, model_name: str) -> Dict[str, Any]:
    """Load the processed Arena-Hard record for a given question/model.

    Search strategy:
    - Iterate subdirectories matching '*-setting*'
    - Prefer 'base_processed' subdir; expect per-model JSONL files named '{model_name}.jsonl'
    - Return the first record with matching 'question_id'

    Raises FileNotFoundError if no matching record is found.
    """
    import json

    base_path = Path(base_dir)

    # If user passed a base_processed directory directly, try it first
    if base_path.name == 'base_processed' and base_path.is_dir():
        model_file = base_path / f"{model_name}.jsonl"
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                    except Exception:
                        continue
                    if data.get('question_id') == question_id:
                        return data

    # Otherwise, search within setting directories
    for setting_dir in base_path.glob("*-setting*"):
        base_processed_dir = setting_dir / "base_processed"
        if not base_processed_dir.exists():
            continue
        model_file = base_processed_dir / f"{model_name}.jsonl"
        if not model_file.exists():
            continue
        with open(model_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except Exception:
                    continue
                if data.get('question_id') == question_id:
                    return data

    raise FileNotFoundError(
        f"Could not find judgment data for question_id={question_id}, model={model_name} under {base_dir}"
    )


def setting_dir_for_judge(judge_name: str) -> Optional[str]:
    """Return expected InDepthAnalysis setting directory for a given judge name.

    Returns None if no explicit mapping is known.
    """
    j = judge_name.strip().lower()
    mapping = {
        'gpt-4o-mini': 'GPT-4o-mini-0718-setting1',
        'gpt-3.5-turbo': 'GPT-3.5-Turbo-0125-setting1',
        'deepseek-r1-32b': 'DeepSeek-R1-32B-setting1',
        'deepseek-r1-32b-gguf': 'DeepSeek-R1-32B-setting1',
        'qwq-32b': 'QwQ-32B-setting1',
        'qwq-32b-gguf': 'QwQ-32B-setting1',
    }
    return mapping.get(j)


def find_sample_data(base_path: Union[str, Path], judge_name: str, max_samples: int = 50):
    """Find sample evaluation data rows for sensitivity measurement.

    Returns a pandas DataFrame with columns: question_id, model, source_dir.
    """
    import json
    import pandas as pd

    base_path = Path(base_path)
    print(f"🔍 Searching for sample data in {base_path}...")

    expected = setting_dir_for_judge(judge_name)
    sample_rows: List[Dict[str, Any]] = []

    candidate_dirs: List[Path] = []
    if expected:
        cand = base_path / expected
        if cand.exists() and cand.is_dir():
            candidate_dirs = [cand]
        else:
            print(f"  ⚠️ Expected data source '{expected}' not found under {base_path}. Falling back to scan.")
    if not candidate_dirs:
        candidate_dirs = [d for d in base_path.glob("*-setting*") if d.is_dir()]

    for setting_dir in candidate_dirs:
        base_processed_dir = setting_dir / "base_processed"
        if not base_processed_dir.exists():
            continue
        print(f"  Found data source: {setting_dir.name}")

        # Load up to ~max_samples across up to 3 model files
        for jsonl_file in list(base_processed_dir.glob("*.jsonl"))[:3]:
            model_name = jsonl_file.stem
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count >= max_samples // 3:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        qid = obj.get('question_id')
                        if not qid:
                            continue
                        sample_rows.append({
                            'question_id': qid,
                            'model': model_name,
                            'source_dir': str(base_processed_dir)
                        })
                        count += 1
            except Exception as e:
                print(f"    ⚠️ Error reading {jsonl_file}: {e}")
                continue

        if sample_rows:
            break

    if not sample_rows:
        raise ValueError(f"No evaluation data found in {base_path}")

    df = pd.DataFrame(sample_rows)
    print(f"✅ Found {len(df)} sample evaluations across {df['model'].nunique()} models")
    return df


def raw_scores_to_elo(scores: Dict[str, float], 
                      scale: float = 400.0, 
                      base: float = 10.0, 
                      init_rating: float = 1000.0,
                      baseline_model: str = "gpt-4-0314") -> Dict[str, float]:
    """
    Convert raw scores (e.g., 1-10 scale) to ELO ratings.
    
    This is a simplified conversion that assumes scores map to win probabilities.
    For a score on 1-10 scale, we treat it as: win_prob = (score - 1) / 9
    
    Parameters:
    -----------
    scores : Dict mapping model names to raw scores
    scale : ELO scale parameter (default 400)
    base : ELO base parameter (default 10)
    init_rating : Initial ELO rating (default 1000)
    baseline_model : Model to use as baseline (default gpt-4-0314)
    
    Returns:
    --------
    Dict mapping model names to ELO ratings
    """
    # Convert scores to win probabilities (assuming 1-10 scale)
    # Adjust this mapping based on your actual score range
    win_probs = {}
    
    # Convert all scores to float first
    scores_float = {}
    for model, score in scores.items():
        try:
            scores_float[model] = float(score)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert score '{score}' for model '{model}' to float. "
                           f"Score type: {type(score)}, value: {repr(score)}")
    
    score_values = list(scores_float.values())
    
    # Detect score range
    min_score = min(score_values)
    max_score = max(score_values)
    
    # Handle case where all scores are the same
    if max_score == min_score:
        return {model: init_rating for model in scores_float}
    
    # Normalize to 0-1 win probability
    for model, score in scores_float.items():
        # Linear mapping from score range to win probability
        win_probs[model] = (score - min_score) / (max_score - min_score)
    
    # Convert win probabilities to ELO differences
    elo_ratings = {}
    
    # If baseline model exists, use it as anchor
    if baseline_model in scores_float:
        baseline_prob = win_probs[baseline_model]
        
        for model, prob in win_probs.items():
            if model == baseline_model:
                elo_ratings[model] = init_rating
            else:
                # ELO difference formula: ΔR = scale * log_base(p/(1-p))
                # where p is win probability of model vs baseline
                if prob == 1.0:
                    prob = 0.9999  # Avoid log(inf)
                if prob == 0.0:
                    prob = 0.0001  # Avoid log(0)
                
                # Calculate relative win probability
                p_vs_baseline = prob / (prob + baseline_prob)
                
                # Clamp to avoid division by zero or log(0)
                p_vs_baseline = max(0.001, min(0.999, p_vs_baseline))
                
                elo_diff = scale * np.log(p_vs_baseline / (1 - p_vs_baseline)) / np.log(base)
                elo_ratings[model] = init_rating + elo_diff
    else:
        # No baseline model - distribute ELO ratings proportionally
        # Map probabilities to ELO range centered at init_rating
        for model, prob in win_probs.items():
            # Map [0, 1] to approximately [-400, +400] ELO range
            elo_ratings[model] = init_rating + scale * (prob - 0.5)
    
    return elo_ratings


def predict_win_rate(elo_ratings: Dict[str, float], 
                     scale: float = 400.0, 
                     base: float = 10.0) -> pd.DataFrame:
    """
    Calculate pairwise win rates from ELO ratings.
    
    Parameters:
    -----------
    elo_ratings : Dict mapping model names to ELO ratings
    scale : ELO scale parameter (default 400)
    base : ELO base parameter (default 10)
    
    Returns:
    --------
    DataFrame with win rates for each model pair
    """
    names = sorted(list(elo_ratings.keys()))
    wins = {}
    
    for a in names:
        wins[a] = {}
        for b in names:
            if a == b:
                wins[a][b] = np.nan
            else:
                # ELO win probability formula
                ea = 1 / (1 + base ** ((elo_ratings[b] - elo_ratings[a]) / scale))
                wins[a][b] = ea

    df = pd.DataFrame(wins)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(elo_ratings: Dict[str, float], 
                        baseline: str = "gpt-4-0314",
                        scale: float = 400.0,
                        base: float = 10.0) -> Dict[str, float]:
    """
    Get win rates against a baseline model, scaled to 0-100.
    
    Parameters:
    -----------
    elo_ratings : Dict mapping model names to ELO ratings
    baseline : Baseline model name
    scale : ELO scale parameter
    base : ELO base parameter
    
    Returns:
    --------
    Dict mapping model names to win rates (0-100 scale) vs baseline
    """
    win_rate_table = predict_win_rate(elo_ratings, scale, base)
    
    if baseline not in win_rate_table.columns:
        # If baseline not found, return normalized scores
        print(f"Warning: Baseline model {baseline} not found. Using normalized scores.")
        max_elo = max(elo_ratings.values())
        min_elo = min(elo_ratings.values())
        range_elo = max_elo - min_elo if max_elo != min_elo else 1
        
        return {
            model: round(50 + 40 * (elo - (max_elo + min_elo) / 2) / range_elo, 2)
            for model, elo in elo_ratings.items()
        }
    
    win_rates = win_rate_table[baseline].fillna(0.5)
    return {model: round(rate * 100, 2) for model, rate in win_rates.items()}


def convert_scores_to_win_rates(model_scores: pd.DataFrame,
                                score_column: str = 'score',
                                baseline_model: str = 'gpt-4-0314') -> pd.DataFrame:
    """
    Convert a DataFrame with model scores to win rates on 0-100 scale.
    
    Parameters:
    -----------
    model_scores : DataFrame with columns ['model', score_column, ...]
    score_column : Name of the column containing scores to convert
    baseline_model : Model to use as baseline (50.0 win rate)
    
    Returns:
    --------
    DataFrame with score_column values replaced by win rates
    """
    # Create score dictionary
    scores_dict = model_scores.set_index('model')[score_column].to_dict()
    
    # Convert to ELO ratings
    elo_ratings = raw_scores_to_elo(scores_dict, baseline_model=baseline_model)
    
    # Convert to win rates
    win_rates = get_win_rate_column(elo_ratings, baseline=baseline_model)
    
    # Update DataFrame
    result_df = model_scores.copy()
    result_df[score_column] = result_df['model'].map(win_rates)
    
    # Ensure baseline model has exactly 50.0 if it exists
    if baseline_model in result_df['model'].values:
        result_df.loc[result_df['model'] == baseline_model, score_column] = 50.0
    
    return result_df


def bootstrap_to_win_rate_ci(model_scores: pd.DataFrame,
                             score_column: str = 'score',
                             baseline_model: str = 'gpt-4-0314',
                             confidence_level: float = 0.95) -> pd.DataFrame:
    """
    Convert confidence intervals to win rate scale using the Arena-Hard method.
    
    This converts the CI bounds (lower and upper) to win rates independently,
    then formats them as deltas from the center score.
    
    Parameters:
    -----------
    model_scores : DataFrame with columns ['model', score_column, f'{score_column}_CI_lower', f'{score_column}_CI_upper']
    score_column : Name of the score column
    baseline_model : Baseline model for win rate calculation
    confidence_level : Confidence level (not used, kept for compatibility)
    
    Returns:
    --------
    DataFrame with updated CI column in win rate scale
    """
    df = model_scores.copy()
    
    # Check if we have the numeric CI bounds
    ci_lower_col = f'{score_column}_CI_lower'
    ci_upper_col = f'{score_column}_CI_upper'
    
    if ci_lower_col not in df.columns or ci_upper_col not in df.columns:
        # If no CI bounds, return with default CI
        df[f'{score_column}_CI'] = '(-0.00, +0.00)'
        return df
    
    # Create DataFrames for lower and upper bounds with the bounds as the score
    lower_df = df[['model', ci_lower_col]].copy()
    lower_df = lower_df.rename(columns={ci_lower_col: score_column})
    lower_df = convert_scores_to_win_rates(lower_df, score_column=score_column, baseline_model=baseline_model)
    
    upper_df = df[['model', ci_upper_col]].copy()
    upper_df = upper_df.rename(columns={ci_upper_col: score_column})
    upper_df = convert_scores_to_win_rates(upper_df, score_column=score_column, baseline_model=baseline_model)
    
    # The main scores should already be converted
    # Calculate deltas from the converted score
    ci_strings = []
    for idx, row in df.iterrows():
        score = row[score_column]
        lower = lower_df.loc[idx, score_column]
        upper = upper_df.loc[idx, score_column]
        
        # Format as deltas
        lower_delta = lower - score
        upper_delta = upper - score
        ci_strings.append(f"({lower_delta:.2f}, +{upper_delta:.2f})")
    
    df[f'{score_column}_CI'] = ci_strings
    
    # Also update the numeric bounds
    df[ci_lower_col] = lower_df[score_column]
    df[ci_upper_col] = upper_df[score_column]
    
    return df
