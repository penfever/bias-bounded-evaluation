"""
Construct Arena-Hard style battles from base_processed JSONL.
Mirrors show_result.get_battles_from_judgment for original (non-debiased) outcomes.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import json
import pandas as pd


LABEL_TO_WINNER = {
    'A=B': 'tie',
    'A>B': 'model_a',
    'A>>B': 'model_a_strong',  # strong indicates weighting
    'B>A': 'model_b',
    'B>>A': 'model_b_strong',
}


def build_original_battles_from_jsonl(base_processed_dir: Path, baseline_model: str,
                                      target_metric: str = 'score',
                                      weight_strong: int = 3,
                                      first_game_only: bool = False) -> pd.DataFrame:
    """
    Create battles from base_processed/*.jsonl using the Arena-Hard mapping.

    Each JSONL has fields: question_id, model, games=[{...}]. For each game we create a battle
    against the baseline_model with winner decided by target_metric label.
    """
    base_processed_dir = Path(base_processed_dir)
    rows: List[Dict] = []

    for jsonl in base_processed_dir.glob('*.jsonl'):
        try:
            with open(jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    qid = d.get('question_id')
                    model = d.get('model')
                    games = d.get('games') or []
                    if not games:
                        continue
                    # Game 1: baseline=A, model=B
                    g = games[0]
                    if target_metric not in g:
                        continue
                    label = g[target_metric]
                    winner_tag = LABEL_TO_WINNER.get(label)
                    weight = weight_strong if winner_tag in ('model_a_strong','model_b_strong') else 1
                    if winner_tag == 'model_a_strong':
                        winner = 'model_a'
                    elif winner_tag == 'model_b_strong':
                        winner = 'model_b'
                    elif winner_tag in ('model_a','model_b','tie'):
                        winner = winner_tag
                    else:
                        winner = 'tie'
                    if weight:
                        rows.extend([{'question_id': qid, 'model_a': baseline_model, 'model_b': model, 'winner': winner}] * weight)

                    if (not first_game_only) and len(games) > 1:
                        # Game 2: roles flipped
                        g2 = games[1]
                        if target_metric not in g2:
                            continue
                        label2 = g2[target_metric]
                        winner_tag2 = LABEL_TO_WINNER.get(label2)
                        weight2 = weight_strong if winner_tag2 in ('model_a_strong','model_b_strong') else 1
                        # With roles flipped, invert model_a/model_b mapping compared to game 1
                        if winner_tag2 == 'model_a_strong':
                            winner2 = 'model_b'  # 'A' is model in second game
                        elif winner_tag2 == 'model_b_strong':
                            winner2 = 'model_a'  # 'B' is baseline in second game
                        elif winner_tag2 == 'model_a':
                            winner2 = 'model_b'
                        elif winner_tag2 == 'model_b':
                            winner2 = 'model_a'
                        elif winner_tag2 == 'tie':
                            winner2 = 'tie'
                        else:
                            winner2 = 'tie'
                        if weight2:
                            rows.extend([{'question_id': qid, 'model_a': baseline_model, 'model_b': model, 'winner': winner2}] * weight2)
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=['question_id','model_a','model_b','winner'])
    return pd.DataFrame(rows)

