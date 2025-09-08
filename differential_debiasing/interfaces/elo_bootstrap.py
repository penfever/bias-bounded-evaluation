"""
ELO + Bootstrap utilities aligned with Arena-Hard show_result.py semantics.

Stage 1 (Track A):
 - Provide compute_mle_elo (logistic regression, SCALE=400, BASE=10, anchor baseline=1000)
 - Provide bootstrap_elo to obtain median and 95% interval in ELO space
 - Provide win-rate conversion vs baseline (0-100 scale)

TODO (Stage 2):
 - Add functions to construct original battles from base_processed JSONL mirroring show_result
 - Add options for A>>B/B>>A weights and dual-game rules for original battles

TODO (Stage 3):
 - Support Bayesian hierarchical bootstrap path (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class EloParams:
    scale: float = 400.0
    base: float = 10.0
    init_rating: float = 1000.0
    baseline_model: str = "gpt-4-0314"


def compute_mle_elo(battles: pd.DataFrame, params: EloParams) -> pd.Series:
    """
    Compute MLE ELO via logistic regression, mirroring show_result.py.

    battles: DataFrame with columns ['model_a', 'model_b', 'winner'] where winner is 'model_a'|'model_b'|'tie'.
    Returns: pd.Series of ELO ratings indexed by model name.
    """
    # Build model index
    models = pd.concat([battles["model_a"], battles["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # Duplicate battles (as in show_result)
    df = pd.concat([battles, battles], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    logB = np.log(params.base)
    X[np.arange(n), models[df["model_a"]]] = +logB
    X[np.arange(n), models[df["model_b"]]] = -logB

    # Build Y with tie handling (first half)
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0
    tie_idx = (df["winner"] == "tie")
    # Mark half of ties as model_a wins (the duplicated half will imply model_b)
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8, max_iter=1000)
    lr.fit(X, Y)

    elo_scores = params.scale * lr.coef_[0] + params.init_rating

    # Anchor baseline
    idx = models.get(params.baseline_model, None)
    if idx is not None:
        elo_scores = elo_scores + (params.init_rating - elo_scores[idx])

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def bootstrap_elo(battles: pd.DataFrame, params: EloParams, num_rounds: int = 100,
                  random_state: Optional[int] = 42) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Standard bootstrap of ELO estimates.

    Returns: (median, q025, q975) Series indexed by model.
    """
    rng = np.random.RandomState(random_state)
    rows = []
    for _ in range(num_rounds):
        sample = battles.sample(frac=1.0, replace=True, random_state=rng.randint(0, 1_000_000))
        rows.append(compute_mle_elo(sample, params))
    df = pd.DataFrame(rows)
    med = df.median().sort_values(ascending=False)
    q025 = df.quantile(0.025)
    q975 = df.quantile(0.975)
    return med, q025, q975


def predict_win_rate(elo: Dict[str, float] | pd.Series, params: EloParams) -> pd.DataFrame:
    """Pairwise win rates from ELO ratings (as in show_result)."""
    if isinstance(elo, pd.Series):
        ratings = elo.to_dict()
    else:
        ratings = dict(elo)
    names = sorted(ratings.keys())
    wins = {}
    for a in names:
        wins[a] = {}
        for b in names:
            ea = 1 / (1 + params.base ** ((ratings[b] - ratings[a]) / params.scale))
            wins[a][b] = ea
    df = pd.DataFrame(wins)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def get_win_rate_column(elo: Dict[str, float] | pd.Series, params: EloParams) -> pd.Series:
    """Win rates vs baseline, scaled 0-100."""
    table = predict_win_rate(elo, params)
    col = params.baseline_model
    if col not in table.columns:
        # If baseline missing, return normalized values centered on median
        s = pd.Series(elo if isinstance(elo, dict) else elo.to_dict())
        s = 50 + 40 * (s - s.mean()) / (s.std() + 1e-9)
        return s.clip(0, 100)
    return table[col].fillna(0.5).apply(lambda x: round(x * 100, 2))


# TODO (Stage 2): implement original battle construction mirroring show_result.get_battles_from_judgment

