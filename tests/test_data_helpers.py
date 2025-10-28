"""
Shared helpers for creating synthetic judgment datasets used across tests.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def create_sample_judgment_data(n_samples: int = 100, random_seed: int = 42) -> pd.DataFrame:
    """Create sample judgment data with factor and overall scores."""
    np.random.seed(random_seed)

    correctness_score = np.random.normal(6.0, 1.5, n_samples)
    clarity_score = np.random.normal(5.5, 1.2, n_samples)
    completeness_score = np.random.normal(6.2, 1.0, n_samples)

    overall_score = (
        0.5 * correctness_score
        + 0.3 * clarity_score
        + 0.2 * completeness_score
        + np.random.normal(0, 0.5, n_samples)
    )

    correctness_score = np.clip(correctness_score, 1, 10)
    clarity_score = np.clip(clarity_score, 1, 10)
    completeness_score = np.clip(completeness_score, 1, 10)
    overall_score = np.clip(overall_score, 1, 10)

    return pd.DataFrame(
        {
            "correctness_score": correctness_score,
            "clarity_score": clarity_score,
            "completeness_score": completeness_score,
            "overall_score": overall_score,
            "score": overall_score,
        }
    )


def create_synthetic_judge_function(df: pd.DataFrame):
    """Create a synthetic judge callable that returns judgments for a context."""

    def judge_function(context: Dict[str, Any]) -> List[float]:
        if isinstance(context, dict):
            base_score = context.get("score", 6.0)
            return [base_score + np.random.normal(0, 0.2)]
        if isinstance(context, pd.DataFrame):
            return context.get("score", pd.Series([6.0] * len(context))).tolist()
        return [6.0 + np.random.normal(0, 0.5)]

    return judge_function


def load_and_prepare_score_data(
    dataset_id: Optional[str] = None,
    base_path: Any = None,
    *,
    n_samples: int = 200,
    random_seed: int = 42,
    **_: Any,
) -> pd.DataFrame:
    """
    Mock loader compatible with legacy test signatures.

    Parameters are accepted for compatibility with archived tests that used
    real Arena-Hard paths, but they are ignored in favor of synthetic data.
    """
    return create_sample_judgment_data(n_samples=n_samples, random_seed=random_seed)
