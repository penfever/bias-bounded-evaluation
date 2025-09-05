"""
Test data helpers for creating synthetic judgment data
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def create_sample_judgment_data(n_samples: int = 100, random_seed: int = 42) -> pd.DataFrame:
    """Create sample judgment data with factor scores."""
    np.random.seed(random_seed)
    
    # Factor scores (representing different aspects like correctness, clarity, etc.)
    correctness_score = np.random.normal(6.0, 1.5, n_samples)
    clarity_score = np.random.normal(5.5, 1.2, n_samples)
    completeness_score = np.random.normal(6.2, 1.0, n_samples)
    
    # Overall score with some known relationship to factors
    overall_score = (
        0.5 * correctness_score + 
        0.3 * clarity_score + 
        0.2 * completeness_score + 
        np.random.normal(0, 0.5, n_samples)  # Some noise
    )
    
    # Clip to reasonable score range (1-10)
    correctness_score = np.clip(correctness_score, 1, 10)
    clarity_score = np.clip(clarity_score, 1, 10) 
    completeness_score = np.clip(completeness_score, 1, 10)
    overall_score = np.clip(overall_score, 1, 10)
    
    return pd.DataFrame({
        'correctness_score': correctness_score,
        'clarity_score': clarity_score,
        'completeness_score': completeness_score,
        'overall_score': overall_score,
        'score': overall_score  # Alias for compatibility
    })


def create_synthetic_judge_function(df: pd.DataFrame):
    """Create a synthetic judge function that returns judgments for a context."""
    
    def judge_function(context: Dict[str, Any]) -> List[float]:
        """Mock judge function that returns random scores based on context."""
        if isinstance(context, dict):
            # Return single judgment
            base_score = context.get('score', 6.0)
            return [base_score + np.random.normal(0, 0.2)]
        elif isinstance(context, pd.DataFrame):
            # Return judgments for all rows
            return context.get('score', pd.Series([6.0] * len(context))).tolist()
        else:
            # Default response
            return [6.0 + np.random.normal(0, 0.5)]
    
    return judge_function


def load_and_prepare_score_data() -> pd.DataFrame:
    """Mock function to replace the script import."""
    return create_sample_judgment_data(n_samples=200)