"""
Utility functions for differential debiasing
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
import warnings


def normalize_judgments(judgments: np.ndarray, 
                       score_min: Optional[float] = None,
                       score_max: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    Normalize judgment scores to [0, 1] range.
    
    Parameters:
    -----------
    judgments : np.ndarray
        Array of judgment scores
    score_min : float, optional
        Minimum score value. If None, uses min of judgments
    score_max : float, optional  
        Maximum score value. If None, uses max of judgments
        
    Returns:
    --------
    tuple : (normalized_judgments, min_val, max_val)
        Normalized scores and the min/max values used for normalization
    """
    judgments = np.asarray(judgments)
    
    # Handle empty array case
    if len(judgments) == 0:
        return judgments, np.nan, np.nan
    
    user_provided_range = score_min is not None and score_max is not None
    
    if score_min is None:
        score_min = float(judgments.min())
    if score_max is None:
        score_max = float(judgments.max())
        
    if score_max <= score_min:
        if user_provided_range:
            # User provided invalid range - this is an error
            raise ValueError(f"score_max ({score_max}) must be greater than score_min ({score_min})")
        else:
            # Auto-computed equal range (single value case) - handle gracefully
            normalized = np.zeros_like(judgments)
            return normalized, score_min, score_max
    
    normalized = (judgments - score_min) / (score_max - score_min)
    return normalized, score_min, score_max


def denormalize_judgments(normalized_judgments: np.ndarray,
                         score_min: float,
                         score_max: float) -> np.ndarray:
    """
    Convert normalized judgments back to original scale.
    
    Parameters:
    -----------
    normalized_judgments : np.ndarray
        Normalized judgment scores in [0, 1]
    score_min : float
        Original minimum score value
    score_max : float
        Original maximum score value
        
    Returns:
    --------
    np.ndarray : Denormalized judgment scores
    """
    return normalized_judgments * (score_max - score_min) + score_min


def calculate_noise_parameter(bias_sensitivity: float,
                            tau: float,
                            delta: float,
                            n_samples: int,
                            use_average_case: bool = True) -> float:
    """
    Calculate the noise parameter sigma for bias-bounded mechanism (original version).
    
    Parameters:
    -----------
    bias_sensitivity : float
        Bias sensitivity (Δ_B f) in same units as judgments
    tau : float
        Bias protection parameter
    delta : float
        Failure probability
    n_samples : int
        Number of samples in dataset
    use_average_case : bool
        Whether to use average-case bounds (recommended for datasets)
        
    Returns:
    --------
    float : Standard deviation for Gaussian noise
    """
    if bias_sensitivity <= 0:
        raise ValueError("bias_sensitivity must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    # Base sigma calculation
    sigma = (bias_sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / tau
    
    # Apply average-case optimization
    if use_average_case and n_samples > 1:
        sigma = sigma / np.sqrt(n_samples)
    
    return sigma


def calculate_abb_noise_parameter(rms_sensitivity: float,
                                tau: float,
                                delta: float,
                                dimensionality: int) -> float:
    """
    Calculate the noise parameter sigma for A-BB Gaussian mechanism.
    
    Implements the A-BB noise formula:
    σ = (τ - Δ*₂(f,D) * sqrt(2/δ)) / sqrt(2(d + 2*sqrt(d*log(2/δ)) + 2*log(2/δ)))
    
    Parameters:
    -----------
    rms_sensitivity : float
        Root-mean-squared sensitivity Δ*₂(f,D)
    tau : float
        Bias protection parameter
    delta : float
        Failure probability (total, will be split as δ_B = δ_Δ = δ/2)
    dimensionality : int
        Dimensionality of the judgment space (d)
        
    Returns:
    --------
    float : Standard deviation for Gaussian noise
    
    Raises:
    -------
    ValueError : If A-BB constraint τ > Δ*₂(f,D) * sqrt(2/δ) is not satisfied
    """
    if rms_sensitivity < 0:
        raise ValueError("rms_sensitivity must be non-negative")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if dimensionality <= 0:
        raise ValueError("dimensionality must be positive")
    
    # Split failure budget: δ_B = δ_Δ = δ/2
    delta_split = delta / 2
    
    # Check A-BB constraint: τ > Δ*₂(f,D) * sqrt(2/δ)
    constraint_threshold = rms_sensitivity * np.sqrt(2.0 / delta)
    if tau <= constraint_threshold:
        raise ValueError(
            f"A-BB constraint violated: τ ({tau:.6f}) must be > "
            f"Δ*₂(f,D) * sqrt(2/δ) ({constraint_threshold:.6f}). "
            f"Either increase τ or reduce sensitivity."
        )
    
    # Calculate numerator: τ - Δ*₂(f,D) * sqrt(2/δ)  
    numerator = tau - constraint_threshold
    
    # Calculate denominator: sqrt(2(d + 2*sqrt(d*log(2/δ)) + 2*log(2/δ)))
    d = float(dimensionality)
    log_term = np.log(2.0 / delta_split)
    sqrt_term = 2.0 * np.sqrt(d * log_term)
    log_term_scaled = 2.0 * log_term
    
    denominator_inner = 2.0 * (d + sqrt_term + log_term_scaled)
    denominator = np.sqrt(denominator_inner)
    
    # Calculate final noise parameter
    sigma = numerator / denominator
    
    if sigma <= 0:
        raise ValueError(
            f"Calculated noise parameter σ = {sigma:.6f} is non-positive. "
            f"This indicates the A-BB constraint is barely satisfied. "
            f"Consider increasing τ."
        )
    
    return sigma


def validate_input_array(arr: Union[np.ndarray, List, pd.Series], 
                        name: str = "array") -> np.ndarray:
    """
    Validate and convert input to numpy array.
    
    Parameters:
    -----------
    arr : array-like
        Input array to validate
    name : str
        Name for error messages
        
    Returns:
    --------
    np.ndarray : Validated array
    """
    arr = np.asarray(arr)
    
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    
    if not np.isfinite(arr).all():
        warnings.warn(f"{name} contains non-finite values")
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError(f"{name} contains only non-finite values")
    
    return arr


def clip_to_range(values: np.ndarray, 
                 min_val: float, 
                 max_val: float) -> np.ndarray:
    """
    Clip values to specified range.
    
    Parameters:
    -----------
    values : np.ndarray
        Values to clip
    min_val : float
        Minimum allowed value
    max_val : float
        Maximum allowed value
        
    Returns:
    --------
    np.ndarray : Clipped values
    """
    return np.clip(values, min_val, max_val)


def check_bias_parameters(tau: float, delta: float) -> None:
    """
    Validate bias protection parameters.
    
    Parameters:
    -----------
    tau : float
        Bias protection parameter (should be > 0)
    delta : float  
        Failure probability (should be in (0, 1))
    """
    if not isinstance(tau, (int, float)) or tau <= 0:
        raise ValueError(f"tau must be a positive number, got {tau}")
    
    if not isinstance(delta, (int, float)) or not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def estimate_r_squared_from_scores(df: pd.DataFrame, 
                                  factor_columns: List[str],
                                  target_column: str = 'score') -> float:
    """
    Estimate R² from factor scores using linear regression.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing scores
    factor_columns : List[str]
        Names of factor score columns
    target_column : str
        Name of target/overall score column
        
    Returns:
    --------
    float : R² value
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # Drop rows with missing values
    subset_cols = factor_columns + [target_column]
    df_clean = df[subset_cols].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data rows after removing missing values")
    
    X = df_clean[factor_columns].values
    y = df_clean[target_column].values
    
    # Fit linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    return r2_score(y, y_pred)