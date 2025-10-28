"""
Utility functions for differential debiasing
"""

import math
import numbers
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Any, Dict
import warnings


def round_sig(x: float, sig: int = 3) -> float:
    """Round a float to the requested number of significant digits."""
    try:
        if x == 0 or not math.isfinite(float(x)):
            return float(x)
        return float(round(x, sig - int(math.floor(math.log10(abs(x)))) - 1))
    except Exception:
        return float(x)


def round_nested(obj: Any, sig: int = 3):
    """Recursively round floats within nested containers."""
    if isinstance(obj, float):
        return round_sig(obj, sig)
    if isinstance(obj, numbers.Number):
        return obj
    if isinstance(obj, list):
        return [round_nested(v, sig) for v in obj]
    if isinstance(obj, dict):
        return {k: round_nested(v, sig) for k, v in obj.items()}
    return obj


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
    
    Notes:
    ------
    Following Proposition 1 (splitting the failure budget), for a fixed split δ = δ_B + δ_Δ
    with δ_B = δ_Δ = δ/2 by default, the mechanism requires τ > Δ*₂(f,D)·sqrt(1/δ_Δ).
    When this holds, an upper bound on σ that satisfies the guarantee is:
        σ = (τ − Δ*₂(f,D)·sqrt(1/δ_Δ)) / sqrt( 2(d + 2√(d·log(1/δ_B)) + 2·log(1/δ_B)) ).
    This bound decreases as τ decreases; if τ is too small (≤ threshold), the guarantee
    cannot be met by adding noise — increasing σ would only increase the chance of violating τ.
    """
    if rms_sensitivity < 0:
        raise ValueError("rms_sensitivity must be non-negative")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if dimensionality <= 0:
        raise ValueError("dimensionality must be positive")
    
    # Split failure budget evenly: δ_B = δ_Δ = δ/2
    delta_split = float(delta) / 2.0
    
    # Threshold: Δ*₂(f,D) · sqrt(1/δ_Δ) = rms_sensitivity * sqrt(2/δ)
    # (since δ_Δ = δ/2)
    constraint_threshold = rms_sensitivity * np.sqrt(2.0 / float(delta))
    
    if tau <= constraint_threshold:
        raise ValueError(
            f"A-BB constraint violated: τ ({tau:.6f}) must be > "
            f"Δ*₂(f,D) * sqrt(1/δ_Δ) ({constraint_threshold:.6f}). "
            f"Either increase τ or reduce sensitivity."
        )
    
    # Headroom available for noise
    numerator = tau - constraint_threshold
    
    # Calculate denominator: sqrt(2(d + 2*sqrt(d*log(2/δ)) + 2*log(2/δ)))
    d = float(dimensionality)
    log_term = np.log(1.0 / delta_split)
    sqrt_term = 2.0 * np.sqrt(d * log_term)
    log_term_scaled = 2.0 * log_term
    
    denominator_inner = 2.0 * (d + sqrt_term + log_term_scaled)
    denominator = np.sqrt(denominator_inner)
    
    # Calculate final noise parameter
    sigma = numerator / denominator
    
    if sigma <= 0:
        # Should not happen when tau > threshold, but guard anyway
        return 0.0
    return float(sigma)


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


# --- RMS Sensitivity utilities ---

def rms_from_differences(diffs: Union[List[float], np.ndarray]) -> float:
    """
    Compute RMS from a collection of per-experiment differences.

    Equivalent to sqrt(mean(diff_i^2)). Returns 0.0 for empty input.
    """
    if diffs is None:
        return 0.0
    arr = np.asarray(list(diffs), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def context_adjusted_rms(total_rms: float, intrinsic_rms: float) -> float:
    """
    Compute context-adjusted RMS given total and intrinsic RMS values.

    Uses sqrt(max(0, total_rms^2 - intrinsic_rms^2)).
    """
    if total_rms is None or intrinsic_rms is None:
        raise ValueError("total_rms and intrinsic_rms must be provided")
    total = float(total_rms)
    intrinsic = float(intrinsic_rms)
    result = float(np.sqrt(max(0.0, total ** 2 - intrinsic ** 2)))
    # Optional debug logging of call-site and inputs
    try:
        import os as _os
        dbg = _os.getenv('A_BB_CONTEXT_RMS_DEBUG', '').strip().lower() in ('1', 'true', 'yes')
        if dbg:
            import inspect as _inspect
            fr = _inspect.stack()[1]
            caller_fn = getattr(fr, 'function', 'unknown')
            caller_file = getattr(fr, 'filename', 'unknown')
            caller_file = caller_file.split('/')[-1]
            print(
                f"🔎 context_adjusted_rms from {caller_file}:{caller_fn} | total={total:.4f}, intrinsic={intrinsic:.4f}, result={result:.4f}"
            )
    except Exception:
        # Never let debug logging break computation
        pass
    return result


def compute_abb_constraint_validation(tau: float,
                                      delta: float,
                                      sensitivity: float,
                                      score_range: Optional[float] = None,
                                      label: str = 'Δ*₂(f,D)') -> dict:
    """
    Compute A-BB constraint validation.

    When score_range is provided and positive, both the sensitivity and tau are
    normalized by that range before evaluating the constraint. The returned dict
    includes the raw and normalized values so callers can log or reuse either form.
    """
    tau = float(tau)
    delta = float(delta)
    sensitivity = float(sensitivity)
    rng = float(score_range) if score_range is not None else None

    if rng is not None and rng > 0:
        normalized_sensitivity = sensitivity / rng
        normalized_tau = tau / rng
        threshold = normalized_sensitivity * float(np.sqrt(2.0 / delta))
        margin = normalized_tau - threshold
        formula = 'τ̂ > Δ̂ * sqrt(2/δ)'
    else:
        normalized_sensitivity = None
        normalized_tau = None
        threshold = sensitivity * float(np.sqrt(2.0 / delta))
        margin = tau - threshold
        formula = f'τ > {label} * sqrt(2/δ)'

    return {
        'constraint_satisfied': bool(margin > 0),
        'tau': tau,
        'delta': delta,
        'combined_sensitivity': sensitivity,
        'normalized_tau': float(normalized_tau) if normalized_tau is not None else None,
        'normalized_sensitivity': float(normalized_sensitivity) if normalized_sensitivity is not None else None,
        'constraint_threshold': float(threshold),
        'margin': float(margin),
        'score_range': float(rng) if rng is not None else None,
        'constraint_formula': formula,
    }


def calculate_effective_alpha(bias_sensitivity: float,
                              score_range: float,
                              tau: float,
                              delta: float,
                              *,
                              shrink_alpha: Optional[float] = None,
                              target_tau: Optional[float] = None) -> float:
    """
    Resolve the effective shrinkage alpha consistent with the ABB mechanism.

    This mirrors the calculation used by ``DifferentialDebias.fit`` so callers
    (including analysis scripts) can reason about the same value before invoking
    the debiaser.
    """
    if shrink_alpha is not None:
        return float(shrink_alpha)

    if target_tau is None:
        return 1.0

    rng = float(score_range)
    if rng <= 0 or bias_sensitivity <= 0:
        return 1.0

    delta_star_norm = float(bias_sensitivity) / rng
    tau_norm = float(target_tau) / rng
    if delta_star_norm <= 0:
        return 1.0

    delta_factor = float(np.sqrt(2.0 / float(delta)))
    alpha_max = tau_norm / (delta_star_norm * delta_factor)
    return float(max(0.0, min(1.0, alpha_max)))


def estimate_s_mu_norm(n_samples: int,
                       *,
                       shrink_center: str = "mean",
                       dimensionality: Optional[int] = None) -> float:
    """
    Estimate the normalized shrinkage radius (S_μ) used in ABB prechecks.

    Currently this mirrors the heuristic used in the analysis workflow, which
    assumes an effective dimensionality of ``n_samples`` when none is provided.
    """
    if n_samples <= 0:
        return 0.0

    center = (shrink_center or "").lower()
    if center not in ("mean", "median"):
        return 0.0

    d_eff = dimensionality if dimensionality is not None else n_samples
    if d_eff <= 0:
        return 0.0

    return float(min(1.0, (float(d_eff) ** 0.5) / max(float(n_samples), 1.0)))


def build_abb_precheck(tau: float,
                       delta: float,
                       sensitivity: float,
                       score_range: Optional[float],
                       *,
                       alpha: float = 1.0,
                       s_mu_norm: float = 0.0,
                       label: str = 'Δ*₂(f,D)') -> Dict[str, float]:
    """
    Construct a canonical ABB precheck summary, reusing
    ``compute_abb_constraint_validation`` for normalized quantities.
    """
    effective_sensitivity = float(sensitivity) * float(alpha)
    validation = compute_abb_constraint_validation(
        tau=float(tau),
        delta=float(delta),
        sensitivity=effective_sensitivity,
        score_range=score_range,
        label=label,
    )

    delta_factor = float(np.sqrt(2.0 / float(delta)))
    normalized_sensitivity = validation.get('normalized_sensitivity')
    if normalized_sensitivity is None:
        a_component = effective_sensitivity * delta_factor
    else:
        a_component = float(normalized_sensitivity) * delta_factor

    b_component = float(s_mu_norm) * delta_factor

    validation.update(
        {
            'effective_sensitivity': effective_sensitivity,
            'alpha': float(alpha),
            's_mu_norm': float(s_mu_norm),
            'delta_factor': delta_factor,
            'A_component': float(a_component),
            'B_component': float(b_component),
        }
    )
    return validation


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
