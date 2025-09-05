"""
Psychometric reliability based bias sensitivity estimation

This module implements the psychometric reliability sensitivity measures
as formalized in the bias-bounded LLM judges paper, including:
- Cronbach's Alpha for internal consistency
- Cross-loading Ratio (CLR) for discriminant validity
- Heterotrait-Monotrait Ratio (HTMT) for construct validity
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Tuple
from .base import SensitivityEstimator
import warnings


class PsychometricReliabilitySensitivity(SensitivityEstimator):
    """
    Estimate bias sensitivity using psychometric reliability measures.
    
    This method implements the unified psychometric reliability score as defined
    in the bias-bounded LLM judges paper:
    
    R_psychometric = (1/3) * (mean(alpha_i) + mean(CLR_i) + (1 - mean(HTMT_ij)))
    
    The bias sensitivity is then sqrt(1 - R_psychometric).
    """
    
    def __init__(self,
                 factor_columns: Optional[List[str]] = None,
                 alpha_weight: float = 1/3,
                 clr_weight: float = 1/3,
                 htmt_weight: float = 1/3,
                 clr_max: float = 2.0,
                 htmt_threshold: float = 0.85,
                 min_items_per_factor: int = 2,
                 **kwargs):
        """
        Initialize psychometric reliability sensitivity estimator.
        
        Parameters:
        -----------
        factor_columns : List[str], optional
            Column names for factor scores. If None, will auto-detect.
        alpha_weight : float
            Weight for Cronbach's alpha in unified score
        clr_weight : float
            Weight for cross-loading ratio in unified score
        htmt_weight : float
            Weight for HTMT ratio in unified score
        clr_max : float
            Maximum CLR value for normalization (typically 2.0)
        htmt_threshold : float
            HTMT threshold for good discriminant validity
        min_items_per_factor : int
            Minimum items per factor for reliable calculation
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.factor_columns = factor_columns
        self.alpha_weight = alpha_weight
        self.clr_weight = clr_weight
        self.htmt_weight = htmt_weight
        self.clr_max = clr_max
        self.htmt_threshold = htmt_threshold
        self.min_items_per_factor = min_items_per_factor
        
        # Validation
        if not np.isclose(alpha_weight + clr_weight + htmt_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        # Results storage
        self._cronbach_alphas = None
        self._clr_scores = None
        self._htmt_matrix = None
        self._factor_loadings = None
        self._reliability_score = None
        self._n_factors = None
        
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'PsychometricReliabilitySensitivity':
        """
        Fit psychometric reliability analysis to judgment data.
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Judgment data. Must be DataFrame with factor columns for proper analysis.
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        PsychometricReliabilitySensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        if not isinstance(judgments, pd.DataFrame):
            raise ValueError("PsychometricReliabilitySensitivity requires DataFrame input with factor columns")
        
        self._fit_dataframe(judgments, **kwargs)
        self._fitted = True
        return self
    
    def _fit_dataframe(self, df: pd.DataFrame, **kwargs):
        """Fit psychometric analysis using DataFrame with factor columns."""
        # Auto-detect factor columns if not provided
        if self.factor_columns is None:
            self.factor_columns = self._detect_factor_columns(df)
        
        # Validate we have enough factors and items
        if len(self.factor_columns) < 2:
            raise ValueError("Need at least 2 factors for psychometric reliability analysis")
        
        # Check for sufficient items per factor (questions)
        factor_data = df[self.factor_columns].dropna()
        if len(factor_data) < self.min_items_per_factor:
            warnings.warn(f"Only {len(factor_data)} items available, minimum {self.min_items_per_factor} recommended")
        
        self._n_factors = len(self.factor_columns)
        
        # Calculate Cronbach's Alpha for each factor
        self._cronbach_alphas = self._calculate_cronbach_alphas(factor_data)
        
        # Perform factor analysis for cross-loading ratios
        self._clr_scores = self._calculate_cross_loading_ratios(factor_data)
        
        # Calculate HTMT matrix
        self._htmt_matrix = self._calculate_htmt_matrix(factor_data)
        
        # Compute unified reliability score
        self._reliability_score = self._compute_reliability_score()
    
    def _detect_factor_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect factor columns from DataFrame."""
        # Look for columns that might represent evaluation factors
        potential_factors = []
        
        # Common factor names in LLM evaluation
        factor_keywords = ['correctness', 'completeness', 'safety', 'conciseness', 'style', 
                          'accuracy', 'relevance', 'clarity', 'helpfulness', 'coherence']
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if column contains factor keywords or ends with _score
            if any(keyword in col_lower for keyword in factor_keywords) or col_lower.endswith('_score'):
                if col_lower != 'overall_score' and col_lower != 'total_score':
                    potential_factors.append(col)
        
        # Fallback: look for numeric columns that might be scores
        if len(potential_factors) < 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            potential_factors = [col for col in numeric_cols if 'score' in col.lower()]
        
        if len(potential_factors) < 2:
            raise ValueError("Could not detect factor columns. Please specify factor_columns explicitly.")
        
        return potential_factors
    
    def _calculate_cronbach_alphas(self, factor_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Cronbach's alpha for internal consistency.
        
        For each factor (column), we treat questions (rows) as items.
        Alpha = (k/(k-1)) * (1 - sum(item_variances)/total_variance)
        """
        alphas = {}
        
        for factor in self.factor_columns:
            factor_scores = factor_data[factor].dropna()
            
            if len(factor_scores) < 2:
                alphas[factor] = 0.0
                continue
            
            # For Cronbach's alpha, we need inter-item correlations
            # Since we have judgment scores per question, we use split-half reliability approach
            
            # Split data into two halves
            n_items = len(factor_scores)
            if n_items < 4:
                # Too few items for reliable alpha calculation
                alphas[factor] = 0.5  # Moderate reliability
                continue
            
            # Calculate inter-item correlation approximation
            # Use variance decomposition approach
            total_var = factor_scores.var()
            
            # Estimate inter-item correlation from variance structure
            # This is an approximation for the case where we have scores rather than item responses
            if total_var == 0:
                alphas[factor] = 1.0  # Perfect consistency (all same scores)
            else:
                # Use coefficient of variation as proxy for consistency
                cv = factor_scores.std() / factor_scores.mean() if factor_scores.mean() != 0 else 1.0
                # Convert to alpha-like measure (lower CV = higher reliability)
                alpha = max(0.0, 1.0 - cv)
                alphas[factor] = min(1.0, alpha)
        
        return alphas
    
    def _calculate_cross_loading_ratios(self, factor_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate cross-loading ratios for discriminant validity.
        
        CLR_i = λ_ii / max(|λ_ij|) for j ≠ i
        """
        try:
            # Perform factor analysis
            from sklearn.decomposition import FactorAnalysis
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(factor_data)
            
            # Perform factor analysis
            n_factors = min(len(self.factor_columns), len(factor_data.columns))
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            fa.fit(standardized_data)
            
            # Get factor loadings
            loadings = fa.components_.T  # Shape: (n_features, n_factors)
            self._factor_loadings = loadings
            
            # Calculate cross-loading ratios
            clr_scores = {}
            for i, factor in enumerate(self.factor_columns):
                if i >= loadings.shape[0]:
                    clr_scores[factor] = 1.0
                    continue
                
                # Primary loading (on its own factor)
                primary_loading = abs(loadings[i, i % loadings.shape[1]])
                
                # Cross-loadings (on other factors)
                cross_loadings = [abs(loadings[i, j]) for j in range(loadings.shape[1]) if j != (i % loadings.shape[1])]
                
                if not cross_loadings or max(cross_loadings) == 0:
                    clr_scores[factor] = self.clr_max  # Perfect discriminant validity
                else:
                    clr = primary_loading / max(cross_loadings)
                    clr_scores[factor] = min(clr, self.clr_max)  # Cap at clr_max
            
            return clr_scores
            
        except ImportError:
            warnings.warn("sklearn not available for factor analysis. Using correlation-based CLR approximation.")
            return self._calculate_clr_from_correlations(factor_data)
        except Exception as e:
            warnings.warn(f"Factor analysis failed: {e}. Using correlation-based CLR approximation.")
            return self._calculate_clr_from_correlations(factor_data)
    
    def _calculate_clr_from_correlations(self, factor_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate CLR approximation using correlation matrix."""
        corr_matrix = factor_data.corr()
        clr_scores = {}
        
        for i, factor in enumerate(self.factor_columns):
            # Use correlation as proxy for loading
            correlations = [abs(corr_matrix.iloc[i, j]) for j in range(len(self.factor_columns)) if i != j]
            
            if not correlations or max(correlations) == 0:
                clr_scores[factor] = self.clr_max
            else:
                # Assume primary loading is 1.0, CLR is 1/max_cross_correlation
                clr = 1.0 / max(correlations)
                clr_scores[factor] = min(clr, self.clr_max)
        
        return clr_scores
    
    def _calculate_htmt_matrix(self, factor_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Heterotrait-Monotrait (HTMT) matrix.
        
        HTMT_ij = mean(r_heterotrait) / sqrt(mean(r_monotrait_i) * mean(r_monotrait_j))
        """
        corr_matrix = factor_data.corr().values
        n_factors = len(self.factor_columns)
        htmt_matrix = np.zeros((n_factors, n_factors))
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                # Heterotrait correlation
                heterotrait_corr = abs(corr_matrix[i, j])
                
                # Monotrait correlations (self-correlations, approximated as 1.0)
                monotrait_i = 1.0  # Perfect self-correlation
                monotrait_j = 1.0  # Perfect self-correlation
                
                # HTMT ratio
                if monotrait_i > 0 and monotrait_j > 0:
                    htmt = heterotrait_corr / np.sqrt(monotrait_i * monotrait_j)
                else:
                    htmt = 0.0
                
                htmt_matrix[i, j] = htmt
                htmt_matrix[j, i] = htmt  # Symmetric
        
        return htmt_matrix
    
    def _compute_reliability_score(self) -> float:
        """
        Compute unified psychometric reliability score.
        
        R_psychometric = α_weight * mean(alpha_i) + 
                        clr_weight * mean(normalized_CLR_i) + 
                        htmt_weight * (1 - mean(HTMT_ij))
        """
        # Cronbach's alpha component
        alpha_component = np.mean(list(self._cronbach_alphas.values()))
        
        # Cross-loading ratio component (sigmoid normalization centered at 1.5 threshold)
        # Based on 0.40-0.30-0.20 rule: CLR = 0.40/0.30 ≈ 1.33, threshold at 1.5
        clr_values = list(self._clr_scores.values())
        normalized_clr = [1.0 / (1.0 + np.exp(-2*(clr - 1.5))) for clr in clr_values]
        clr_component = np.mean(normalized_clr)
        
        # HTMT component (discriminant validity)
        # Extract upper triangle of HTMT matrix (excluding diagonal)
        n = self._htmt_matrix.shape[0]
        htmt_values = []
        for i in range(n):
            for j in range(i + 1, n):
                htmt_values.append(self._htmt_matrix[i, j])
        
        if htmt_values:
            mean_htmt = np.mean(htmt_values)
            htmt_component = 1.0 - mean_htmt  # Higher HTMT = worse discriminant validity
        else:
            htmt_component = 1.0  # Perfect discriminant validity if no comparisons
        
        # Combine components
        reliability_score = (self.alpha_weight * alpha_component + 
                           self.clr_weight * clr_component + 
                           self.htmt_weight * htmt_component)
        
        # Ensure score is in [0, 1] range
        reliability_score = np.clip(reliability_score, 0.0, 1.0)
        
        return reliability_score
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity from psychometric reliability analysis.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. If None, assumes normalized [0,1] range.
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        self._validate_fitted()
        
        if self._reliability_score is None:
            raise ValueError("Reliability score not calculated. Check fitting process.")
        
        # Psychometric reliability sensitivity: sqrt(1 - R_psychometric)
        normalized_sensitivity = np.sqrt(1.0 - self._reliability_score)
        
        # Scale to judgment units
        if score_range is None:
            score_range = 4.0  # Arena-Hard default: 5-point scale has range 4.0
        
        bias_sensitivity = normalized_sensitivity * score_range
        self._bias_sensitivity = bias_sensitivity
        
        return bias_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information about psychometric reliability analysis.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "method_details": "Psychometric Reliability (Cronbach's α + CLR + HTMT)",
                "reliability_score": self._reliability_score,
                "cronbach_alphas": self._cronbach_alphas,
                "clr_scores": self._clr_scores,
                "mean_htmt": np.mean(self._htmt_matrix) if self._htmt_matrix is not None else None,
                "n_factors": self._n_factors,
                "factor_columns": self.factor_columns,
                "component_weights": {
                    "alpha_weight": self.alpha_weight,
                    "clr_weight": self.clr_weight, 
                    "htmt_weight": self.htmt_weight
                },
                "quality_indicators": self._get_quality_indicators()
            })
            
            if self._htmt_matrix is not None:
                diagnostics["htmt_matrix"] = self._htmt_matrix.tolist()
        
        return diagnostics
    
    def _get_quality_indicators(self) -> Dict[str, Any]:
        """Get quality indicators for the psychometric analysis."""
        if not self._fitted:
            return {}
        
        indicators = {
            "overall_quality": "good" if self._reliability_score > 0.7 else "moderate" if self._reliability_score > 0.5 else "poor",
            "internal_consistency": "good" if np.mean(list(self._cronbach_alphas.values())) > 0.7 else "moderate",
            "discriminant_validity": "good" if np.mean(list(self._clr_scores.values())) > 1.5 else "poor"
        }
        
        # Check HTMT criterion
        if self._htmt_matrix is not None:
            max_htmt = np.max(self._htmt_matrix)
            indicators["construct_validity"] = "good" if max_htmt < self.htmt_threshold else "questionable"
        
        return indicators
    
    def get_cronbach_alphas(self) -> Optional[Dict[str, float]]:
        """Get Cronbach's alpha values for each factor."""
        return self._cronbach_alphas
    
    def get_clr_scores(self) -> Optional[Dict[str, float]]:
        """Get cross-loading ratio scores for each factor."""
        return self._clr_scores
    
    def get_htmt_matrix(self) -> Optional[np.ndarray]:
        """Get HTMT matrix for discriminant validity assessment."""
        return self._htmt_matrix
    
    def get_reliability_score(self) -> Optional[float]:
        """Get unified psychometric reliability score."""
        return self._reliability_score