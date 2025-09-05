"""
Schematic adherence based bias sensitivity estimation

This module implements the schematic adherence sensitivity measures
as formalized in the bias-bounded LLM judges paper, including:
- Linear regression analysis between factor scores and overall judgment
- Polynomial regression with quadratic terms and interactions
- Context-dependent schematic pattern analysis through clustering
- Integration bias metrics (weight disparity, entropy, context stability)
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Tuple
from .base import SensitivityEstimator
import warnings


class SchematicAdherenceSensitivity(SensitivityEstimator):
    """
    Estimate bias sensitivity using schematic adherence analysis.
    
    This method analyzes how well overall judgments follow from explicit
    factor scores, implementing the approach defined in the bias-bounded
    LLM judges paper:
    
    R²_schematic = max(R²_linear, R²_polynomial)
    Sensitivity = sqrt(1 - R²_schematic)
    """
    
    def __init__(self,
                 factor_columns: Optional[List[str]] = None,
                 target_column: str = 'overall_score',
                 use_polynomial: bool = True,
                 use_interactions: bool = True,
                 use_clustering: bool = False,
                 n_clusters: int = 3,
                 cluster_column: Optional[str] = None,
                 min_samples_per_cluster: int = 10,
                 regularization_alpha: float = 0.01,
                 **kwargs):
        """
        Initialize schematic adherence sensitivity estimator.
        
        Parameters:
        -----------
        factor_columns : List[str], optional
            Column names for factor scores. If None, will auto-detect.
        target_column : str
            Column name for overall/target score
        use_polynomial : bool
            Whether to include quadratic terms in regression
        use_interactions : bool
            Whether to include factor interaction terms
        use_clustering : bool
            Whether to perform context-dependent clustering analysis
        n_clusters : int
            Number of clusters for context-dependent analysis
        cluster_column : str, optional
            Column to use for clustering (e.g., question_id, domain)
        min_samples_per_cluster : int
            Minimum samples required per cluster
        regularization_alpha : float
            L2 regularization parameter for polynomial regression
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.factor_columns = factor_columns
        self.target_column = target_column
        self.use_polynomial = use_polynomial
        self.use_interactions = use_interactions
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        self.cluster_column = cluster_column
        self.min_samples_per_cluster = min_samples_per_cluster
        self.regularization_alpha = regularization_alpha
        
        # Results storage
        self._linear_r2 = None
        self._polynomial_r2 = None
        self._schematic_r2 = None
        self._linear_weights = None
        self._polynomial_weights = None
        self._cluster_weights = None
        self._integration_bias_metrics = None
        self._n_factors = None
        self._regression_models = {}
        
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'SchematicAdherenceSensitivity':
        """
        Fit schematic adherence analysis to judgment data.
        
        Parameters:
        -----------
        judgments : DataFrame or array-like
            Judgment data. Must be DataFrame with factor and target columns.
        **kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        SchematicAdherenceSensitivity : Self for method chaining
        """
        judgments = self._validate_input(judgments)
        
        if not isinstance(judgments, pd.DataFrame):
            raise ValueError("SchematicAdherenceSensitivity requires DataFrame input with factor and target columns")
        
        self._fit_dataframe(judgments, **kwargs)
        self._fitted = True
        return self
    
    def _fit_dataframe(self, df: pd.DataFrame, **kwargs):
        """Fit schematic adherence analysis using DataFrame."""
        # Auto-detect factor columns if not provided
        if self.factor_columns is None:
            self.factor_columns = self._detect_factor_columns(df)
        
        # Validate required columns exist
        missing_cols = [col for col in self.factor_columns + [self.target_column] 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Clean data
        analysis_cols = self.factor_columns + [self.target_column]
        clean_df = df[analysis_cols].dropna()
        
        if len(clean_df) < len(self.factor_columns) + 1:
            raise ValueError(f"Insufficient clean samples ({len(clean_df)}) for regression analysis")
        
        self._n_factors = len(self.factor_columns)
        
        # Prepare feature matrix and target vector
        X = clean_df[self.factor_columns].values
        y = clean_df[self.target_column].values
        
        # Fit linear model
        self._fit_linear_model(X, y)
        
        # Fit polynomial model if requested
        if self.use_polynomial:
            self._fit_polynomial_model(X, y)
        
        # Perform clustering analysis if requested
        if self.use_clustering:
            self._fit_cluster_analysis(clean_df)
        
        # Calculate final schematic R²
        self._schematic_r2 = max(self._linear_r2, self._polynomial_r2 or 0.0)
        
        # Calculate integration bias metrics
        self._calculate_integration_bias_metrics()
    
    def _detect_factor_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect factor columns from DataFrame."""
        potential_factors = []
        
        # Common factor names in LLM evaluation
        factor_keywords = ['correctness', 'completeness', 'safety', 'conciseness', 'style', 
                          'accuracy', 'relevance', 'clarity', 'helpfulness', 'coherence']
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if column contains factor keywords or ends with _score
            if any(keyword in col_lower for keyword in factor_keywords) or col_lower.endswith('_score'):
                if col != self.target_column and col_lower not in ['overall_score', 'total_score']:
                    potential_factors.append(col)
        
        # Fallback: look for numeric columns that might be factor scores
        if len(potential_factors) < 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            potential_factors = [col for col in numeric_cols 
                               if col != self.target_column and 'score' in col.lower()]
        
        if len(potential_factors) < 1:
            raise ValueError("Could not detect factor columns. Please specify factor_columns explicitly.")
        
        return potential_factors
    
    def _fit_linear_model(self, X: np.ndarray, y: np.ndarray):
        """Fit linear regression model: overall = β₀ + Σβᵢ·factorᵢ + ε"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Fit linear regression
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            
            # Calculate R²
            y_pred = linear_model.predict(X)
            self._linear_r2 = r2_score(y, y_pred)
            
            # Store weights (coefficients)
            self._linear_weights = {
                'intercept': linear_model.intercept_,
                'coefficients': dict(zip(self.factor_columns, linear_model.coef_))
            }
            
            # Store model for diagnostics
            self._regression_models['linear'] = linear_model
            
        except ImportError:
            warnings.warn("sklearn not available. Using simple linear regression approximation.")
            self._fit_linear_model_basic(X, y)
    
    def _fit_linear_model_basic(self, X: np.ndarray, y: np.ndarray):
        """Basic linear regression using numpy (fallback when sklearn unavailable)."""
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Solve normal equations: (X'X)⁻¹X'y
        try:
            coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            # Predictions
            y_pred = X_with_intercept @ coefficients
            
            # R² calculation
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self._linear_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            # Store weights
            self._linear_weights = {
                'intercept': coefficients[0],
                'coefficients': dict(zip(self.factor_columns, coefficients[1:]))
            }
            
        except np.linalg.LinAlgError:
            warnings.warn("Linear regression failed due to singular matrix. Using correlation-based R².")
            # Fallback to correlation-based R²
            correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
            self._linear_r2 = np.mean([r**2 for r in correlations if not np.isnan(r)])
            
            # Dummy weights
            self._linear_weights = {
                'intercept': np.mean(y),
                'coefficients': dict(zip(self.factor_columns, [1.0/len(self.factor_columns)] * len(self.factor_columns)))
            }
    
    def _fit_polynomial_model(self, X: np.ndarray, y: np.ndarray):
        """Fit polynomial regression with quadratic terms and interactions."""
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.metrics import r2_score
            
            # Create polynomial features
            include_interactions = self.use_interactions and self._n_factors > 1
            poly_features = PolynomialFeatures(
                degree=2, 
                include_bias=True,
                interaction_only=not include_interactions  # If False, includes both interactions and squares
            )
            X_poly = poly_features.fit_transform(X)
            
            # Fit ridge regression (with regularization to handle multicollinearity)
            ridge_model = Ridge(alpha=self.regularization_alpha)
            ridge_model.fit(X_poly, y)
            
            # Calculate R²
            y_pred = ridge_model.predict(X_poly)
            self._polynomial_r2 = r2_score(y, y_pred)
            
            # Store weights
            feature_names = poly_features.get_feature_names_out(self.factor_columns)
            self._polynomial_weights = {
                'feature_names': list(feature_names),
                'coefficients': ridge_model.coef_,
                'intercept': ridge_model.intercept_
            }
            
            # Store models for diagnostics
            self._regression_models['polynomial'] = ridge_model
            self._regression_models['poly_features'] = poly_features
            
        except ImportError:
            warnings.warn("sklearn not available for polynomial regression. Skipping polynomial analysis.")
            self._polynomial_r2 = None
            self._polynomial_weights = None
        except Exception as e:
            warnings.warn(f"Polynomial regression failed: {e}. Using linear R² only.")
            self._polynomial_r2 = self._linear_r2  # Fallback to linear
    
    def _fit_cluster_analysis(self, df: pd.DataFrame):
        """Perform context-dependent clustering analysis."""
        if self.cluster_column and self.cluster_column in df.columns:
            # Use provided clustering column
            self._fit_explicit_cluster_analysis(df)
        else:
            # Perform automatic clustering
            self._fit_automatic_cluster_analysis(df)
    
    def _fit_explicit_cluster_analysis(self, df: pd.DataFrame):
        """Analyze schematic patterns using explicit clustering column."""
        cluster_weights = {}
        unique_clusters = df[self.cluster_column].unique()
        
        for cluster in unique_clusters:
            cluster_df = df[df[self.cluster_column] == cluster]
            
            if len(cluster_df) < self.min_samples_per_cluster:
                continue
            
            X_cluster = cluster_df[self.factor_columns].values
            y_cluster = cluster_df[self.target_column].values
            
            # Fit linear model for this cluster
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_cluster, y_cluster)
                
                cluster_weights[cluster] = {
                    'intercept': model.intercept_,
                    'coefficients': dict(zip(self.factor_columns, model.coef_)),
                    'n_samples': len(cluster_df)
                }
            except:
                # Fallback to basic regression
                pass
        
        self._cluster_weights = cluster_weights
    
    def _fit_automatic_cluster_analysis(self, df: pd.DataFrame):
        """Perform automatic clustering based on factor score patterns."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.linear_model import LinearRegression
            
            X = df[self.factor_columns].values
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            # Analyze each cluster
            cluster_weights = {}
            for cluster_id in range(self.n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_df = df[cluster_mask]
                
                if len(cluster_df) < self.min_samples_per_cluster:
                    continue
                
                X_cluster = cluster_df[self.factor_columns].values
                y_cluster = cluster_df[self.target_column].values
                
                # Fit linear model for this cluster
                model = LinearRegression()
                model.fit(X_cluster, y_cluster)
                
                cluster_weights[f'cluster_{cluster_id}'] = {
                    'intercept': model.intercept_,
                    'coefficients': dict(zip(self.factor_columns, model.coef_)),
                    'n_samples': len(cluster_df)
                }
            
            self._cluster_weights = cluster_weights
            
        except ImportError:
            warnings.warn("sklearn not available for clustering analysis. Skipping cluster analysis.")
            self._cluster_weights = None
        except Exception as e:
            warnings.warn(f"Clustering analysis failed: {e}")
            self._cluster_weights = None
    
    def _calculate_integration_bias_metrics(self):
        """Calculate integration bias metrics (weight disparity, entropy, context stability)."""
        if self._linear_weights is None:
            self._integration_bias_metrics = {}
            return
        
        metrics = {}
        
        # Weight Disparity: σ(|β|) / μ(|β|)
        coeffs = list(self._linear_weights['coefficients'].values())
        abs_coeffs = [abs(c) for c in coeffs]
        
        if abs_coeffs and np.mean(abs_coeffs) > 0:
            metrics['weight_disparity'] = np.std(abs_coeffs) / np.mean(abs_coeffs)
        else:
            metrics['weight_disparity'] = 0.0
        
        # Weight Entropy: -Σ pᵢ log(pᵢ)
        if abs_coeffs and sum(abs_coeffs) > 0:
            probs = [c / sum(abs_coeffs) for c in abs_coeffs]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            metrics['weight_entropy'] = entropy
            metrics['weight_entropy_normalized'] = entropy / np.log(len(abs_coeffs)) if len(abs_coeffs) > 1 else 1.0
        else:
            metrics['weight_entropy'] = 0.0
            metrics['weight_entropy_normalized'] = 0.0
        
        # Context Stability (if clustering was performed)
        if self._cluster_weights and len(self._cluster_weights) > 1:
            stability_scores = []
            cluster_ids = list(self._cluster_weights.keys())
            
            for i, cluster_i in enumerate(cluster_ids):
                for j, cluster_j in enumerate(cluster_ids[i+1:], i+1):
                    coeffs_i = list(self._cluster_weights[cluster_i]['coefficients'].values())
                    coeffs_j = list(self._cluster_weights[cluster_j]['coefficients'].values())
                    
                    # Cosine similarity between weight vectors
                    norm_i = np.linalg.norm(coeffs_i)
                    norm_j = np.linalg.norm(coeffs_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(coeffs_i, coeffs_j) / (norm_i * norm_j)
                        stability_scores.append(similarity)
            
            if stability_scores:
                metrics['context_stability'] = np.mean(stability_scores)
            else:
                metrics['context_stability'] = 1.0  # Perfect stability if only one cluster
        else:
            metrics['context_stability'] = 1.0  # No clustering performed
        
        self._integration_bias_metrics = metrics
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Estimate bias sensitivity from schematic adherence analysis.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. If None, assumes normalized [0,1] range.
            
        Returns:
        --------
        float : Bias sensitivity estimate
        """
        self._validate_fitted()
        
        if self._schematic_r2 is None:
            raise ValueError("Schematic R² not calculated. Check fitting process.")
        
        # Schematic adherence sensitivity: sqrt(1 - R²_schematic)
        normalized_sensitivity = np.sqrt(1.0 - self._schematic_r2)
        
        # Scale to judgment units
        if score_range is None:
            score_range = 4.0  # Arena-Hard default: 5-point scale has range 4.0
        
        bias_sensitivity = normalized_sensitivity * score_range
        self._bias_sensitivity = bias_sensitivity
        
        return bias_sensitivity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information about schematic adherence analysis.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            diagnostics.update({
                "method_details": "Schematic Adherence (Linear + Polynomial Regression)",
                "linear_r2": self._linear_r2,
                "polynomial_r2": self._polynomial_r2,
                "schematic_r2": self._schematic_r2,
                "unexplained_variance": 1.0 - self._schematic_r2,
                "factor_columns": self.factor_columns,
                "target_column": self.target_column,
                "n_factors": self._n_factors,
                "linear_weights": self._linear_weights,
                "integration_bias_metrics": self._integration_bias_metrics,
                "model_configuration": {
                    "use_polynomial": self.use_polynomial,
                    "use_interactions": self.use_interactions,
                    "use_clustering": self.use_clustering,
                    "regularization_alpha": self.regularization_alpha
                },
                "quality_indicators": self._get_quality_indicators()
            })
            
            if self._polynomial_weights:
                diagnostics["polynomial_weights"] = self._polynomial_weights
            
            if self._cluster_weights:
                diagnostics["cluster_analysis"] = {
                    "n_clusters": len(self._cluster_weights),
                    "cluster_weights": self._cluster_weights
                }
        
        return diagnostics
    
    def _get_quality_indicators(self) -> Dict[str, Any]:
        """Get quality indicators for the schematic adherence analysis."""
        if not self._fitted:
            return {}
        
        indicators = {
            "overall_quality": "good" if self._schematic_r2 > 0.7 else "moderate" if self._schematic_r2 > 0.5 else "poor",
            "linear_fit_quality": "good" if self._linear_r2 > 0.6 else "moderate" if self._linear_r2 > 0.3 else "poor",
            "model_improvement": "significant" if (self._polynomial_r2 or 0) - self._linear_r2 > 0.1 else "minimal"
        }
        
        # Integration bias assessment
        if self._integration_bias_metrics:
            wd = self._integration_bias_metrics.get('weight_disparity', 0)
            indicators["weight_balance"] = "good" if wd < 0.5 else "moderate" if wd < 1.0 else "poor"
            
            entropy_norm = self._integration_bias_metrics.get('weight_entropy_normalized', 0)
            indicators["factor_utilization"] = "balanced" if entropy_norm > 0.8 else "unbalanced"
        
        return indicators
    
    def get_linear_model(self):
        """Get the fitted linear regression model."""
        return self._regression_models.get('linear')
    
    def get_polynomial_model(self):
        """Get the fitted polynomial regression model."""
        return self._regression_models.get('polynomial')
    
    def get_linear_r2(self) -> Optional[float]:
        """Get R² from linear regression."""
        return self._linear_r2
    
    def get_polynomial_r2(self) -> Optional[float]:
        """Get R² from polynomial regression.""" 
        return self._polynomial_r2
    
    def get_schematic_r2(self) -> Optional[float]:
        """Get final schematic R² score."""
        return self._schematic_r2
    
    def get_linear_weights(self) -> Optional[Dict[str, Any]]:
        """Get weights from linear regression."""
        return self._linear_weights
    
    def get_integration_bias_metrics(self) -> Optional[Dict[str, float]]:
        """Get integration bias metrics."""
        return self._integration_bias_metrics
    
    def get_cluster_weights(self) -> Optional[Dict[str, Any]]:
        """Get cluster-specific weights if clustering was performed."""
        return self._cluster_weights