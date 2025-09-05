"""
A-BB (Average Bias-Bounded) sensitivity estimation using neighbor sampling
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Any, Optional, List, Callable

from .base import SensitivityEstimator
from ..neighbors import BaseNeighborGenerator, HammingNeighborGenerator


class ABBSensitivity(SensitivityEstimator):
    """
    A-BB sensitivity estimation using root-mean-squared sensitivity calculation.
    
    This estimator implements the A-BB Gaussian mechanism by:
    1. Sampling neighbors D'_1, ..., D'_m using a neighbor generator T
    2. Computing judge outputs f(D) and f(D'_i) for all neighbors
    3. Calculating RMS sensitivity: Δ*₂(f,D) = (1/m * Σᵢ ||f(D) - f(D'ᵢ)||₂²)^(1/2)
    """
    
    def __init__(self, 
                 judge_function: Optional[Callable] = None,
                 neighbor_generator: Optional[Union[str, BaseNeighborGenerator]] = "hamming",
                 num_neighbors: int = 10,
                 target_samples: int = 10,
                 **kwargs):
        """
        Initialize A-BB sensitivity estimator.
        
        Parameters:
        -----------
        judge_function : Callable, optional
            Function that takes context and returns judgment vector.
            If None, will attempt to extract from fit() data.
        neighbor_generator : str or BaseNeighborGenerator, optional
            Neighbor generator to use. Options: "hamming", "formatting", "order", or custom generator.
        num_neighbors : int
            Number of neighbors to sample for RMS calculation (m in algorithm)
        target_samples : int
            Target number of samples for dynamic scoring (default: 50)
        **kwargs : dict
            Additional parameters passed to parent class and neighbor generator
        """
        super().__init__(**kwargs)
        self.judge_function = judge_function
        self.num_neighbors = num_neighbors
        self.target_samples = target_samples
        
        # Set up random number generator (needed for neighbor sampling)
        self.rng = np.random.RandomState(kwargs.get('random_seed'))
        
        # Initialize neighbor generator
        self.neighbor_generator = self._create_neighbor_generator(neighbor_generator, **kwargs)
        
        # State for RMS calculation
        self._original_judgments = None
        self._neighbor_judgments = []
        self._rms_sensitivity = None
        self._original_context = None
        self._sampled_neighbors = []
        
    def _create_neighbor_generator(self, 
                                 generator: Union[str, BaseNeighborGenerator],
                                 **kwargs) -> BaseNeighborGenerator:
        """Create neighbor generator from string or instance."""
        if isinstance(generator, BaseNeighborGenerator):
            return generator
        
        # Import here to avoid circular imports
        from ..neighbors import FormattingNeighborGenerator, OrderNeighborGenerator
        
        generator_map = {
            "hamming": HammingNeighborGenerator,
            "formatting": FormattingNeighborGenerator,
            "order": OrderNeighborGenerator,
        }
        
        if generator not in generator_map:
            available = list(generator_map.keys())
            raise ValueError(f"Unknown neighbor generator '{generator}'. "
                           f"Available: {available}")
        
        # Extract generator-specific kwargs
        generator_kwargs = {k: v for k, v in kwargs.items() 
                          if not k.startswith('_') and k != 'judge_function'}
        
        return generator_map[generator](**generator_kwargs)
    
    def fit(self, judgments: Union[np.ndarray, pd.DataFrame, Dict], **kwargs) -> 'ABBSensitivity':
        """
        Fit A-BB sensitivity by sampling neighbors and computing RMS sensitivity.
        
        Parameters:
        -----------
        judgments : array-like, DataFrame, or Dict
            Judgment context and data. Can be:
            - Dict: judgment context for sampling neighbors
            - DataFrame: judgment context for sampling neighbors  
            - numpy array: pre-computed judgment vectors (limited functionality)
        **kwargs : dict
            Additional fitting parameters:
            - judge_function: Override default judge function
            - context: Explicit context for neighbor sampling
            
        Returns:
        --------
        ABBSensitivity : Self for method chaining
        """
        # Handle judge function override
        if 'judge_function' in kwargs:
            self.judge_function = kwargs['judge_function']
        
        # Handle explicit context
        if 'context' in kwargs:
            context = kwargs['context']
            if isinstance(judgments, np.ndarray):
                self._original_judgments = judgments
            else:
                context = judgments
        else:
            context = judgments
        
        # Store original context
        self._original_context = context
        
        if isinstance(judgments, np.ndarray):
            # Limited functionality: use variance-based estimate
            self._fit_from_array(judgments)
        elif isinstance(judgments, (pd.DataFrame, dict)):
            # Full A-BB functionality: sample neighbors
            self._fit_from_context(context, **kwargs)
        else:
            raise ValueError(f"Unsupported judgment type: {type(judgments)}")
        
        self._fitted = True
        return self
    
    def _fit_from_array(self, judgments: np.ndarray):
        """
        Fit from pre-computed judgment array (limited A-BB functionality).
        
        When only judgment vectors are provided without context, we cannot
        sample true neighbors. Instead, we use bootstrap sampling as approximation.
        """
        judgments = self._validate_input(judgments)
        self._original_judgments = judgments.copy()
        
        if len(judgments) < 2:
            self._rms_sensitivity = 0.0
            return
        
        # Use bootstrap sampling as neighbor approximation
        neighbor_judgments = []
        for _ in range(self.num_neighbors):
            # Sample with replacement
            indices = self.rng.choice(len(judgments), size=len(judgments), replace=True)
            neighbor_sample = judgments[indices]
            neighbor_judgments.append(neighbor_sample)
        
        self._neighbor_judgments = neighbor_judgments
        self._calculate_rms_sensitivity()
    
    def _fit_from_context(self, context: Union[Dict, pd.DataFrame], **kwargs):
        """
        Fit from judgment context using full A-BB neighbor sampling.
        
        This is the primary A-BB method that samples neighboring contexts
        and computes judgments for RMS sensitivity calculation.
        """
        if self.judge_function is None:
            # Try to extract judge function from context or provide default
            if isinstance(context, pd.DataFrame) and 'score' in context.columns:
                # Use the scores as judgment outputs
                self._original_judgments = context['score'].values
                self._fit_from_dataframe_scores(context)
            else:
                raise ValueError("judge_function required for context-based fitting")
        else:
            # Use provided judge function
            self._fit_with_judge_function(context)
    
    def _fit_from_dataframe_scores(self, df: pd.DataFrame):
        """
        Fit from DataFrame by treating rows as different contexts and scores as judgments.
        """
        if 'score' not in df.columns:
            raise ValueError("DataFrame must contain 'score' column")
        
        self._original_judgments = df['score'].values
        
        # Sample neighbor contexts
        neighbors = self.neighbor_generator.sample_neighbors(df, self.num_neighbors)
        self._sampled_neighbors = neighbors
        
        # Extract neighbor judgments (assuming score column exists after perturbation)
        neighbor_judgments = []
        for neighbor in neighbors:
            if 'score' in neighbor.columns:
                neighbor_judgments.append(neighbor['score'].values)
            else:
                # If score column missing, use original scores (no effect)
                neighbor_judgments.append(self._original_judgments.copy())
        
        self._neighbor_judgments = neighbor_judgments
        self._calculate_rms_sensitivity()
    
    def _fit_with_judge_function(self, context: Union[Dict, pd.DataFrame]):
        """
        Fit using provided judge function to evaluate original context and neighbors.
        """
        # Compute original judgment
        # For formatting sensitivity with sampling, use the judge's last used context if available
        self._original_judgments = np.asarray(self.judge_function(context))
        if hasattr(self.judge_function, '_last_context_used_df'):
            try:
                self._sampled_original_context = getattr(self.judge_function, '_last_context_used_df')
                print(f"🔧 Using judge-selected baseline context: {len(self._sampled_original_context)} rows")
            except Exception:
                pass
        # Baseline scoring summary
        try:
            bs = getattr(self.judge_function, 'interface', None)
            batch_size = getattr(bs, 'batch_size', None) if bs else None
            if batch_size:
                print(f"📊 Baseline scoring complete: {len(self._original_judgments)} samples (batch_size={batch_size})")
            else:
                print(f"📊 Baseline scoring complete: {len(self._original_judgments)} samples")
        except Exception:
            pass
        
        # Sample neighbor contexts
        # Check if this is a formatting generator and we have Arena-Hard context
        from ..neighbors import FormattingNeighborGenerator
        
        fast_path = False
        if (isinstance(self.neighbor_generator, FormattingNeighborGenerator) and 
            isinstance(context, pd.DataFrame) and 
            'question_id' in context.columns and 'model' in context.columns):
            
            # For formatting sensitivity: Use efficient sampling approach
            # Sample subset, reformat, and compare original vs reformatted
            if not hasattr(self, '_formatting_logged'):
                print(f"🔧 Using efficient Arena-Hard formatting sensitivity (sample-based)")
                self._formatting_logged = True
            
            # Use the judge-selected baseline context to ensure index alignment
            if not hasattr(self, '_sampled_original_context') or self._sampled_original_context is None:
                raise ValueError("Baseline context not available; cannot generate aligned neighbors")
            baseline_df = self._sampled_original_context
            sample_size = len(baseline_df)
            print(f"📊 Using all {sample_size} samples for dynamic scoring")

            # Determine base_processed directory from baseline context
            if 'source_dir' in baseline_df.columns:
                dirs = baseline_df['source_dir'].unique().tolist()
                if len(dirs) > 1:
                    print(f"⚠️ Multiple source_dir values found; using the first one: {dirs[0]}")
                base_processed_dir = dirs[0]
            else:
                raise ValueError("baseline context must include 'source_dir' column pointing to base_processed directory")

            # Create all neighbors at once with different random perturbations
            neighbors = self.neighbor_generator.create_efficient_arena_hard_neighbors(
                baseline_df, base_processed_dir, self.num_neighbors
            )
            fast_path = True
        else:
            # For other generators: Use standard neighbor sampling
            neighbors = self.neighbor_generator.sample_neighbors(context, self.num_neighbors)
            
        self._sampled_neighbors = neighbors
        
        # Compute neighbor judgments with progress tracking
        from tqdm import tqdm
        neighbor_judgments = []
        
        # Get neighbor type name for better progress description
        neighbor_type = self.neighbor_generator.__class__.__name__.replace('NeighborGenerator', '').lower()
        
        if fast_path:
            print(f"🧪 Neighbor experiments: {len(neighbors)} (single-sample fast path)")
        else:
            print(f"🧪 Neighbor experiments: {len(neighbors)}")

        for exp_idx, neighbor in enumerate(tqdm(neighbors, desc=f"Computing {neighbor_type} neighbors")):
            # Optimization: if this neighbor is a single-sample formatting perturbation experiment,
            # avoid re-scoring unchanged samples. Only score the perturbed sample and splice it into
            # the cached original judgments.
            if (
                isinstance(neighbor, list)
                and any(isinstance(ctx, dict) and ctx.get('is_formatting_neighbor', False) for ctx in neighbor)
                and hasattr(self, '_sampled_original_context')
                and self._original_judgments is not None
            ):
                base = np.asarray(self._original_judgments).flatten().copy()
                perturbed_ctx = next(ctx for ctx in neighbor if ctx.get('is_formatting_neighbor', False))
                idx = int(perturbed_ctx.get('perturbed_sample_index', -1))
                print(f"   ↻ Experiment {exp_idx + 1}/{len(neighbors)}: recomputing only index {idx}")
                if 0 <= idx < len(base):
                    new_score_arr = np.asarray(self.judge_function(perturbed_ctx)).flatten()
                    if new_score_arr.size > 0 and np.isfinite(new_score_arr[0]):
                        base[idx] = float(new_score_arr[0])
                        neighbor_judgments.append(base)
                    else:
                        print("     ⚠️ Skipping neighbor: unparseable or missing score for perturbed sample")
                        continue
                else:
                    print("     ⚠️ Skipping neighbor: invalid perturbed index")
                    continue
            else:
                neighbor_judgment = np.asarray(self.judge_function(neighbor)).flatten()
                if neighbor_judgment.size > 0 and np.all(np.isfinite(neighbor_judgment)):
                    neighbor_judgments.append(neighbor_judgment)
                else:
                    print("     ⚠️ Skipping neighbor: empty or non-finite judgment vector")
        
        self._neighbor_judgments = neighbor_judgments
        self._calculate_rms_sensitivity()
    
    def _calculate_rms_sensitivity(self):
        """
        Calculate Hamming-1 RMS sensitivity by computing sensitivity for single-element changes.
        
        For true Hamming-1 neighbors, we need to:
        1. For each position i in the dataset, compute sensitivity when only that element changes
        2. Average the RMS sensitivities across all positions
        
        Formula: Δ*₂(f,D) = E_i[√(||f(D) - f(D'ᵢ)||₂²)] where D'ᵢ differs from D only at position i
        """
        if self._original_judgments is None or not self._neighbor_judgments:
            self._rms_sensitivity = 0.0
            return
        
        # Ensure original judgments are 1D
        original = np.asarray(self._original_judgments).flatten()
        dataset_size = len(original)
        
        if len(self._neighbor_judgments) != self.num_neighbors:
            print(f"⚠️ Expected {self.num_neighbors} neighbors, got {len(self._neighbor_judgments)}")
        
        # Strategy: Each neighbor represents a true single-sample perturbation experiment
        # Compute RMS sensitivity from mean absolute changes
        
        neighbor_differences = self.get_neighbor_differences()
        
        # Compute RMS of mean absolute changes
        if neighbor_differences:
            rms_sensitivity = np.sqrt(np.mean([diff**2 for diff in neighbor_differences]))
            self._rms_sensitivity = rms_sensitivity
            
            # Check for problematic values (NaN, inf)
            if np.isnan(self._rms_sensitivity) or np.isinf(self._rms_sensitivity):
                print(f"⚠️ {self.__class__.__name__}: Invalid sensitivity value {self._rms_sensitivity}")
                print(f"   Mean absolute changes: {neighbor_differences[:5]}")
                # Set to a reasonable fallback
                self._rms_sensitivity = 0.0
            
            # Debug output
            print(f"🔍 {self.__class__.__name__} single-sample sensitivity debug:")
            print(f"   Dataset size: {dataset_size}")
            print(f"   Original scores range: [{original.min():.1f}, {original.max():.1f}]")
            print(f"   Mean absolute changes: {[f'{d:.3f}' for d in neighbor_differences[:5]]}")
            print(f"   RMS sensitivity: {self._rms_sensitivity:.4f}")
        else:
            self._rms_sensitivity = 0.0
    
    def estimate(self, score_range: Optional[float] = None) -> float:
        """
        Return the RMS sensitivity estimate.
        
        Parameters:
        -----------
        score_range : float, optional
            Range of judgment scores. Used for validation/scaling if provided.
            
        Returns:
        --------
        float : RMS sensitivity estimate Δ*₂(f,D)
        """
        self._validate_fitted()
        
        if self._rms_sensitivity is None:
            raise ValueError("RMS sensitivity not calculated. Check fit() call.")
        
        sensitivity = float(self._rms_sensitivity)
        
        # Check for problematic final values
        if np.isnan(sensitivity) or np.isinf(sensitivity):
            print(f"⚠️ {self.__class__.__name__}: Invalid final sensitivity {sensitivity}, using 0.0")
            sensitivity = 0.0
        
        # Validate against score range if provided
        if score_range is not None and sensitivity > score_range:
            # Cap at score range (sensitivity can't exceed total range)
            print(f"⚠️  {self.__class__.__name__}: Capping sensitivity {sensitivity:.4f} → {score_range:.4f} (score_range={score_range})")
            sensitivity = score_range
        
        self._bias_sensitivity = sensitivity
        return sensitivity
    
    def get_neighbor_differences(self) -> List[float]:
        """
        Get individual mean absolute changes for each single-sample neighbor experiment.
        
        For true Hamming-1 neighbors where exactly one sample is perturbed, this computes
        the absolute difference at the perturbed position only.
        
        Returns:
        --------
        List[float] : Mean absolute changes |f(D)ᵢ - f(D'ᵢ)| for each neighbor experiment
        """
        if self._original_judgments is None or not self._neighbor_judgments:
            return []
        
        original = np.asarray(self._original_judgments).flatten()
        differences = []
        
        for neighbor_judgment in self._neighbor_judgments:
            neighbor = np.asarray(neighbor_judgment).flatten()
            
            # Handle dimension mismatches
            if len(neighbor) != len(original):
                min_len = min(len(original), len(neighbor))
                original_subset = original[:min_len]
                neighbor_subset = neighbor[:min_len]
            else:
                original_subset = original
                neighbor_subset = neighbor
            
            # For single-sample neighbors, compute mean absolute change
            # This should be the absolute difference at the perturbed position
            abs_diff = np.abs(original_subset - neighbor_subset)
            changed = abs_diff > 0
            if np.any(changed):
                mean_abs_change = float(np.mean(abs_diff[changed]))
            else:
                # No changed positions; fall back to overall mean without emitting warnings
                mean_abs_change = float(np.mean(abs_diff)) if abs_diff.size else 0.0
            
            differences.append(float(mean_abs_change))
        
        return differences
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get diagnostic information about A-BB sensitivity estimation.
        
        Returns:
        --------
        Dict[str, Any] : Diagnostic information
        """
        diagnostics = super().get_diagnostics()
        
        if self._fitted:
            neighbor_diffs = self.get_neighbor_differences()
            
            diagnostics.update({
                "rms_sensitivity": self._rms_sensitivity,
                "num_neighbors": self.num_neighbors,
                "actual_neighbors_sampled": len(self._neighbor_judgments),
                "neighbor_generator": self.neighbor_generator.get_diagnostics(),
                "neighbor_differences": neighbor_diffs,
                "mean_neighbor_difference": float(np.mean(neighbor_diffs)) if neighbor_diffs else 0,
                "std_neighbor_difference": float(np.std(neighbor_diffs)) if neighbor_diffs else 0,
                "max_neighbor_difference": float(np.max(neighbor_diffs)) if neighbor_diffs else 0,
                "original_judgment_shape": np.asarray(self._original_judgments).shape if self._original_judgments is not None else None,
                "has_judge_function": self.judge_function is not None
            })
        
        return diagnostics
    
    def validate_abb_constraint(self, tau: float, delta: float) -> Dict[str, Any]:
        """
        Validate the A-BB constraint: τ > Δ*₂(f,D) * sqrt(2/δ).
        
        Parameters:
        -----------
        tau : float
            Bias protection parameter
        delta : float
            Failure probability
            
        Returns:
        --------
        Dict[str, Any] : Validation results
        """
        if not self._fitted:
            raise ValueError("Must fit estimator before validating constraint")
        
        rms_sensitivity = self._rms_sensitivity or 0.0
        constraint_threshold = rms_sensitivity * np.sqrt(2.0 / delta)
        constraint_satisfied = tau > constraint_threshold
        
        return {
            "constraint_satisfied": constraint_satisfied,
            "tau": tau,
            "delta": delta,
            "rms_sensitivity": rms_sensitivity,
            "constraint_threshold": constraint_threshold,
            "margin": tau - constraint_threshold,
            "constraint_formula": "τ > Δ*₂(f,D) * sqrt(2/δ)"
        }
    
    def get_sampled_neighbors(self) -> List[Union[Dict, pd.DataFrame]]:
        """
        Get the sampled neighbor contexts.
        
        Returns:
        --------
        List[Union[Dict, DataFrame]] : Sampled neighbor contexts
        """
        return self._sampled_neighbors.copy()
    
    def get_neighbor_judgments(self) -> List[np.ndarray]:
        """
        Get the computed neighbor judgments.
        
        Returns:
        --------
        List[np.ndarray] : Computed neighbor judgments
        """
        return [np.asarray(judgment) for judgment in self._neighbor_judgments]
