"""
A-BB neighbor generators for sampling neighboring judgment contexts
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
import random
import copy
from tqdm import tqdm


class BaseNeighborGenerator(ABC):
    """
    Abstract base class for A-BB neighbor generators.
    
    Neighbor generators sample neighboring contexts D' from an original context D,
    where neighbors represent semantically equivalent but potentially bias-inducing variations.
    """
    
    def __init__(self, random_seed: Optional[int] = None, **kwargs):
        """
        Initialize neighbor generator.
        
        Parameters:
        -----------
        random_seed : int, optional
            Random seed for reproducible neighbor generation
        **kwargs : dict
            Generator-specific parameters
        """
        self.random_seed = random_seed
        self.params = kwargs
        
        # Set up random state
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
            random.seed(random_seed)
        else:
            self.rng = np.random.RandomState()
    
    @abstractmethod
    def sample_neighbors(self, context: Union[Dict, pd.DataFrame], num_neighbors: int) -> List[Union[Dict, pd.DataFrame]]:
        """
        Sample neighboring contexts from the original context.
        
        Parameters:
        -----------
        context : Dict or DataFrame
            Original judgment context D
        num_neighbors : int
            Number of neighbors to sample (m in A-BB algorithm)
            
        Returns:
        --------
        List[Union[Dict, DataFrame]] : List of neighboring contexts D'_1, ..., D'_m
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of this neighbor generator."""
        pass
    
    def validate_context(self, context: Union[Dict, pd.DataFrame]) -> None:
        """
        Validate that the context is appropriate for this generator.
        
        Parameters:
        -----------
        context : Dict or DataFrame
            Context to validate
        """
        if context is None:
            raise ValueError("Context cannot be None")
        
        if isinstance(context, pd.DataFrame) and context.empty:
            raise ValueError("Context DataFrame cannot be empty")
        elif isinstance(context, dict) and not context:
            raise ValueError("Context dictionary cannot be empty")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the generator."""
        return {
            "generator_type": self.__class__.__name__,
            "random_seed": self.random_seed,
            "params": self.params.copy(),
            "description": self.get_description()
        }


class HammingNeighborGenerator(BaseNeighborGenerator):
    """
    Generates Hamming-1 neighbors by modifying single elements of judgment contexts.
    
    For judgment contexts represented as dictionaries or DataFrames, this generator
    creates neighbors by changing one field/column at a time while keeping others constant.
    """
    
    def __init__(self, 
                 modifiable_fields: Optional[List[str]] = None,
                 perturbation_scale: float = 0.1,
                 **kwargs):
        """
        Initialize Hamming neighbor generator.
        
        Parameters:
        -----------
        modifiable_fields : List[str], optional
            List of field names that can be modified. If None, all numeric fields are modifiable.
        perturbation_scale : float
            Scale of numerical perturbations (as fraction of range)
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.modifiable_fields = modifiable_fields
        self.perturbation_scale = perturbation_scale
    
    def sample_neighbors(self, context: Union[Dict, pd.DataFrame], num_neighbors: int) -> List[Union[Dict, pd.DataFrame]]:
        """
        Sample Hamming-1 neighbors by modifying single fields.
        
        Parameters:
        -----------
        context : Dict or DataFrame
            Original judgment context
        num_neighbors : int
            Number of neighbors to sample
            
        Returns:
        --------
        List[Union[Dict, DataFrame]] : List of neighboring contexts
        """
        self.validate_context(context)
        neighbors = []
        
        if isinstance(context, pd.DataFrame):
            neighbors = self._sample_dataframe_neighbors(context, num_neighbors)
        elif isinstance(context, dict):
            neighbors = self._sample_dict_neighbors(context, num_neighbors)
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
        
        return neighbors
    
    def _sample_dataframe_neighbors(self, df: pd.DataFrame, num_neighbors: int) -> List[pd.DataFrame]:
        """Sample Hamming-1 neighbors from DataFrame context.
        
        Hamming-1 neighbors are datasets that differ by exactly 1 sample (row).
        This includes:
        - Row deletion (removing one evaluation sample)
        - Single element modification (changing one score in one row)
        """
        neighbors = []
        
        # Determine modifiable columns (score columns for sensitivity analysis)
        if self.modifiable_fields is not None:
            modifiable_cols = [col for col in self.modifiable_fields if col in df.columns]
        else:
            # Use numeric columns by default
            modifiable_cols = list(df.select_dtypes(include=[np.number]).columns)
        
        if not modifiable_cols:
            raise ValueError("No modifiable fields found in DataFrame")
        
        if len(df) == 0:
            raise ValueError("Cannot generate Hamming-1 neighbors from empty DataFrame")
        
        for _ in range(num_neighbors):
            # For judge evaluation sensitivity: Hamming-1 means exactly one score changed
            # in a dataset of the same size. No row deletion to avoid dimension mismatches.
            neighbor_df = df.copy()
            
            # Choose random row and column to modify (single element modification)
            row_to_modify = self.rng.randint(0, len(df))
            col_to_modify = self.rng.choice(modifiable_cols)
            
            # Apply small perturbation to single element
            if df[col_to_modify].dtype in [np.float64, np.float32, np.int64, np.int32]:
                original_value = neighbor_df.iloc[row_to_modify][col_to_modify]
                
                # Use a small fixed perturbation scale (0.1 of the value, minimum 0.1)
                perturbation_magnitude = max(abs(original_value) * 0.1, 0.1)
                perturbation = self.rng.normal(0, perturbation_magnitude)
                
                neighbor_df.iloc[row_to_modify, neighbor_df.columns.get_loc(col_to_modify)] = original_value + perturbation
            
            neighbors.append(neighbor_df)
        
        return neighbors
    
    def _sample_dict_neighbors(self, context_dict: Dict, num_neighbors: int) -> List[Dict]:
        """Sample neighbors from dictionary context."""
        neighbors = []
        
        # Determine modifiable fields
        if self.modifiable_fields is not None:
            modifiable_fields = [k for k in self.modifiable_fields if k in context_dict]
        else:
            # Use numeric fields by default
            modifiable_fields = [k for k, v in context_dict.items() 
                               if isinstance(v, (int, float, np.number))]
        
        if not modifiable_fields:
            raise ValueError("No modifiable fields found in context dictionary")
        
        for _ in range(num_neighbors):
            neighbor = copy.deepcopy(context_dict)
            
            # Choose random field to modify
            field_to_modify = self.rng.choice(modifiable_fields)
            original_value = context_dict[field_to_modify]
            
            # Apply perturbation
            if isinstance(original_value, (int, float, np.number)):
                # For single values, use relative perturbation
                perturbation = self.rng.normal(0, abs(original_value) * self.perturbation_scale)
                neighbor[field_to_modify] = original_value + perturbation
            
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_description(self) -> str:
        """Get description of Hamming neighbor generator."""
        return f"Hamming-1 neighbors with {self.perturbation_scale:.3f} perturbation scale"


class FormattingNeighborGenerator(BaseNeighborGenerator):
    """
    Generates neighbors by applying semantic-preserving variations that should not affect judgment quality.
    
    This includes changes like:
    - LLM-based rephrasing preserving meaning and functionality
    - Minor synonym substitutions
    - Style variations without content changes
    - (Legacy) Minor formatting changes like whitespace, capitalization, punctuation
    """
    
    def __init__(self, 
                 text_fields: Optional[List[str]] = None,
                 formatting_types: Optional[List[str]] = None,
                 single_field: bool = True,
                 disable_transforms: bool = False,
                 oumi_config_path: Optional[str] = None,
                 oumi_retries: int = 3,
                 oumi_backoff_sec: float = 1.0,
                 rephrase_temperature: Optional[float] = None,
                 # Optional: reuse an existing Oumi engine to avoid double-instantiation (e.g., LLAMACPP 32B)
                 oumi_shared_engine: Optional[Any] = None,
                 **kwargs):
        """
        Initialize formatting neighbor generator.
        
        Parameters:
        -----------
        text_fields : List[str], optional
            Fields containing text to modify. If None, auto-detects text fields.
        formatting_types : List[str], optional
            Types of formatting changes to apply. Options: 'whitespace', 'capitalization', 'punctuation', 'rephrasing'
        single_field : bool
            If True, guarantee only a single field/cell is modified per neighbor (Hamming-1 behavior).
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        # Default to answer fields when present
        self.text_fields = text_fields if text_fields is not None else ['answer_a', 'answer_b']
        # Default to minimal, formatting-only perturbations
        self.formatting_types = formatting_types or ['whitespace', 'capitalization', 'punctuation']
        self.disable_transforms = disable_transforms
        self.single_field = single_field
        # Oumi settings for rephrasing
        self.oumi_config_path = oumi_config_path
        self.oumi_retries = int(oumi_retries)
        self.oumi_backoff_sec = float(oumi_backoff_sec)
        self._oumi_engine = None
        self._oumi_generation_params = None
        self.rephrase_temperature = rephrase_temperature
        # If provided, reuse an existing engine (prevents second heavy model load)
        self.oumi_shared_engine = oumi_shared_engine

    def _get_oumi_engine(self):
        """Lazily build or reuse an Oumi inference engine for rephrasing."""
        if self._oumi_engine is not None:
            return self._oumi_engine
        # Prefer a shared engine if supplied (avoids double instantiation / OOM)
        if self.oumi_shared_engine is not None:
            try:
                # Create default generation params if not already set
                if self._oumi_generation_params is None:
                    from oumi.core.configs import GenerationParams
                    self._oumi_generation_params = GenerationParams(
                        max_new_tokens=1024,
                        temperature=float(self.rephrase_temperature) if self.rephrase_temperature is not None else 0.2,
                        top_p=0.9,
                    )
                self._oumi_engine = self.oumi_shared_engine
                return self._oumi_engine
            except Exception as e:
                raise RuntimeError(f"Failed to use shared Oumi engine for rephrasing: {e}")
        import os
        cfg_path = self.oumi_config_path or os.getenv('OUMI_REPHRASE_CONFIG')
        if not cfg_path:
            raise RuntimeError("Oumi config path is required for rephrasing (set oumi_config_path or OUMI_REPHRASE_CONFIG)")
        try:
            from oumi.builders.inference_engines import build_inference_engine
            from oumi.core.configs import InferenceConfig, ModelParams, GenerationParams, InferenceEngineType, RemoteParams
            import yaml
            from pathlib import Path
            p = Path(cfg_path)
            if not p.exists():
                raise FileNotFoundError(f"Oumi config not found: {p}")
            with open(p, 'r') as f:
                cfg = yaml.safe_load(f)
            engine_type = InferenceEngineType(cfg.get('engine', 'NATIVE'))
            mc = cfg.get('model', {})
            model_params = ModelParams(
                model_name=mc.get('model_name'),
                tokenizer_name=mc.get('tokenizer_name', mc.get('model_name')),
                model_max_length=mc.get('model_max_length', 8192),
                torch_dtype_str=mc.get('torch_dtype_str', 'float16'),
                trust_remote_code=mc.get('trust_remote_code', True),
                model_kwargs=mc.get('model_kwargs', {})
            )
            gc = cfg.get('generation', {})
            self._oumi_generation_params = GenerationParams(
                max_new_tokens=gc.get('max_new_tokens', 1024),
                temperature=gc.get('temperature', 0.2),
                top_p=gc.get('top_p', 0.9)
            )
            # Override temperature for rephrasing if requested
            if self.rephrase_temperature is not None:
                try:
                    self._oumi_generation_params.temperature = float(self.rephrase_temperature)
                except Exception:
                    pass
            rc = cfg.get('remote', {})
            api_key = rc.get('api_key')
            if not api_key:
                if engine_type.name == 'OPENAI':
                    api_key = os.getenv('OPENAI_API_KEY')
                elif engine_type.name == 'ANTHROPIC':
                    api_key = os.getenv('ANTHROPIC_API_KEY')
            remote_params = None
            if engine_type.name in ['OPENAI', 'ANTHROPIC']:
                remote_params = RemoteParams(
                    api_key=api_key,
                    num_workers=int(rc.get('num_workers', 4)),
                    politeness_policy=float(rc.get('politeness_policy', 60.0))
                )
            self._oumi_engine = build_inference_engine(
                engine_type=engine_type,
                model_params=model_params,
                generation_params=self._oumi_generation_params,
                remote_params=remote_params
            )
            return self._oumi_engine
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Oumi rephrase engine: {e}")
    
    def sample_neighbors(self, context: Union[Dict, pd.DataFrame], num_neighbors: int) -> List[Union[Dict, pd.DataFrame]]:
        """
        Sample neighbors with formatting variations.
        
        Parameters:
        -----------
        context : Dict or DataFrame
            Original judgment context
        num_neighbors : int
            Number of neighbors to sample
            
        Returns:
        --------
        List[Union[Dict, DataFrame]] : List of neighboring contexts
        """
        self.validate_context(context)
        neighbors = []
        
        if isinstance(context, pd.DataFrame):
            neighbors = self._sample_dataframe_formatting_neighbors(context, num_neighbors)
        elif isinstance(context, dict):
            neighbors = self._sample_dict_formatting_neighbors(context, num_neighbors)
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
        
        return neighbors
    
    def _sample_dataframe_formatting_neighbors(self, df: pd.DataFrame, num_neighbors: int) -> List[pd.DataFrame]:
        """Sample formatting neighbors from DataFrame."""
        neighbors = []
        
        # Determine text fields
        if self.text_fields is not None:
            text_cols = [col for col in self.text_fields if col in df.columns]
        else:
            # Auto-detect text columns
            text_cols = list(df.select_dtypes(include=['object', 'string']).columns)
        
        if not text_cols:
            # If no text fields, return copies (formatting doesn't apply)
            return [df.copy() for _ in range(num_neighbors)]
        
        for _ in range(num_neighbors):
            neighbor_df = df.copy()
            
            # Choose random text column to modify
            col_to_modify = self.rng.choice(text_cols)
            
            if self.single_field:
                # Modify exactly one cell for Hamming-1 behavior
                if len(df) == 0:
                    neighbors.append(neighbor_df)
                    continue
                row_idx = int(self.rng.randint(0, len(df)))
                original_val = df.iloc[row_idx][col_to_modify]
                if pd.notnull(original_val):
                    neighbor_df.iat[row_idx, neighbor_df.columns.get_loc(col_to_modify)] = \
                        self._apply_formatting_changes(str(original_val))
            else:
                # Apply to entire column
                neighbor_df[col_to_modify] = df[col_to_modify].apply(
                    lambda x: self._apply_formatting_changes(str(x)) if pd.notnull(x) else x
                )
            
            neighbors.append(neighbor_df)
        
        return neighbors
    
    def _sample_dict_formatting_neighbors(self, context_dict: Dict, num_neighbors: int) -> List[Dict]:
        """Sample formatting neighbors from dictionary."""
        neighbors = []
        
        # Determine text fields
        if self.text_fields is not None:
            text_fields = [k for k in self.text_fields if k in context_dict]
        else:
            # Auto-detect text fields
            text_fields = [k for k, v in context_dict.items() if isinstance(v, str)]
        
        if not text_fields:
            # If no text fields, return copies
            return [copy.deepcopy(context_dict) for _ in range(num_neighbors)]
        
        for _ in range(num_neighbors):
            neighbor = copy.deepcopy(context_dict)
            
            # Choose random text field to modify
            field_to_modify = self.rng.choice(text_fields)
            original_text = context_dict[field_to_modify]
            
            if isinstance(original_text, str):
                neighbor[field_to_modify] = self._apply_formatting_changes(original_text)
            
            neighbors.append(neighbor)
        
        return neighbors
    
    def _apply_formatting_changes(self, text: str) -> str:
        """Apply random formatting changes to text."""
        if getattr(self, 'disable_transforms', False):
            return text
        modified_text = text
        
        # Choose random formatting type to apply
        formatting_type = self.rng.choice(self.formatting_types)
        
        if formatting_type == 'whitespace':
            modified_text = self._modify_whitespace(modified_text)
        elif formatting_type == 'capitalization':
            modified_text = self._modify_capitalization(modified_text)
        elif formatting_type == 'punctuation':
            modified_text = self._modify_punctuation(modified_text)
        elif formatting_type == 'rephrasing':
            modified_text = self._rephrase_text(modified_text)
        
        return modified_text

    def generate_neighbors(self, text: str, num_neighbors: int = 3) -> List[str]:
        """Generate simple formatting-only neighbors for a raw text string.

        This utility is used primarily for producing illustrative samples in reports.
        It applies minimal, non-semantic formatting changes (whitespace, capitalization, punctuation)
        to the provided string and returns a list of perturbed strings.
        """
        neighbors: List[str] = []
        for _ in range(max(0, int(num_neighbors))):
            neighbors.append(self._apply_formatting_changes(text))
        return neighbors
    
    def _modify_whitespace(self, text: str) -> str:
        """Apply whitespace modifications."""
        # Add or remove extra spaces
        if self.rng.random() < 0.5:
            # Add extra spaces
            words = text.split()
            if len(words) > 1:
                insert_pos = self.rng.randint(0, len(words)-1)
                words.insert(insert_pos, ' ')
            return ' '.join(words)
        else:
            # Normalize spacing
            return ' '.join(text.split())
    
    def _modify_capitalization(self, text: str) -> str:
        """Apply capitalization modifications to non-content words."""
        words = text.split()
        # Only modify articles, prepositions, etc.
        non_content_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by']
        
        for i, word in enumerate(words):
            if word.lower() in non_content_words and i > 0:  # Don't modify first word
                if self.rng.random() < 0.3:  # 30% chance to modify
                    words[i] = word.upper() if word.islower() else word.lower()
        
        return ' '.join(words)
    
    def _modify_punctuation(self, text: str) -> str:
        """Apply minor punctuation modifications."""
        # Add or remove trailing punctuation
        if text and self.rng.random() < 0.3:
            if text[-1] in '.!?':
                # Remove punctuation
                return text[:-1]
            else:
                # Add punctuation
                punct = self.rng.choice(['.', '!', '?'])
                return text + punct
        
        return text
    
    def _oumi_rephrase(self, text: str) -> str:
        """Rephrase via Oumi with retries and strict length bounds."""
        engine = self._get_oumi_engine()
        from oumi.core.configs import InferenceConfig
        from oumi.core.types.conversation import Conversation, Message, ContentItem, Type, Role
        import time
        prompt = (
            "Make MINIMAL stylistic changes to the following text while preserving the exact same meaning. "
            "Only small synonyms or light restructuring. Do NOT change technical terms, code, numbers, or key concepts. "
            "Keep length and structure nearly the same.\n\nText to rephrase:\n{t}\n\nMinimally rephrased version:"
        ).format(t=text)
        attempt = 0
        last_err = None
        while True:
            try:
                inf_cfg = InferenceConfig(generation=self._oumi_generation_params,
                                          model=getattr(engine, '_model_params', None))
                if hasattr(engine, 'infer_online'):
                    outs = engine.infer_online([Conversation(messages=[Message(role=Role.USER, content=prompt)])], inf_cfg)
                else:
                    conv = Conversation(messages=[Message(role=Role.USER, content=[ContentItem(type=Type.TEXT, content=prompt)])])
                    outs = engine.infer(input=[conv])
                if not outs:
                    raise RuntimeError("Empty Oumi response")
                oc = outs[0]
                out_text = None
                if hasattr(oc, 'messages') and oc.messages:
                    for m in reversed(oc.messages):
                        if getattr(m, 'role', None) == Role.ASSISTANT:
                            if isinstance(m.content, str):
                                out_text = m.content
                            elif isinstance(m.content, list):
                                out_text = m.compute_flattened_text_content()
                            else:
                                out_text = str(m.content)
                            break
                if out_text is None:
                    out_text = str(oc)
                rephrased = out_text.strip()
                # Word-based comparison with simple markup cleanup
                import re as _re
                def _clean(s: str) -> str:
                    return _re.sub(r"<\|[^|]+\|>", " ", s)
                orig_clean = _clean(text)
                reph_clean = _clean(rephrased)
                orig_words = [w for w in orig_clean.split() if w]
                reph_words = [w for w in reph_clean.split() if w]
                ratio = (len(reph_words) / max(len(orig_words), 1))
                if 0.75 <= ratio <= 1.25:
                    return rephrased
                # Include full original and rephrased text for debugging
                raise ValueError(
                    (
                        f"Rephrase length ratio out of bounds: {ratio:.2f}; "
                        f"original_words={len(orig_words)}; rephrased_words={len(reph_words)}; "
                        f"original=<<<{text}>>>; rephrased=<<<{rephrased}>>>"
                    )
                )
            except Exception as e:
                last_err = e
                attempt += 1
                if attempt > max(1, self.oumi_retries):
                    raise RuntimeError(f"Rephrasing failed after {self.oumi_retries} retries: {last_err}")
                time.sleep(self.oumi_backoff_sec * (2 ** (attempt - 1)))

    def _rephrase_text(self, text: str) -> str:
        """Rephrase text using Oumi (strict)."""
        return self._oumi_rephrase(text)
    
    # Fallback modifications removed: strict rephrasing only
    
    def get_description(self) -> str:
        """Get description of formatting neighbor generator."""
        return f"Formatting variations: {', '.join(self.formatting_types)}"
    
    def create_arena_hard_neighbors(self, df: pd.DataFrame, num_neighbors: int, base_processed_dir: str) -> List[Dict]:
        """
        Create formatting neighbors by loading and perturbing actual Arena-Hard content.
        
        This is specifically for formatting sensitivity measurement where we need to
        create variations of Arena-Hard questions and responses.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with question_id and model columns
        num_neighbors : int
            Number of neighbors to generate
        base_processed_dir : str
            Path to Arena-Hard base_processed directory
            
        Returns:
        --------
        List[Dict] : List of contexts with perturbed Arena-Hard content
        """
        from pathlib import Path
        import json
        
        neighbors = []
        
        # Import Arena-Hard utilities from shared module (single source of truth)
        from .interfaces.arena_hard_utils import load_judgment_data as _load_judgment_data
        from .interfaces.arena_hard_utils import parse_arena_hard_prompt as _parse_arena_hard_prompt
        
        for _ in range(num_neighbors):
            neighbor_contexts = []
            
            for _, row in df.iterrows():
                question_id = row['question_id'] 
                model_name = row['model']
                
                try:
                    # Load original Arena-Hard data
                    judgment_data = _load_judgment_data(base_processed_dir, question_id, model_name)
                    user_prompt = judgment_data['games'][0]['user_prompt']
                    question, answer_a, answer_b = _parse_arena_hard_prompt(user_prompt)
                    
                    # Apply formatting perturbations respecting configured fields and single_field
                    # Only allow configured text fields (defaults to assistant answers only)
                    allowed_fields = self.text_fields if self.text_fields is not None else ['answer_a', 'answer_b']
                    fields_to_modify = allowed_fields
                    if self.single_field and len(allowed_fields) > 0:
                        fields_to_modify = [self.rng.choice(allowed_fields)]
                    
                    perturbed_question = question
                    perturbed_answer_a = answer_a
                    perturbed_answer_b = answer_b
                    
                    if 'question' in fields_to_modify:
                        perturbed_question = self._apply_formatting_changes(question)
                    if 'answer_a' in fields_to_modify:
                        perturbed_answer_a = self._apply_formatting_changes(answer_a)
                    if 'answer_b' in fields_to_modify:
                        perturbed_answer_b = self._apply_formatting_changes(answer_b)
                    
                    # Create context with perturbed content
                    perturbed_context = {
                        'question': perturbed_question,
                        'answer_a': perturbed_answer_a,
                        'answer_b': perturbed_answer_b,
                        'model': model_name,
                        'question_id': question_id,
                        'is_formatting_neighbor': True  # Flag to identify this as formatting neighbor
                    }
                    
                    neighbor_contexts.append(perturbed_context)
                    
                except Exception as e:
                    # If we can't load/parse data, create a minimal neighbor context
                    neighbor_contexts.append({
                        'question': 'Formatting perturbation failed',
                        'answer_a': 'Error loading data',
                        'answer_b': 'Error loading data', 
                        'model': model_name,
                        'question_id': question_id,
                        'is_formatting_neighbor': True,
                        'error': str(e)
                    })
            
            neighbors.append(neighbor_contexts)
        
        return neighbors
    
    def create_efficient_arena_hard_neighbors(self, sampled_df: pd.DataFrame, base_processed_dir: str, num_neighbors: int) -> List[List[Dict]]:
        """
        Create single-sample formatting neighbors for true Hamming-1 sensitivity measurement.
        
        This creates num_neighbors single-sample perturbations where exactly one sample
        is formatting-perturbed while others remain unchanged. This enables proper
        Hamming-1 sensitivity calculation using mean absolute changes.
        
        Parameters:
        -----------
        sampled_df : pd.DataFrame
            Pre-sampled subset of the dataset (e.g., 20 rows)
        base_processed_dir : str
            Path to Arena-Hard base_processed directory
        num_neighbors : int
            Number of single-sample perturbation experiments (e.g., 100)
            
        Returns:
        --------
        List[List[Dict]] : List of neighbors, each containing original contexts plus one perturbed sample
        """
        from pathlib import Path
        import json
        
        # Import the Arena-Hard data loading functions from our local utils
        from .interfaces.arena_hard_utils import load_judgment_data as _load_judgment_data
        from .interfaces.arena_hard_utils import parse_arena_hard_prompt as _parse_arena_hard_prompt
        
        print(f"🔧 Creating {num_neighbors} single-sample perturbation experiments for {len(sampled_df)} samples...")
        
        # Load all original Arena-Hard data first (do this once)
        original_data = {}
        successful_samples = []
        original_contexts = []
        
        import os as _os
        debug_rephrase = _os.getenv('A_BB_REPHRASE_DEBUG', '').strip() in ('1', 'true', 'yes')

        for _, row in sampled_df.iterrows():
            question_id = row['question_id'] 
            model_name = row['model']
            
            try:
                # Load original Arena-Hard data for this sample
                judgment_data = _load_judgment_data(base_processed_dir, question_id, model_name)
                user_prompt = judgment_data['games'][0]['user_prompt']
                question, answer_a, answer_b = _parse_arena_hard_prompt(user_prompt)
                
                if debug_rephrase:
                    qw = len(str(question).split())
                    aw = len(str(answer_a).split())
                    bw = len(str(answer_b).split())
                    print(f"   📦 Loaded qid={question_id} model={model_name} words: question={qw}, answer_a={aw}, answer_b={bw}")

                original_context = {
                    'question': question,
                    'answer_a': answer_a, 
                    'answer_b': answer_b,
                    'model': model_name,
                    'question_id': question_id,
                    'is_formatting_neighbor': False
                }
                
                original_data[question_id] = original_context
                successful_samples.append(question_id)
                original_contexts.append(original_context)
                
            except Exception as e:
                # If we can't load/parse data, skip this sample
                print(f"⚠️ Skipping sample {question_id}: {e}")
                continue
        
        print(f"✅ Loaded {len(successful_samples)}/{len(sampled_df)} samples successfully")
        
        if len(successful_samples) == 0:
            print("❌ No samples loaded successfully")
            return []
        
        # Create num_neighbors single-sample perturbation experiments
        all_neighbors = []
        
        import os as _os
        debug_rephrase = _os.getenv('A_BB_REPHRASE_DEBUG', '').strip() in ('1', 'true', 'yes')

        # Progress bar for rephrasing stage (one rephrase per experiment in single_field mode)
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=num_neighbors, desc="Rephrasing (Oumi)", leave=True)
        skipped_due_to_rephrase = 0

        for neighbor_idx in range(num_neighbors):
            # For each experiment, choose exactly ONE sample to perturb
            sample_to_perturb_idx = self.rng.randint(0, len(successful_samples))
            question_id_to_perturb = successful_samples[sample_to_perturb_idx]
            
            # Create neighbor dataset: all original samples + one perturbed sample
            neighbor_contexts = []
            
            skip_experiment = False
            for i, question_id in enumerate(successful_samples):
                original = original_data[question_id]
                
                if i == sample_to_perturb_idx:
                    # This is the sample we perturb for this experiment
                    allowed_fields = self.text_fields if self.text_fields is not None else ['question', 'answer_a', 'answer_b']
                    fields_to_modify = allowed_fields
                    if self.single_field and len(allowed_fields) > 0:
                        chosen = self.rng.choice(allowed_fields)
                        fields_to_modify = [chosen]
                        if debug_rephrase:
                            print(f"   🎯 Experiment {neighbor_idx}: qid={question_id} will perturb field={chosen}")

                    perturbed_question = original['question']
                    perturbed_answer_a = original['answer_a']
                    perturbed_answer_b = original['answer_b']
                    
                    try:
                        if 'question' in fields_to_modify:
                            if debug_rephrase:
                                ow = len(str(original['question']).split())
                                print(f"   🪄 Rephrase target qid={question_id} field=question words={ow} exp={neighbor_idx} idx={i} sample")
                            perturbed_question = self._apply_formatting_changes(original['question'])
                        if 'answer_a' in fields_to_modify:
                            if debug_rephrase:
                                ow = len(str(original['answer_a']).split())
                                snippet = ' '.join(str(original['answer_a']).split()[:12])
                                print(f"   🪄 Rephrase target qid={question_id} field=answer_a words={ow} exp={neighbor_idx} idx={i} snippet=<<<{snippet}>>>")
                            perturbed_answer_a = self._apply_formatting_changes(original['answer_a'])
                        if 'answer_b' in fields_to_modify:
                            if debug_rephrase:
                                ow = len(str(original['answer_b']).split())
                                snippet = ' '.join(str(original['answer_b']).split()[:12])
                                print(f"   🪄 Rephrase target qid={question_id} field=answer_b words={ow} exp={neighbor_idx} idx={i} snippet=<<<{snippet}>>>")
                            perturbed_answer_b = self._apply_formatting_changes(original['answer_b'])
                    except Exception as e:
                        print(f"⚠️ Skipping experiment {neighbor_idx} due to rephrasing error on {fields_to_modify} (qid={question_id}): {e}")
                        if debug_rephrase:
                            # Log originals to confirm wrong-variable issues
                            oq = ' '.join(str(original['question']).split()[:25])
                            oa = ' '.join(str(original['answer_a']).split()[:25])
                            ob = ' '.join(str(original['answer_b']).split()[:25])
                            print(f"   ↪︎ Original snippets qid={question_id}: question=<<<{oq}>>> | answer_a=<<<{oa}>>> | answer_b=<<<{ob}>>>")
                        skip_experiment = True
                        skipped_due_to_rephrase += 1
                        break
                    
                    perturbed_context = {
                        'question': perturbed_question,
                        'answer_a': perturbed_answer_a,
                        'answer_b': perturbed_answer_b,
                        'model': original['model'],
                        'question_id': question_id,
                        'neighbor_index': neighbor_idx,
                        'perturbed_sample_index': i,
                        'is_formatting_neighbor': True,
                        'original_question': original['question'],
                        'original_answer_a': original['answer_a'],
                        'original_answer_b': original['answer_b']
                    }
                    neighbor_contexts.append(perturbed_context)
                else:
                    # This sample remains unchanged
                    unchanged_context = original.copy()
                    unchanged_context['neighbor_index'] = neighbor_idx
                    unchanged_context['perturbed_sample_index'] = sample_to_perturb_idx
                    neighbor_contexts.append(unchanged_context)
            
            if skip_experiment:
                continue
            all_neighbors.append(neighbor_contexts)
            # Count this experiment as one completed rephrase
            try:
                pbar.update(1)
            except Exception:
                pass
        
        # Close progress bar and print summary
        try:
            pbar.close()
        except Exception:
            pass

        print(f"✅ Created {len(all_neighbors)} single-sample perturbation experiments")
        print(f"   Each experiment perturbs exactly 1 out of {len(successful_samples)} samples")
        if skipped_due_to_rephrase:
            print(f"   ⚠️ Skipped {skipped_due_to_rephrase} experiments due to rephrasing errors")
        return all_neighbors


class OrderNeighborGenerator(BaseNeighborGenerator):
    """
    Generates neighbors by reordering elements that should not affect judgment quality.
    
    This includes:
    - Shuffling rows in DataFrames (if order shouldn't matter)
    - Reordering list-type fields
    - Swapping equivalent sections
    """
    
    def __init__(self, 
                 reorderable_fields: Optional[List[str]] = None,
                 preserve_first: bool = True,
                 **kwargs):
        """
        Initialize order neighbor generator.
        
        Parameters:
        -----------
        reorderable_fields : List[str], optional
            Fields that can be reordered. If None, auto-detects list-type fields.
        preserve_first : bool
            Whether to preserve the first element when reordering
        **kwargs : dict
            Additional parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.reorderable_fields = reorderable_fields
        self.preserve_first = preserve_first
    
    def sample_neighbors(self, context: Union[Dict, pd.DataFrame], num_neighbors: int) -> List[Union[Dict, pd.DataFrame]]:
        """
        Sample neighbors with element reordering.
        
        Parameters:
        -----------
        context : Dict or DataFrame
            Original judgment context
        num_neighbors : int
            Number of neighbors to sample
            
        Returns:
        --------
        List[Union[Dict, DataFrame]] : List of neighboring contexts
        """
        self.validate_context(context)
        neighbors = []
        
        if isinstance(context, pd.DataFrame):
            neighbors = self._sample_dataframe_order_neighbors(context, num_neighbors)
        elif isinstance(context, dict):
            neighbors = self._sample_dict_order_neighbors(context, num_neighbors)
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
        
        return neighbors
    
    def _sample_dataframe_order_neighbors(self, df: pd.DataFrame, num_neighbors: int) -> List[pd.DataFrame]:
        """Sample order neighbors from DataFrame."""
        neighbors = []
        
        for _ in range(num_neighbors):
            neighbor_df = df.copy()
            
            # Shuffle rows if there are multiple rows
            if len(df) > 1:
                indices = list(range(len(df)))
                if self.preserve_first and len(indices) > 1:
                    # Shuffle all but the first
                    to_shuffle = indices[1:]
                    self.rng.shuffle(to_shuffle)
                    new_indices = [indices[0]] + to_shuffle
                else:
                    self.rng.shuffle(indices)
                    new_indices = indices
                
                neighbor_df = df.iloc[new_indices].reset_index(drop=True)
            
            neighbors.append(neighbor_df)
        
        return neighbors
    
    def _sample_dict_order_neighbors(self, context_dict: Dict, num_neighbors: int) -> List[Dict]:
        """Sample order neighbors from dictionary."""
        neighbors = []
        
        # Determine reorderable fields
        if self.reorderable_fields is not None:
            reorderable_fields = [k for k in self.reorderable_fields if k in context_dict]
        else:
            # Auto-detect list-type fields
            reorderable_fields = [k for k, v in context_dict.items() 
                                if isinstance(v, (list, tuple)) and len(v) > 1]
        
        if not reorderable_fields:
            # No reorderable fields, return copies
            return [copy.deepcopy(context_dict) for _ in range(num_neighbors)]
        
        for _ in range(num_neighbors):
            neighbor = copy.deepcopy(context_dict)
            
            # Choose random field to reorder
            field_to_reorder = self.rng.choice(reorderable_fields)
            original_list = context_dict[field_to_reorder]
            
            if isinstance(original_list, (list, tuple)):
                new_list = list(original_list)
                if len(new_list) > 1:
                    if self.preserve_first and len(new_list) > 2:
                        # Shuffle all but the first
                        to_shuffle = new_list[1:]
                        self.rng.shuffle(to_shuffle)
                        new_list = [new_list[0]] + to_shuffle
                    else:
                        self.rng.shuffle(new_list)
                
                # Convert back to original type
                if isinstance(original_list, tuple):
                    neighbor[field_to_reorder] = tuple(new_list)
                else:
                    neighbor[field_to_reorder] = new_list
            
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_description(self) -> str:
        """Get description of order neighbor generator."""
        preserve_note = " (preserving first)" if self.preserve_first else ""
        return f"Element reordering{preserve_note}"
