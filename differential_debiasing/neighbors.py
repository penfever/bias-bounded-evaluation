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
    
    def _rephrase_text(self, text: str) -> str:
        """Rephrase text using LLM while preserving functional utility and meaning."""
        try:
            # Import here to avoid circular imports
            from openai import OpenAI
            import os
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Create rephrasing prompt
            rephrasing_prompt = """Make MINIMAL stylistic changes to the following text while preserving the exact same meaning and functionality. Only make very small changes like:
- Replace 1-2 words with close synonyms (e.g. "use" → "utilize", "help" → "assist") 
- Minor grammatical restructuring (e.g. "You can do X" → "One can do X")
- Do NOT change technical terms, code, numbers, or key concepts
- Do NOT change the overall message or tone
- Keep the same length and structure

Text to rephrase:
{text}

Minimally rephrased version:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": rephrasing_prompt.format(text=text)}
                ],
                temperature=0.2,  # Very low creativity for minimal changes
                max_tokens=len(text.split()) * 3  # Reasonable length limit
            )
            
            rephrased = response.choices[0].message.content.strip()
            
            # Basic validation - ensure rephrasing isn't too different in length
            length_ratio = len(rephrased) / max(len(text), 1)
            if 0.5 <= length_ratio <= 2.0:  # Allow 50%-200% length variation
                return rephrased
            else:
                # If rephrasing changes length too much, fall back to original
                print(f"⚠️ Rephrasing length check failed: {len(text)} -> {len(rephrased)}")
                return text
                
        except Exception as e:
            print(f"⚠️ Rephrasing failed: {e}")
            # Fall back to minor text modifications
            return self._apply_fallback_changes(text)
    
    def _apply_fallback_changes(self, text: str) -> str:
        """Apply minimal fallback changes when rephrasing fails."""
        # Simple synonym replacements that don't change meaning
        replacements = [
            ('the', 'a'),
            ('and', '&'),
            ('you', 'one'),
            ('can', 'may'),
            ('should', 'ought to'),
            ('will', 'shall'),
            ('because', 'since'),
            ('also', 'additionally'),
            ('however', 'nevertheless'),
            ('therefore', 'thus')
        ]
        
        modified_text = text
        
        # Try one random replacement
        if replacements and self.rng.random() < 0.3:
            old_word, new_word = self.rng.choice(replacements)
            if f' {old_word} ' in modified_text.lower():
                # Case-preserving replacement
                import re
                pattern = re.compile(rf'\b{re.escape(old_word)}\b', re.IGNORECASE)
                match = pattern.search(modified_text)
                if match:
                    original = match.group()
                    if original.isupper():
                        replacement = new_word.upper()
                    elif original.istitle():
                        replacement = new_word.title()
                    else:
                        replacement = new_word
                    modified_text = pattern.sub(replacement, modified_text, count=1)
        
        return modified_text
    
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
        from ..interfaces.arena_hard_utils import load_judgment_data as _load_judgment_data
        from ..interfaces.arena_hard_utils import parse_arena_hard_prompt as _parse_arena_hard_prompt
        
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
                    allowed_fields = self.text_fields if self.text_fields is not None else ['question', 'answer_a', 'answer_b']
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
        
        # Import the Arena-Hard data loading functions
        try:
            from ..interfaces.oumi_interface import _load_judgment_data, _parse_arena_hard_prompt
        except ImportError:
            try:
                from differential_debiasing.interfaces.oumi_interface import _load_judgment_data, _parse_arena_hard_prompt
            except ImportError:
                import sys
                from pathlib import Path
                parent_dir = Path(__file__).parent.parent
                sys.path.insert(0, str(parent_dir))
                from interfaces.oumi_interface import _load_judgment_data, _parse_arena_hard_prompt
        
        print(f"🔧 Creating {num_neighbors} single-sample perturbation experiments for {len(sampled_df)} samples...")
        
        # Load all original Arena-Hard data first (do this once)
        original_data = {}
        successful_samples = []
        original_contexts = []
        
        for _, row in sampled_df.iterrows():
            question_id = row['question_id'] 
            model_name = row['model']
            
            try:
                # Load original Arena-Hard data for this sample
                judgment_data = _load_judgment_data(base_processed_dir, question_id, model_name)
                user_prompt = judgment_data['games'][0]['user_prompt']
                question, answer_a, answer_b = _parse_arena_hard_prompt(user_prompt)
                
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
        
        for neighbor_idx in range(num_neighbors):
            # For each experiment, choose exactly ONE sample to perturb
            sample_to_perturb_idx = self.rng.randint(0, len(successful_samples))
            question_id_to_perturb = successful_samples[sample_to_perturb_idx]
            
            # Create neighbor dataset: all original samples + one perturbed sample
            neighbor_contexts = []
            
            for i, question_id in enumerate(successful_samples):
                original = original_data[question_id]
                
                if i == sample_to_perturb_idx:
                    # This is the sample we perturb for this experiment
                    allowed_fields = self.text_fields if self.text_fields is not None else ['question', 'answer_a', 'answer_b']
                    fields_to_modify = allowed_fields
                    if self.single_field and len(allowed_fields) > 0:
                        fields_to_modify = [self.rng.choice(allowed_fields)]

                    perturbed_question = original['question']
                    perturbed_answer_a = original['answer_a']
                    perturbed_answer_b = original['answer_b']
                    
                    if 'question' in fields_to_modify:
                        perturbed_question = self._apply_formatting_changes(original['question'])
                    if 'answer_a' in fields_to_modify:
                        perturbed_answer_a = self._apply_formatting_changes(original['answer_a'])
                    if 'answer_b' in fields_to_modify:
                        perturbed_answer_b = self._apply_formatting_changes(original['answer_b'])
                    
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
            
            all_neighbors.append(neighbor_contexts)
        
        print(f"✅ Created {len(all_neighbors)} single-sample perturbation experiments")
        print(f"   Each experiment perturbs exactly 1 out of {len(successful_samples)} samples")
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
