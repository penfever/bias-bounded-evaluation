#!/usr/bin/env python3
"""
Judge Query Interface for Dynamic A-BB Bias Measurement

This module provides a unified interface for querying both remote API judges
(OpenAI, Anthropic, Together) and local GGUF models via Oumi for bias sensitivity testing.
"""

import os
import json
import yaml
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import time
import hashlib
import numpy as np
import pandas as pd

from .arena_hard_utils import (
    build_pairwise_judge_prompt as ah_build_pairwise_prompt,
    extract_pairwise_from_judge_response as ah_extract_pairwise,
    verdict_token_to_score as ah_token_to_score,
)


class JudgeQueryInterface:
    """
    Unified interface for querying judges using Arena-Hard Auto infrastructure
    and Oumi for local models.
    """
    
    def __init__(self, 
                 judge_name: str,
                 api_config_path: Optional[str] = None,
                 judge_config_path: Optional[str] = None,
                 cache_responses: bool = True,
                 cost_budget_usd: float = 10.0):
        """
        Initialize judge query interface.
        
        Parameters:
        -----------
        judge_name : str
            Name of the judge model (must match API config)
        api_config_path : str, optional
            Path to Arena-Hard Auto API config. Defaults to arena-hard-auto/config/api_config.yaml
        judge_config_path : str, optional  
            Path to judge prompt config. Defaults to arena-hard-auto/config/judge_config.yaml
        cache_responses : bool
            Whether to cache responses to avoid duplicate queries
        cost_budget_usd : float
            Maximum cost budget for API queries (safety limit)
        """
        self.judge_name = judge_name
        self.cache_responses = cache_responses
        self.cost_budget_usd = cost_budget_usd
        self.spent_usd = 0.0
        
        # Load configurations
        self.api_config_path = api_config_path or (ARENA_HARD_PATH / "config" / "api_config.yaml")
        self.judge_config_path = judge_config_path or (ARENA_HARD_PATH / "config" / "judge_config.yaml") 
        
        self._load_configs()
        
        # Response cache
        self._response_cache = {} if cache_responses else None
        
        # Cost tracking
        self._cost_per_token = self._estimate_cost_per_token()
        
    def _load_configs(self):
        """Load API and judge configurations."""
        try:
            with open(self.api_config_path, 'r') as f:
                self.api_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load API config from {self.api_config_path}: {e}")
            self.api_config = {}
            
        try:
            with open(self.judge_config_path, 'r') as f:
                self.judge_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load judge config from {self.judge_config_path}: {e}")
            self.judge_config = {}
            
        # Get judge-specific configuration
        if self.judge_name in self.api_config:
            self.judge_api_config = self.api_config[self.judge_name]
        else:
            raise ValueError(f"Judge '{self.judge_name}' not found in API config")
            
    def _estimate_cost_per_token(self) -> float:
        """Estimate cost per token based on model type."""
        model_name = self.judge_api_config.get('model_name', '').lower()
        api_type = self.judge_api_config.get('api_type', '')
        
        # Rough cost estimates (USD per 1000 tokens)
        if 'gpt-4' in model_name and 'turbo' not in model_name:
            return 0.03 / 1000  # GPT-4
        elif 'gpt-4' in model_name:
            return 0.01 / 1000  # GPT-4 Turbo  
        elif 'gpt-3.5' in model_name:
            return 0.002 / 1000  # GPT-3.5
        elif 'claude' in model_name:
            return 0.015 / 1000  # Claude
        elif api_type in ['huggingface_local', 'oumi']:
            return 0.0  # Local models are free
        else:
            return 0.01 / 1000  # Default estimate
            
    def _get_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for context."""
        if not self.cache_responses:
            return None
        
        # Create hash of context for caching
        context_str = json.dumps(context, sort_keys=True)
        cache_key = f"{self.judge_name}:{hashlib.md5(context_str.encode()).hexdigest()}"
        return cache_key
        
    def query_judge(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Query the judge with a given context.
        
        Parameters:
        -----------
        context : Dict[str, Any]
            Context containing question, answer, and other relevant information
        **kwargs : dict
            Additional parameters for the judge query
            
        Returns:
        --------
        Dict[str, Any] : Judge response with scores and reasoning
        """
        # Check cache first
        cache_key = self._get_cache_key(context)
        if cache_key and cache_key in self._response_cache:
            return self._response_cache[cache_key]
            
        # Check cost budget
        if self.spent_usd >= self.cost_budget_usd:
            raise RuntimeError(f"Cost budget exceeded: ${self.spent_usd:.2f} >= ${self.cost_budget_usd:.2f}")
        
        # Route to appropriate query method
        api_type = self.judge_api_config.get('api_type')
        
        if api_type == 'oumi':
            response = self._query_oumi_judge(context, **kwargs)
        elif api_type == 'openai':
            response = self._query_openai_judge(context, **kwargs)
        elif api_type == 'anthropic':
            response = self._query_anthropic_judge(context, **kwargs)
        elif api_type == 'together':
            response = self._query_together_judge(context, **kwargs)
        elif api_type in ['huggingface', 'huggingface_local']:
            response = self._query_huggingface_judge(context, **kwargs)
        else:
            raise ValueError(f"Unsupported API type: {api_type}")
            
        # Update cost tracking
        if 'token_count' in response:
            cost = response['token_count'] * self._cost_per_token
            self.spent_usd += cost
            response['estimated_cost_usd'] = cost
            
        # Cache response
        if cache_key:
            self._response_cache[cache_key] = response
            
        return response
        
    def _query_oumi_judge(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Query judge using Oumi local inference."""
        # Create temporary config file for this query
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            oumi_config = self._create_oumi_config(context)
            yaml.dump(oumi_config, f)
            config_path = f.name
            
        try:
            # Create input prompt
            prompt = self._format_judge_prompt(context)
            
            # Run Oumi inference
            cmd = [
                'oumi', 'infer', 
                '-c', config_path,
                '--input', prompt
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Oumi inference failed: {result.stderr}")
                
            # Parse output
            output_text = result.stdout.strip()
            scores = self._parse_judge_response(output_text)
            
            return {
                'raw_output': output_text,
                'scores': scores,
                'token_count': len(output_text.split()),  # Rough estimate
                'api_type': 'oumi'
            }
            
        finally:
            # Clean up temp file
            os.unlink(config_path)
            
    def _query_openai_judge(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Query judge using OpenAI API (direct SDK fallback)."""
        return self._fallback_openai_query(context, **kwargs)
            
    def _fallback_openai_query(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Fallback OpenAI query using direct API call."""
        try:
            import openai
            
            prompt = self._format_judge_prompt(context)
            
            # Get API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not found in environment")
                
            client = openai.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.judge_api_config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048
            )
            
            response_text = response.choices[0].message.content
            scores = self._parse_judge_response(response_text)
            
            return {
                'raw_output': response_text,
                'scores': scores,
                'token_count': response.usage.total_tokens if response.usage else len(response_text.split()),
                'api_type': 'openai'
            }
            
        except ImportError:
            raise RuntimeError("OpenAI library not available and Arena-Hard Auto utils failed to load")
        except Exception as e:
            raise RuntimeError(f"Fallback OpenAI query failed: {e}")
            
    def _query_anthropic_judge(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Query judge using Anthropic SDK directly."""
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            if client.api_key is None:
                raise RuntimeError("ANTHROPIC_API_KEY not found in environment")
            prompt = self._format_judge_prompt(context)
            msg = client.messages.create(
                model=self.judge_api_config['model_name'],
                max_tokens=2048,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text
            content = ''.join([blk.text for blk in msg.content if hasattr(blk, 'text')]) if hasattr(msg, 'content') else str(msg)
            scores = self._parse_judge_response(content)
            tok = getattr(msg, 'usage', None)
            total_tokens = getattr(tok, 'input_tokens', 0) + getattr(tok, 'output_tokens', 0) if tok else len(content.split())
            return { 'raw_output': content, 'scores': scores, 'token_count': total_tokens, 'api_type': 'anthropic' }
        except Exception as e:
            raise RuntimeError(f"Anthropic judge query failed: {e}")
            
    def _query_together_judge(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Query judge using Together SDK directly."""
        try:
            import together
            together.api_key = os.getenv('TOGETHER_API_KEY')
            if not together.api_key:
                raise RuntimeError("TOGETHER_API_KEY not found in environment")
            prompt = self._format_judge_prompt(context)
            resp = together.Chat.completions.create(
                model=self.judge_api_config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            content = resp.choices[0].message.get('content', '') if hasattr(resp, 'choices') else str(resp)
            scores = self._parse_judge_response(content)
            return { 'raw_output': content, 'scores': scores, 'token_count': len(content.split()), 'api_type': 'together' }
        except Exception as e:
            raise RuntimeError(f"Together judge query failed: {e}")
            
    def _query_huggingface_judge(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Query judge using HuggingFace Inference API directly (or local)."""
        try:
            prompt = self._format_judge_prompt(context)
            if self.judge_api_config.get('api_type') == 'huggingface_local':
                # Local inference not implemented here; prefer Oumi path
                raise RuntimeError("Use api_type='oumi' for local GGUF models")
            from huggingface_hub import InferenceClient
            token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            if not token:
                raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not found in environment")
            client = InferenceClient(token=token)
            resp = client.chat.completions.create(
                model=self.judge_api_config['model_name'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            content = resp.choices[0].message.get('content', '')
            scores = self._parse_judge_response(content)
            return { 'raw_output': content, 'scores': scores, 'token_count': len(content.split()), 'api_type': 'huggingface' }
        except Exception as e:
            raise RuntimeError(f"HuggingFace judge query failed: {e}")
            
    def _create_oumi_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create Oumi configuration for local inference."""
        # Base template similar to the DeepSeek example
        base_config = {
            'model': {
                'model_name': self.judge_api_config['model_name'],
                'tokenizer_name': self.judge_api_config.get('tokenizer_name', self.judge_api_config['model_name']),
                'model_max_length': 8192,
                'torch_dtype_str': 'float16',
                'trust_remote_code': True
            },
            'generation': {
                'max_new_tokens': 2048,
                'temperature': 0.0,
                'top_p': 0.9,
                'do_sample': False
            },
            'engine': 'LLAMACPP'
        }
        
        # Add model-specific parameters
        if 'model_kwargs' in self.judge_api_config:
            base_config['model']['model_kwargs'] = self.judge_api_config['model_kwargs']
            
        return base_config
        
    def _format_judge_prompt(self, context: Dict[str, Any]) -> str:
        """Format the judge prompt.

        If pairwise answers are present (answer_a, answer_b), use the Arena-Hard
        pairwise prompt from our local utils. Otherwise, fall back to judge config
        template or a default single-answer template.
        """
        # Pairwise path
        if 'answer_a' in context and 'answer_b' in context:
            q = context.get('question', '')
            return ah_build_pairwise_prompt(q, context.get('answer_a', ''), context.get('answer_b', ''))
        # Default template if judge config not available
        default_template = """Please act as an impartial judge and evaluate the quality of the response provided to the user question below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a brief explanation of your reasoning. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[8]]".

[Question]
{question}

[The Start of Response]
{answer}
[The End of Response]"""

        # Use judge config template if available
        if isinstance(self.judge_config, dict) and 'prompt_template' in self.judge_config:
            template = self.judge_config['prompt_template']
            if isinstance(template, list):
                # If template is a list, join it or take first element
                template = template[0] if template else default_template
        else:
            template = default_template
            
        # Ensure template is a string
        if not isinstance(template, str):
            template = default_template
            
        # Format with context
        try:
            # Create a safe context dict without conflicting keys
            format_context = {
                'question': context.get('question', ''),
                'answer': context.get('answer', '')
            }
            # Add other context fields that don't conflict
            for key, value in context.items():
                if key not in format_context and isinstance(value, (str, int, float)):
                    format_context[key] = value
                    
            return template.format(**format_context)
        except KeyError as e:
            # If template has missing keys, use default template
            print(f"Warning: Template formatting failed ({e}), using default template")
            return default_template.format(
                question=context.get('question', ''),
                answer=context.get('answer', '')
            )
        
    def _parse_judge_response(self, response: str) -> Dict[str, float]:
        """Parse judge response to extract scores.

        Supports Arena-Hard pairwise tokens mapped to Likert [1..5] and numeric
        ratings as fallbacks.
        """
        scores = {}
        # First try Arena-Hard pairwise extraction
        token = ah_extract_pairwise(response)
        if token:
            num = ah_token_to_score(token)
            if num is not None:
                scores['overall_score'] = float(num)
                scores['pairwise_comparison'] = token
                return scores

        # Look for numeric rating patterns
        import re
        # Pattern 1: [[rating]] format
        rating_match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', response)
        if rating_match:
            scores['overall_score'] = float(rating_match.group(1))
            
        # Pattern 2: Rating: X format  
        if 'overall_score' not in scores:
            rating_match = re.search(r'[Rr]ating:\s*(\d+(?:\.\d+)?)', response)
            if rating_match:
                scores['overall_score'] = float(rating_match.group(1))
                
        # Pattern 3: Score: X format
        if 'overall_score' not in scores:
            rating_match = re.search(r'[Ss]core:\s*(\d+(?:\.\d+)?)', response)
            if rating_match:
                scores['overall_score'] = float(rating_match.group(1))
                
        # If no score found, default to neutral
        if 'overall_score' not in scores:
            scores['overall_score'] = 3.0
        # Convert to 1-5 scale if needed (Arena-Hard uses 1-5)
        if scores['overall_score'] > 5:
            scores['overall_score'] = scores['overall_score'] / 2.0  # 10-scale to 5-scale
            
        return scores
        
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost and usage summary."""
        return {
            'judge_name': self.judge_name,
            'api_type': self.judge_api_config.get('api_type'),
            'spent_usd': self.spent_usd,
            'budget_usd': self.cost_budget_usd,
            'budget_remaining': self.cost_budget_usd - self.spent_usd,
            'queries_cached': len(self._response_cache) if self._response_cache else 0
        }


def create_judge_function(judge_name: str, **kwargs) -> callable:
    """
    Create a judge function compatible with A-BB sensitivity estimation.
    
    Parameters:
    -----------
    judge_name : str
        Name of the judge model to use
    **kwargs : dict
        Additional parameters for JudgeQueryInterface
        
    Returns:
    --------
    callable : Judge function that takes context and returns scores
    """
    judge_interface = JudgeQueryInterface(judge_name, **kwargs)
    
    def judge_function(context):
        """
        Judge function for A-BB sensitivity estimation.
        
        Parameters:
        -----------
        context : Dict or DataFrame
            Context containing question/answer information
            
        Returns:
        --------
        np.ndarray : Array of scores from the judge
        """
        # Handle different context types
        if isinstance(context, pd.DataFrame):
            # For DataFrame, evaluate each row
            scores = []
            for _, row in context.iterrows():
                query_context = {
                    'question': row.get('question', ''),
                    'answer': row.get('answer', ''),
                    'model': row.get('model', '')
                }
                response = judge_interface.query_judge(query_context)
                scores.append(response['scores']['overall_score'])
            return np.array(scores)
            
        elif isinstance(context, dict):
            # For single context dict
            response = judge_interface.query_judge(context)
            return np.array([response['scores']['overall_score']])
            
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
    
    # Attach interface for cost tracking
    judge_function.interface = judge_interface
    return judge_function


def test_judge_interface():
    """Test function for the judge interface."""
    print("Testing Judge Query Interface...")
    
    # Test basic interface
    try:
        judge = JudgeQueryInterface('gpt-3.5-turbo-0125')  # Start with API model
        
        test_context = {
            'question': 'What is 2 + 2?',
            'answer': 'The answer is 4.'
        }
        
        response = judge.query_judge(test_context)
        print(f"Basic test response: {response}")
        print(f"Cost summary: {judge.get_cost_summary()}")
        
        # Test judge function wrapper
        judge_func = create_judge_function('gpt-3.5-turbo-0125')
        score = judge_func(test_context)
        print(f"Judge function score: {score}")
        print(f"Judge function cost: {judge_func.interface.get_cost_summary()}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_judge_interface()
