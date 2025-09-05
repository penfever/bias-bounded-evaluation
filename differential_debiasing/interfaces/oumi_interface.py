#!/usr/bin/env python3
"""
Oumi-Based Judge Interface for Dynamic A-BB Bias Measurement

This module provides a unified interface for querying both API-based judges
and local GGUF models through Oumi's inference system.
"""

import os
import sys
import yaml
import tempfile
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import numpy as np
import pandas as pd

# Import sampling utilities
from ..core.sampling_utils import apply_intelligent_sampling

# Import Arena-Hard-Auto utilities for proper score extraction  
ARENA_HARD_PATH = Path(__file__).parent.parent.parent / "examples" / "arena-hard-auto"
sys.path.insert(0, str(ARENA_HARD_PATH))

# Import both utils and gen_judgment for complete Arena-Hard-Auto functionality
import importlib.util
spec = importlib.util.spec_from_file_location("arena_utils", ARENA_HARD_PATH / "utils.py")
arena_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arena_utils)

spec_judgment = importlib.util.spec_from_file_location("gen_judgment", ARENA_HARD_PATH / "gen_judgment.py")
gen_judgment = importlib.util.module_from_spec(spec_judgment)
spec_judgment.loader.exec_module(gen_judgment)

# Extract required functions
get_score = gen_judgment.get_score
print("✅ Arena-Hard-Auto score extraction loaded successfully")

# Import Oumi from environment
try:
    from oumi.builders.inference_engines import build_inference_engine
    from oumi.core.configs import InferenceConfig, ModelParams, GenerationParams, InferenceEngineType, RemoteParams
    from oumi.core.types.conversation import Conversation, Message, Role, ContentItem, Type
    print("✅ Oumi imports successful")
except ImportError as e:
    raise ImportError(f"Oumi not available in environment: {e}")
    
# Arena-Hard pairwise comparison to numerical score mapping
PAIRWISE_TO_SCORE = {
    'A>>B': 1.0,  # Assistant A significantly better
    'A>B': 2.0,   # Assistant A slightly better  
    'A=B': 3.0,   # Tie
    'B>A': 4.0,   # Assistant B slightly better
    'B>>A': 5.0,  # Assistant B significantly better
    # Handle alternative formats
    'A<<B': 5.0,
    'A<B': 4.0,
    'B=A': 3.0,
    'B<A': 2.0,
    'B<<A': 1.0
}

# Reverse mapping for converting scores back to pairwise comparisons
SCORE_TO_PAIRWISE = {v: k for k, v in PAIRWISE_TO_SCORE.items()}


class OumiJudgeInterface:
    """
    Unified judge interface using Oumi for both API-based and local GGUF models.
    """
    
    def __init__(self, 
                 config_path: str,
                 cost_budget_usd: float = 10.0,
                 cache_responses: bool = True,
                 prefer_existing_scores: bool = True):
        """
        Initialize Oumi judge interface.
        
        Parameters:
        -----------
        config_path : str
            Path to Oumi inference config (YAML)
        cost_budget_usd : float
            Maximum cost budget for API queries
        cache_responses : bool
            Whether to cache responses
        """
        self.config_path = config_path
        self.cost_budget_usd = cost_budget_usd
        self.spent_usd = 0.0
        self.cache_responses = cache_responses
        self.prefer_existing_scores = prefer_existing_scores
        
        # Load config
        self.config = self._load_oumi_config()
        
        # Response cache
        self._response_cache = {} if cache_responses else None
        
        # Cost tracking
        self._cost_per_token = self._estimate_cost_per_token()
        
        # Initialize inference engine (lazy loading)
        self._inference_engine = None
        
        # Batching configuration
        self.batch_size = int(self.config.get('batch_size', os.getenv('OUMI_BATCH_SIZE', 8)))
        self.max_retries = int(self.config.get('max_retries', os.getenv('OUMI_MAX_RETRIES', 3)))
        self.retry_backoff_sec = float(self.config.get('retry_backoff_sec', os.getenv('OUMI_RETRY_BACKOFF_SEC', 1.0)))
    
    def _load_oumi_config(self) -> Dict[str, Any]:
        """Load Oumi inference configuration."""
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _get_inference_engine(self):
        """Get or create inference engine (lazy loading)."""
        if self._inference_engine is None:
            try:
                # Extract engine type from config
                engine_str = self.config.get('engine', 'NATIVE')
                engine_type = InferenceEngineType(engine_str)
                
                # Create model params
                model_config = self.config.get('model', {})
                model_params = ModelParams(
                    model_name=model_config.get('model_name'),
                    tokenizer_name=model_config.get('tokenizer_name'),
                    model_max_length=model_config.get('model_max_length', 8192),
                    torch_dtype_str=model_config.get('torch_dtype_str', 'float16'),
                    trust_remote_code=model_config.get('trust_remote_code', True),
                    model_kwargs=model_config.get('model_kwargs', {})
                )
                
                # Create generation params - only include supported parameters
                gen_config = self.config.get('generation', {})
                generation_params = GenerationParams(
                    max_new_tokens=gen_config.get('max_new_tokens', 2048),
                    temperature=gen_config.get('temperature', 0.0),
                    top_p=gen_config.get('top_p', 0.9)
                )

                # Remote params (API engines): api_key, workers, politeness
                remote_cfg = self.config.get('remote', {})
                # Derive API key based on engine if not present in YAML
                api_key = remote_cfg.get('api_key')
                if not api_key:
                    if engine_type == InferenceEngineType.OPENAI:
                        api_key = os.getenv('OPENAI_API_KEY')
                    elif engine_type == InferenceEngineType.ANTHROPIC:
                        api_key = os.getenv('ANTHROPIC_API_KEY')
                num_workers = int(remote_cfg.get('num_workers', os.getenv('OUMI_NUM_WORKERS', 8)))
                politeness_policy = float(remote_cfg.get('politeness_policy', os.getenv('OUMI_POLITENESS_SEC', 60.0)))
                remote_params = None
                # Only pass RemoteParams for API engines
                if engine_type in [InferenceEngineType.OPENAI, InferenceEngineType.ANTHROPIC]:
                    remote_params = RemoteParams(
                        api_key=api_key,
                        num_workers=num_workers,
                        politeness_policy=politeness_policy
                    )
                
                # Build inference engine
                self._inference_engine = build_inference_engine(
                    engine_type=engine_type,
                    model_params=model_params,
                    generation_params=generation_params,
                    remote_params=remote_params
                )
                
                print(f"✅ Initialized {engine_type} inference engine")
                
            except Exception as e:
                print(f"❌ Failed to initialize Oumi inference engine: {e}")
                self._inference_engine = None
        
        return self._inference_engine
    
    def _estimate_cost_per_token(self) -> float:
        """Estimate cost per token based on engine type."""
        engine = self.config.get('engine', '').upper()
        model_name = self.config.get('model', {}).get('model_name', '').lower()
        
        # Cost estimates (USD per 1000 tokens)
        if engine == 'OPENAI':
            if 'gpt-4o' in model_name:
                return 0.01 / 1000  # GPT-4o
            elif 'gpt-4' in model_name:
                return 0.03 / 1000  # GPT-4
            elif 'gpt-3.5' in model_name:
                return 0.002 / 1000  # GPT-3.5
        elif engine == 'ANTHROPIC':
            return 0.015 / 1000  # Claude
        elif engine in ['LLAMACPP', 'NATIVE', 'VLLM']:
            return 0.0  # Local models are free
        else:
            return 0.01 / 1000  # Default estimate
    
    def _get_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for context."""
        if not self.cache_responses:
            return None
        
        # Create hash of context for caching
        import hashlib
        context_str = json.dumps(context, sort_keys=True)
        cache_key = f"{self.config_path}:{hashlib.md5(context_str.encode()).hexdigest()}"
        return cache_key
    
    def query_judge(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Query the judge with a given context using Oumi.
        
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
        
        # Format prompt
        prompt = self._format_judge_prompt(context)
        
        # Query using Oumi (no fallback) with timeout in test mode
        test_mode = os.getenv('TEST_MODE') == 'true'
        if test_mode:
            # Add timeout for test mode to prevent hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Judge query timed out in test mode")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(15)  # 15 second timeout for test mode
            
            try:
                response = self._query_with_oumi(prompt)
            finally:
                signal.alarm(0)  # Cancel timeout
        else:
            response = self._query_with_oumi(prompt)
        
        # Parse scores
        scores = self._parse_judge_response(response['raw_output'])
        response['scores'] = scores
        
        # Update cost tracking
        if 'token_count' in response and self._cost_per_token > 0:
            cost = response['token_count'] * self._cost_per_token
            self.spent_usd += cost
            response['estimated_cost_usd'] = cost
        
        # Cache response
        if cache_key:
            self._response_cache[cache_key] = response
            
        return response
    
    def _query_with_oumi(self, prompt: str) -> Dict[str, Any]:
        """Query using Oumi library directly."""
        engine = self._get_inference_engine()
        if engine is None:
            raise RuntimeError("Oumi inference engine not available")
        
        # Create conversation (string content for infer_online)
        conversation = Conversation(messages=[Message(role=Role.USER, content=prompt)])
        
        # Generate response using correct Oumi API
        try:
            # Build inference config from existing config/engine
            gen_config = self.config.get('generation', {})
            generation_params = GenerationParams(
                max_new_tokens=gen_config.get('max_new_tokens', 2048),
                temperature=gen_config.get('temperature', 0.0),
                top_p=gen_config.get('top_p', 0.9)
            )
            inference_config = InferenceConfig(
                generation=generation_params,
                model=getattr(engine, '_model_params', None)
            )
            
            # Prefer infer_online; fall back to infer if not available
            if hasattr(engine, 'infer_online'):
                response_conversations = engine.infer_online([conversation], inference_config)
            else:
                conv = Conversation(messages=[
                    Message(role=Role.USER, content=[ContentItem(type=Type.TEXT, content=prompt)])
                ])
                response_conversations = engine.infer(input=[conv])
            
            # Extract response text from the returned conversation list
            if response_conversations and len(response_conversations) > 0:
                response_conversation = response_conversations[0]
                if hasattr(response_conversation, 'messages') and response_conversation.messages:
                    # Find the assistant's response (last message)
                    for message in reversed(response_conversation.messages):
                        if hasattr(message, 'role') and message.role == Role.ASSISTANT:
                            if hasattr(message, 'content') and message.content:
                                # Handle both string content and ContentItem list
                                if isinstance(message.content, str):
                                    response_text = message.content
                                elif isinstance(message.content, list):
                                    # Use helper method to get flattened text
                                    response_text = message.compute_flattened_text_content()
                                else:
                                    response_text = str(message.content)
                                break
                    else:
                        # No assistant message found, take the last message
                        last_message = response_conversation.messages[-1]
                        if hasattr(last_message, 'content') and last_message.content:
                            if isinstance(last_message.content, str):
                                response_text = last_message.content
                            elif isinstance(last_message.content, list):
                                response_text = last_message.compute_flattened_text_content()
                            else:
                                response_text = str(last_message.content)
                        else:
                            response_text = str(last_message)
                else:
                    response_text = str(response_conversation)
            else:
                response_text = ""
            
            return {
                'raw_output': response_text,
                'token_count': len(response_text.split()) if isinstance(response_text, str) else 0,
                'api_type': 'oumi_direct'
            }
            
        except Exception as e:
            raise RuntimeError(f"Oumi direct inference failed: {e}")

    def _infer_conversations(self, conversations: List[Conversation]) -> List[str]:
        """Infer a batch of conversations and return assistant text per item (order-preserving)."""
        engine = self._get_inference_engine()
        if engine is None:
            raise RuntimeError("Oumi inference engine not available")

        # Simple retry with backoff for API engines
        attempt = 0
        while True:
            try:
                # Build inference config fresh to allow runtime tuning
                gen_config = self.config.get('generation', {})
                generation_params = GenerationParams(
                    max_new_tokens=gen_config.get('max_new_tokens', 2048),
                    temperature=gen_config.get('temperature', 0.0),
                    top_p=gen_config.get('top_p', 0.9)
                )
                inference_config = InferenceConfig(
                    generation=generation_params,
                    model=getattr(engine, '_model_params', None)
                )
                if hasattr(engine, 'infer_online'):
                    response_conversations = engine.infer_online(conversations, inference_config)
                else:
                    # Ensure messages use ContentItem for infer()
                    convs_for_infer: List[Conversation] = []
                    for conv in conversations:
                        msgs = []
                        for m in conv.messages:
                            if isinstance(m.content, str):
                                msgs.append(Message(role=m.role, content=[ContentItem(type=Type.TEXT, content=m.content)]))
                            else:
                                msgs.append(m)
                        convs_for_infer.append(Conversation(messages=msgs))
                    response_conversations = engine.infer(input=convs_for_infer)
                outputs: List[str] = []
                for resp_conv in response_conversations or []:
                    response_text = ""
                    if hasattr(resp_conv, 'messages') and resp_conv.messages:
                        for message in reversed(resp_conv.messages):
                            if hasattr(message, 'role') and message.role == Role.ASSISTANT:
                                if isinstance(message.content, str):
                                    response_text = message.content
                                elif isinstance(message.content, list):
                                    response_text = message.compute_flattened_text_content()
                                else:
                                    response_text = str(message.content)
                                break
                        else:
                            last_message = resp_conv.messages[-1]
                            if isinstance(last_message.content, str):
                                response_text = last_message.content
                            elif isinstance(last_message.content, list):
                                response_text = last_message.compute_flattened_text_content()
                            else:
                                response_text = str(last_message.content)
                    else:
                        response_text = str(resp_conv)
                    outputs.append(response_text)
                # If engine returns fewer responses than inputs, pad with empty strings
                while len(outputs) < len(conversations):
                    outputs.append("")
                return outputs
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(f"Oumi batched inference failed after {self.max_retries} retries: {e}")
                import time
                time.sleep(self.retry_backoff_sec * (2 ** (attempt - 1)))

    def query_judge_batch(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batched judge query with caching, chunking, and order preservation."""
        # Prepare outputs list
        results: List[Optional[Dict[str, Any]]] = [None] * len(contexts)

        # Resolve cache and prepare items to query
        to_query_indices: List[int] = []
        to_query_prompts: List[str] = []
        to_query_cache_keys: List[Optional[str]] = []

        for idx, ctx in enumerate(contexts):
            cache_key = self._get_cache_key(ctx)
            if cache_key and cache_key in self._response_cache:
                results[idx] = self._response_cache[cache_key]
            else:
                to_query_indices.append(idx)
                to_query_prompts.append(self._format_judge_prompt(ctx))
                to_query_cache_keys.append(cache_key)

        # If nothing to query, return cached results
        if not to_query_indices:
            return [r for r in results if r is not None]

        # Chunk prompts into conversations and infer
        for start in range(0, len(to_query_prompts), self.batch_size):
            end = min(start + self.batch_size, len(to_query_prompts))
            chunk_prompts = to_query_prompts[start:end]
            chunk_indices = to_query_indices[start:end]
            chunk_cache_keys = to_query_cache_keys[start:end]

            conversations = [Conversation(messages=[Message(role=Role.USER, content=p)]) for p in chunk_prompts]

            outputs = self._infer_conversations(conversations)

            # Parse each output and record
            for out_text, idx, cache_key in zip(outputs, chunk_indices, chunk_cache_keys):
                parsed = {
                    'raw_output': out_text,
                    'token_count': len(out_text.split()) if isinstance(out_text, str) else 0,
                    'api_type': 'oumi_direct'
                }
                parsed['scores'] = self._parse_judge_response(parsed['raw_output'])

                # Cost tracking per item
                if 'token_count' in parsed and self._cost_per_token > 0:
                    cost = parsed['token_count'] * self._cost_per_token
                    self.spent_usd += cost
                    parsed['estimated_cost_usd'] = cost

                results[idx] = parsed
                if cache_key:
                    self._response_cache[cache_key] = parsed

            # Budget check after each chunk
            if self.spent_usd >= self.cost_budget_usd:
                raise RuntimeError(f"Cost budget exceeded: ${self.spent_usd:.2f} >= ${self.cost_budget_usd:.2f}")

        return [r for r in results if r is not None]
    
    
    def _format_judge_prompt(self, context: Dict[str, Any]) -> str:
        """Format the judge prompt using Arena-Hard-Auto pairwise comparison template."""
        # Use Arena-Hard-Auto system prompt and template for pairwise comparison
        system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

IMPORTANT: You must end your response with exactly one of the bracketed patterns above (e.g., [[A>B]]). Do not include any text after the final verdict pattern.

Example output: "My final verdict is tie: [[A=B]]"."""

        # Arena-Hard-Auto prompt template for pairwise comparison
        template = """<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""

        # For sensitivity analysis, we need to create a pairwise comparison
        # Use the original answer as A and a neighbor/variation as B (if available)
        question = context.get('question', '')
        answer_a = context.get('answer', context.get('answer_a', ''))
        answer_b = context.get('answer_b', answer_a)  # Use same answer if no variation provided
        
        # If we only have one answer, create a simple variation for comparison
        if answer_a == answer_b and 'neighbor_answer' in context:
            answer_b = context['neighbor_answer']
        elif answer_a == answer_b:
            # Create a minimal variation by adding a comment
            answer_b = answer_a + "\n\n[This response has been slightly modified for comparison purposes.]"

        try:
            formatted_prompt = template.format(
                question=question,
                answer_a=answer_a,
                answer_b=answer_b
            )
            return system_prompt + "\n\n" + formatted_prompt
        except KeyError as e:
            raise ValueError(f"Missing required context key for Arena-Hard-Auto template: {e}")
    
    def _parse_judge_response(self, response: str) -> Dict[str, float]:
        """Parse judge response using Arena-Hard-Auto pairwise comparison patterns."""
        scores = {}
        
        # Use Arena-Hard-Auto pattern for pairwise comparisons
        # Pattern matches: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], [[B>>A]]
        arena_pattern = re.compile(r'\[\[([AB<>=]+)\]\]')
        
        # Use Arena-Hard-Auto's get_score function
        pairwise_result, continue_flag = get_score(response, arena_pattern, pairwise=True)
        
        if pairwise_result:
            # Convert pairwise comparison to numerical score using mapping
            if pairwise_result in PAIRWISE_TO_SCORE:
                scores['overall_score'] = PAIRWISE_TO_SCORE[pairwise_result]
                scores['pairwise_comparison'] = pairwise_result
            else:
                raise ValueError(f"Unknown Arena-Hard-Auto pairwise comparison result: '{pairwise_result}'. Expected one of: {list(PAIRWISE_TO_SCORE.keys())}")
        else:
            # Fallback for formatting issues: try to infer from text patterns
            fallback_result = self._fallback_parse_comparison(response)
            if fallback_result:
                scores['overall_score'] = PAIRWISE_TO_SCORE[fallback_result]
                scores['pairwise_comparison'] = fallback_result + "_fallback"
                print(f"⚠️  Used fallback parsing: '{fallback_result}' from response")
            else:
                # For dynamic neighbor generation, use neutral score to avoid breaking the pipeline
                scores['overall_score'] = 3.0  # Neutral/tie score
                scores['pairwise_comparison'] = "A=B_fallback"
                print(f"⚠️  Used neutral fallback score for unparseable response (length: {len(response)} chars)")
            
        return scores
    
    def _fallback_parse_comparison(self, response: str) -> Optional[str]:
        """
        Fallback parser for judge responses that don't follow exact Arena-Hard patterns.
        
        Looks for natural language indicators of preference.
        """
        response_lower = response.lower()
        
        # Strong preference indicators
        if any(phrase in response_lower for phrase in [
            "assistant a is significantly better", "a is much better", "a is clearly superior",
            "assistant a significantly outperforms", "a is substantially better"
        ]):
            return "A>>B"
            
        if any(phrase in response_lower for phrase in [
            "assistant b is significantly better", "b is much better", "b is clearly superior", 
            "assistant b significantly outperforms", "b is substantially better"
        ]):
            return "B>>A"
        
        # Moderate preference indicators
        if any(phrase in response_lower for phrase in [
            "assistant a is better", "a is slightly better", "prefer assistant a",
            "a provides a better", "assistant a's answer is better"
        ]):
            return "A>B"
            
        if any(phrase in response_lower for phrase in [
            "assistant b is better", "b is slightly better", "prefer assistant b",
            "b provides a better", "assistant b's answer is better" 
        ]):
            return "B>A"
        
        # Tie indicators
        if any(phrase in response_lower for phrase in [
            "both answers are", "tie", "equally good", "both provide",
            "similar quality", "comparable answers", "both assistants"
        ]):
            return "A=B"
        
        # Check for verdict-like statements at the end
        last_sentences = response[-300:].lower()  # Check last 300 characters
        
        if "assistant a" in last_sentences and ("better" in last_sentences or "winner" in last_sentences):
            return "A>B"
        elif "assistant b" in last_sentences and ("better" in last_sentences or "winner" in last_sentences):
            return "B>A"
        
        return None
    

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost and usage summary."""
        return {
            'config_path': self.config_path,
            'engine': self.config.get('engine'),
            'model_name': self.config.get('model', {}).get('model_name'),
            'spent_usd': self.spent_usd,
            'budget_usd': self.cost_budget_usd,
            'budget_remaining': self.cost_budget_usd - self.spent_usd,
            'queries_cached': len(self._response_cache) if self._response_cache else 0
        }


def _load_judgment_data(base_dir: str, question_id: str, model_name: str) -> Dict[str, Any]:
    """Load judgment data from Arena-Hard-Auto base_processed directory."""
    import json
    
    # Find the correct setting directory
    for setting_dir in Path(base_dir).glob("*-setting*"):
        base_processed_dir = setting_dir / "base_processed"
        if base_processed_dir.exists():
            model_file = base_processed_dir / f"{model_name}.jsonl"
            if model_file.exists():
                with open(model_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if data.get('question_id') == question_id:
                            return data
    
    raise FileNotFoundError(f"Could not find judgment data for question_id={question_id}, model={model_name}")

def _parse_arena_hard_prompt(user_prompt: str) -> tuple[str, str, str]:
    """Parse Arena-Hard-Auto prompt to extract question and two answers."""
    import re
    
    # Extract the user question
    question_match = re.search(r'<\|User Prompt\|>\s*\n(.*?)\s*\n<\|The Start of Assistant A', user_prompt, re.DOTALL)
    if not question_match:
        raise ValueError("Could not extract question from Arena-Hard prompt")
    question = question_match.group(1).strip()
    
    # Extract Assistant A's answer
    answer_a_match = re.search(r'<\|The Start of Assistant A\'s Answer\|>\s*\n(.*?)\s*\n<\|The End of Assistant A\'s Answer\|>', user_prompt, re.DOTALL)
    if not answer_a_match:
        raise ValueError("Could not extract Assistant A's answer from Arena-Hard prompt")
    answer_a = answer_a_match.group(1).strip()
    
    # Extract Assistant B's answer  
    answer_b_match = re.search(r'<\|The Start of Assistant B\'s Answer\|>\s*\n(.*?)\s*\n<\|The End of Assistant B\'s Answer\|>', user_prompt, re.DOTALL)
    if not answer_b_match:
        raise ValueError("Could not extract Assistant B's answer from Arena-Hard prompt")
    answer_b = answer_b_match.group(1).strip()
    
    return question, answer_a, answer_b


def create_oumi_judge_function(config_path: str, **kwargs) -> callable:
    """
    Create a judge function compatible with A-BB sensitivity estimation using Oumi.
    
    Parameters:
    -----------
    config_path : str
        Path to Oumi inference config
    **kwargs : dict
        Additional parameters for OumiJudgeInterface
        
    Returns:
    --------
    callable : Judge function that takes context and returns scores
    """
    judge_interface = OumiJudgeInterface(config_path, **kwargs)
    
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
            # Import tqdm for progress tracking
            from tqdm import tqdm
            
            # Check if this looks like a Hamming neighbor (has modified score fields)
            # Hamming neighbors should have their scores used directly, not reloaded from Arena-Hard
            has_score_columns = any(col.endswith('_score') for col in context.columns)
            is_likely_hamming_neighbor = has_score_columns and len(context.columns) > 2
            
            if is_likely_hamming_neighbor:
                # For Hamming sensitivity: Use DataFrame scores directly
                # Only log once per batch to avoid repetitive output
                if not hasattr(judge_function, '_hamming_logged'):
                    print(f"🔧 Using DataFrame scores directly for Hamming neighbors (batch of {len(context)})")
                    judge_function._hamming_logged = True
                if 'overall_score' in context.columns:
                    scores = context['overall_score'].values
                    result = np.array(scores)
                    return result
                else:
                    # Fallback to first numeric score column
                    score_cols = [col for col in context.columns if col.endswith('_score')]
                    if score_cols:
                        scores = context[score_cols[0]].values
                        result = np.array(scores)
                        return result
                    else:
                        raise ValueError("No score columns found in DataFrame for Hamming sensitivity")
            
            # For Formatting sensitivity and original data: Load Arena-Hard-Auto data
            # Apply intelligent sampling to manage computational cost
            # CRITICAL: Use fixed random seed for consistent sampling across neighbor calls
            sampled_context = apply_intelligent_sampling(context, random_state=42)
            
            # Load actual Arena-Hard-Auto data from base_processed directory
            base_processed_dir = "/Users/benjaminfeuer/Library/CloudStorage/GoogleDrive-penfever@gmail.com/My Drive/Current Papers/bias-bounded-evaluation/sos-addl-data/InDepthAnalysis"
            
            # Build query contexts for batching and/or harvest precomputed scores
            query_contexts: List[Dict[str, Any]] = []
            precomputed_scores: List[Optional[float]] = []
            resolved_mask: List[bool] = []
            row_records: List[tuple] = []  # keep (question_id, model) for alignment
            for _, row in sampled_context.iterrows():
                question_id = row['question_id']
                model_name = row['model']
                try:
                    judgment_data = _load_judgment_data(base_processed_dir, question_id, model_name)
                    user_prompt = judgment_data['games'][0]['user_prompt']
                    question, answer_a, answer_b = _parse_arena_hard_prompt(user_prompt)
                    row_records.append((question_id, model_name))
                    # Try to use precomputed baseline score if available and preferred
                    score_used = None
                    if self.prefer_existing_scores:
                        try:
                            # Heuristic: search for pairwise verdict tokens in record
                            rec_str = json.dumps(judgment_data, ensure_ascii=False)
                            for token in ['A>>B', 'B>>A', 'A>B', 'B>A', 'A=B']:
                                if token in rec_str:
                                    score_used = PAIRWISE_TO_SCORE.get(token)
                                    break
                        except Exception:
                            score_used = None
                    if score_used is not None:
                        precomputed_scores.append(float(score_used))
                        resolved_mask.append(True)
                        # placeholder for alignment; no query context for this row
                        query_contexts.append(None)  # type: ignore
                    else:
                        precomputed_scores.append(None)
                        resolved_mask.append(False)
                        query_contexts.append({
                            'question': question,
                            'answer_a': answer_a,
                            'answer_b': answer_b,
                            'model': model_name
                        })
                except Exception:
                    # Skip this row if load/parse fails
                    continue

            # If any unresolved, run batched infer only for those rows
            scores: List[float] = []
            if any(not r for r in resolved_mask):
                to_query = [qc for qc, res in zip(query_contexts, resolved_mask) if not res and qc is not None]
                responses = judge_interface.query_judge_batch(to_query) if to_query else []
                queried_scores = [r['scores']['overall_score'] for r in responses if 'scores' in r and 'overall_score' in r['scores']]
                # Merge back preserving order
                q_iter = iter(queried_scores)
                for res, pc in zip(resolved_mask, precomputed_scores):
                    if res and pc is not None:
                        scores.append(pc)
                    else:
                        try:
                            scores.append(float(next(q_iter)))
                        except StopIteration:
                            # Fallback if mismatch
                            pass
            else:
                # All precomputed
                scores = [float(pc) for pc in precomputed_scores if pc is not None]

            total_attempted = len(query_contexts)
            successful_samples = len(scores)
            if successful_samples == 0:
                raise ValueError(f"No valid judgment data found for dynamic scoring in {total_attempted} samples")

            result = np.array(scores)
            # Report precomputed usage
            try:
                if self.prefer_existing_scores:
                    used = sum(1 for pc in precomputed_scores if pc is not None)
                    print(f"📎 Used precomputed baseline scores for {used}/{successful_samples} samples")
            except Exception:
                pass
            
            # Debug: Check for suspicious score patterns that could indicate fallback issues
            if len(scores) > 5:  # Only analyze if we have enough samples
                unique_scores, counts = np.unique(result, return_counts=True)
                fallback_dominant = any(count / len(result) > 0.8 for count in counts)  # >80% same score
                if fallback_dominant:
                    print(f"⚠️  Warning: Score distribution may indicate fallback dominance: {dict(zip(unique_scores, counts))}")
            
            return result
            
        elif isinstance(context, dict):
            # For single context dict (usually used in testing)
            response = judge_interface.query_judge(context)
            score = response['scores']['overall_score']
            result = np.array([score])
            return result
            
        elif isinstance(context, list):
            # Handle list of contexts (e.g., formatting neighbors)
            # Batched infer over list of contexts
            responses = judge_interface.query_judge_batch(context)
            scores = [r['scores']['overall_score'] for r in responses if 'scores' in r and 'overall_score' in r['scores']]
            if not scores:
                raise ValueError("No valid contexts could be evaluated")
            result = np.array(scores)
            return result
            
        else:
            raise ValueError(f"Unsupported context type: {type(context)}")
    
    # Attach interface for cost tracking
    judge_function.interface = judge_interface
    return judge_function


def create_judge_configs():
    """Create Oumi configs for all judges we want to test."""
    configs_dir = Path(__file__).parent / "judge_configs"
    configs_dir.mkdir(exist_ok=True)
    
    configs = {
        # API-based judges
        "gpt-3.5-turbo": {
            "model": {
                "model_name": "gpt-3.5-turbo-0125"
            },
            "engine": "OPENAI",
            "generation": {
                "max_new_tokens": 2048,
                "temperature": 0.0
            }
        },
        "gpt-4o-mini": {
            "model": {
                "model_name": "gpt-4o-mini-2024-07-18"
            },
            "engine": "OPENAI", 
            "generation": {
                "max_new_tokens": 2048,
                "temperature": 0.0
            }
        },
        "claude-3-5-sonnet": {
            "model": {
                "model_name": "claude-3-5-sonnet-latest"
            },
            "engine": "ANTHROPIC",
            "generation": {
                "max_new_tokens": 2048,
                "temperature": 0.0
            }
        },
        
        # Local GGUF models
        "deepseek-r1-32b-gguf": {
            "model": {
                "model_name": "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
                "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "model_max_length": 8192,
                "torch_dtype_str": "float16",
                "trust_remote_code": True,
                "model_kwargs": {
                    "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
                }
            },
            "engine": "LLAMACPP",
            "generation": {
                "max_new_tokens": 2048,
                "temperature": 0.0,
                "top_p": 0.9
            }
        },
        
        "qwq-32b-gguf": {
            "model": {
                "model_name": "unsloth/QwQ-32B-GGUF",
                "tokenizer_name": "Qwen/QwQ-32B-Preview",
                "model_max_length": 8192,
                "torch_dtype_str": "float16", 
                "trust_remote_code": True,
                "model_kwargs": {
                    "filename": "QwQ-32B-UD-Q4_K_XL.gguf"
                }
            },
            "engine": "LLAMACPP",
            "generation": {
                "max_new_tokens": 2048,
                "temperature": 0.0,
                "top_p": 0.9
            }
        }
    }
    
    created_configs = {}
    for name, config in configs.items():
        config_path = configs_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        created_configs[name] = str(config_path)
        print(f"✅ Created config: {config_path}")
    
    return created_configs


if __name__ == "__main__":
    print("🔧 Creating Oumi Judge Configs...")
    configs = create_judge_configs()
    
    print(f"\n📋 Available Judge Configs:")
    for name, path in configs.items():
        print(f"  • {name}: {path}")
    
    print(f"\n✅ Oumi judge interface ready!")
