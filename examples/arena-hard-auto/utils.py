import os
import json
import time
import yaml
import random
import requests

from typing import Optional
from glob import glob

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


OPENAI_MODEL_LIST = (
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0613-verbose",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
)


temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}


def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    return questions


def load_model_answers(answer_dir: str):
    """Load model answers.

    The return value is a python dict of type:
    Dict[model_name: str -> Dict[question_id: int -> answer: dict]]
    """
    filenames = glob(os.path.join(answer_dir, "*.jsonl"))
    filenames.sort()
    model_answers = {}

    for filename in filenames:
        model_name = os.path.basename(filename)[:-6]
        answer = {}
        with open(filename) as fin:
            for line in fin:
                line = json.loads(line)
                answer[line["question_id"]] = line
        model_answers[model_name] = answer

    return model_answers


def get_endpoint(endpoint_list):
    if endpoint_list is None:
        return None
    assert endpoint_list is not None
    # randomly pick one
    api_dict = random.choices(
        endpoint_list
    )[0]
    return api_dict


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs

import re

def remove_special_tokens(messages):
    # List of special tokens to remove
    special_tokens = [
        r'<\|im_start\|>', r'<\|im_end\|>',  # ChatML tokens
        r'<s>', r'</s>', r'<\|eot_id\|>',     # LLaMA tokens
        r'<\|endoftext\|>',                  # GPT special token
    ]

    new_messages = []
    
    # Combine all special tokens into a single regex pattern
    pattern = '|'.join(special_tokens)
    
    for m in messages:


        # Remove all instances of special tokens from the prompt
        m['content'] = re.sub(pattern, '', m['content'])
    
        # Remove any extra whitespace that might have been left behind
        m['content'] = ' '.join(m['content'].split())
        new_messages.append(m)
    
    return new_messages

def remove_duplicate_char_ngrams(messages, n):
    new_messages = []
    for m in messages:
        # Generate character n-grams
        text = m['content']
        ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
        
        # Keep track of seen n-grams
        seen = set()
        result = []
        
        for i, ngram in enumerate(ngrams):
            if ngram not in seen:
                # This is a new n-gram, add it to the result
                result.append(text[i])
                seen.add(ngram)
            else:
                # This is a duplicate n-gram, skip it
                pass
        
        # Add any remaining characters
        result.extend(text[len(text)-n+1:])
        
        # Join the characters back into a string
        s = ''.join(result)
        m['content'] = s
        new_messages.append(m)
    return new_messages

def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None, return_logprobs=False):
    import openai
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict.get("api_key", ""),
        )
    else:
        client = openai.OpenAI()
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # Add logprobs parameter if return_logprobs is True
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": 60,
            }
            
            if return_logprobs:
                kwargs["logprobs"] = True
                
            completion = client.chat.completions.create(**kwargs)
            
            if return_logprobs:
                # Return both content and logprobs
                output = {
                    "content": completion.choices[0].message.content,
                    "logprobs": completion.choices[0].logprobs
                }
            else:
                # Return just the content as before
                output = completion.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e, dir(e))
            #NOTE: this has the potential to produce some pretty weird strings; should be careful here
            # if e.error['code'] == 'invalid_prompt':
            messages = remove_special_tokens(messages)
            messages = remove_duplicate_char_ngrams(messages, 50)

        except KeyError:
            print(type(e), e)
            break
    
    return output


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    import openai
    from openai import AzureOpenAI

    api_base = api_dict["api_base"]
    client = AzureOpenAI(
        azure_endpoint = api_base,
        api_key= api_dict["api_key"],
        api_version=api_dict["api_version"],
        timeout=240,
        max_retries=2
    )

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=42,
            )
            output = response.choices[0].message.content
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(type(e), e)
            break

    return output


def chat_completion_anthropic(model, messages, temperature, max_tokens, api_dict=None):
    import anthropic

    if api_dict:
        api_key = api_dict["api_key"]
    else:
        api_key = os.environ["ANTHROPIC_API_KEY"]

    sys_msg = ""
    if messages[0]["role"] == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            c = anthropic.Anthropic(api_key=api_key)
            response = c.messages.create(
                model=model,
                messages=messages,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg
            )
            output = response.content[0].text
            break
        except anthropic.APIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return output

def chat_completion_huggingface(model, conv, temperature, max_tokens, return_logprobs=False):
    """
    Makes a request to HuggingFace's API for text generation.
    
    Args:
        model: The HuggingFace model ID to use
        conv: The conversation/prompt to send to the model (list of message dicts)
        temperature: Temperature setting for generation
        max_tokens: Maximum number of tokens to generate
        return_logprobs: Whether to return logprobs alongside the generated text
        
    Returns:
        If return_logprobs is False, returns the generated text.
        If return_logprobs is True, returns a dict with 'content' and 'logprobs' keys.
    """
    API_BASE = "https://api-inference.huggingface.co/models/"
    API_URL = API_BASE + model
    headers = {"Authorization": "Bearer " + str(os.environ["HUGGINGFACE_API_KEY"])}
    
    # Convert the conversation format to a single text prompt that HF models expect
    prompt = ""
    
    # Format the conversation into a text prompt
    for message in conv:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    
    # Add the final assistant prefix to prompt the model to generate
    prompt += "<|assistant|>\n"
    
    # Use different payload configurations based on the model
    if "llama" in model.lower():
        # Llama-specific parameters
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "details": True,  # Llama models may need this for logprobs
            }
        }
    else:
        # Generic payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            }
        }
    
    if return_logprobs:
        # Add all possible parameters for returning logprobs
        # Different models might use different parameter names
        payload["parameters"]["return_scores"] = True
        payload["parameters"]["return_token_scores"] = True
        payload["parameters"]["return_tokens"] = True
        payload["parameters"]["return_details"] = True
        payload["parameters"]["details"] = True
        payload["parameters"]["output_scores"] = True
    
    # print(f"DEBUG - Sending request to HuggingFace API for model: {model}")
    # print(f"DEBUG - Payload parameters: {payload['parameters']}")
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        result = response.json()
        # Check the structure of the response
        if isinstance(result[0], dict):                
            # Try different possible keys for scores
            score_key = None
            token_key = None
            
            # Check for "details" key used by Llama and some other models
            if "details" in result[0]:
                #print(f"DEBUG - Found details key with type: {type(result[0]['details'])}")
                details = result[0]["details"]
                
                if isinstance(details, dict):
                    #print(f"DEBUG - Details keys: {details.keys()}")
                    
                    # Look for token logprobs in details
                    if "logprobs" in details:
                        # print("DEBUG - Found logprobs in details")
                        
                        logprobs_data = details["logprobs"]
                        if isinstance(logprobs_data, dict):
                            #print(f"DEBUG - Logprobs keys: {logprobs_data.keys()}")
                            
                            if "tokens" in logprobs_data and "token_logprobs" in logprobs_data:
                                # print("DEBUG - Found tokens and token_logprobs in logprobs")
                                token_texts = logprobs_data["tokens"]
                                token_scores = logprobs_data["token_logprobs"]
                                
                                #print(f"DEBUG - Found {len(token_scores)} token scores and {len(token_texts)} token texts")
                                
                                if len(token_scores) > 0 and len(token_texts) > 0:
                                    logprobs = {
                                        "content": [
                                            {"text": token, "logprob": score}
                                            for token, score in zip(token_texts, token_scores)
                                        ]
                                    }
                                    
                                    return {
                                        "content": result[0].get("generated_text", ""),
                                        "logprobs": logprobs
                                    }
                    
                    # Extract logprobs from tokens field (present in Llama-3.1 responses)
                    if "tokens" in details:
                        # print("DEBUG - Found tokens field in details, attempting to extract logprobs")
                        tokens_data = details["tokens"]
                        
                        if isinstance(tokens_data, list) and len(tokens_data) > 0:
                            #print(f"DEBUG - Found {len(tokens_data)} tokens")
                            
                            # Print sample token to understand the structure
                            if len(tokens_data) > 0:
                                #print(f"DEBUG - Token structure example: {str(tokens_data[0])}")
                            
                            # Try to extract id and logprob
                                try:
                                    # Adapt to different possible token structures
                                    token_content = []
                                    for t in tokens_data:
                                        if isinstance(t, dict):
                                            # Extract text/id and logprob from token dict
                                            text = t.get("text", str(t.get("id", "")))
                                            logprob = t.get("logprob", 0)
                                            token_content.append({"text": text, "logprob": logprob})
                                        else:
                                            # If token isn't a dict, use string representation with default logprob
                                            token_content.append({"text": str(t), "logprob": 0})
                                    
                                    if token_content:
                                        #print(f"DEBUG - Successfully extracted {len(token_content)} tokens with logprobs")
                                        logprobs = {"content": token_content}
                                        return {
                                            "content": result[0].get("generated_text", ""),
                                            "logprobs": logprobs
                                        }
                                except Exception as e:
                                    print(f"DEBUG - Error extracting token information: {e}")
                
                # Standard response format
                if "scores" in result[0]:
                    score_key = "scores"
                elif "token_scores" in result[0]:
                    score_key = "token_scores"
                elif "log_probs" in result[0]:
                    score_key = "log_probs"
                
                if "tokens" in result[0]:
                    token_key = "tokens"
                elif "token_texts" in result[0]:
                    token_key = "token_texts"
                
                content = result[0].get("generated_text", "")
                
                if return_logprobs and score_key and token_key and content:
                    # Extract the logprobs from the response
                    token_scores = result[0].get(score_key, [])
                    token_texts = result[0].get(token_key, [])
                    
                    #print(f"DEBUG - Found {score_key}: {len(token_scores)}")
                    #print(f"DEBUG - Found {token_key}: {len(token_texts)}")
                    
                    if len(token_scores) > 0 and len(token_texts) > 0:
                        # Create a logprobs object similar to OpenAI's format
                        logprobs = {
                            "content": [
                                {"text": token, "logprob": score}
                                for token, score in zip(token_texts, token_scores)
                            ]
                        }
                        
                        return {
                            "content": content,
                            "logprobs": logprobs
                        }
                    else:
                        # print("DEBUG - No token scores or texts found, returning without logprobs")
                        return content
                else:
                    # print("DEBUG - Missing required keys for logprobs, returning without logprobs")
                    return content
            else:
                # print("DEBUG - First result is not a dictionary")
                # Return just the generated text
                return result[0].get("generated_text", "")
        else:
            #print(f"DEBUG - Unexpected response format from HuggingFace API: {result}")
            return API_ERROR_OUTPUT
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return API_ERROR_OUTPUT
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return API_ERROR_OUTPUT
    except ValueError as json_err:
        print(f"JSON parsing error: {json_err}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
        return API_ERROR_OUTPUT
    except Exception as e:
        print(f"Error in HuggingFace API call: {e}")
        print(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return API_ERROR_OUTPUT

# Global model and tokenizer cache
_HF_MODEL_CACHE = {}
_HF_TOKENIZER_CACHE = {}

def chat_completion_huggingface_local(model, conv, temperature, max_tokens, return_logprobs=False):
    """
    Loads and runs a HuggingFace model locally for text generation with support for CUDA, ROCm, and Metal.
    Models are cached in memory to avoid reloading them for subsequent calls.
    
    Args:
        model: The HuggingFace model ID to load locally
        conv: The conversation/prompt to send to the model (list of message dicts)
        temperature: Temperature setting for generation
        max_tokens: Maximum number of tokens to generate
        return_logprobs: Whether to return logprobs alongside the generated text
        
    Returns:
        If return_logprobs is False, returns the generated text.
        If return_logprobs is True, returns a dict with 'content' and 'logprobs' keys.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
        from threading import Thread
    except ImportError:
        print("ERROR: Required libraries not installed. Please run: pip install torch transformers")
        return API_ERROR_OUTPUT
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # Detect available computing resources
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                print("Using Apple Metal (MPS) device")
            elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
                device = "cuda"  # ROCm uses the CUDA interface
                print("Using AMD ROCm device")
            else:
                device = "cpu"
                print("WARNING: No GPU detected. Using CPU which may be slow.")
            
            # Create a unique key for caching the model based on model ID and configuration
            cache_key = model
            if os.environ.get("USE_4BIT_QUANTIZATION", "").lower() == "true":
                cache_key += "_4bit"
            cache_key += f"_{device}"

            # Check tokenizer cache first 
            if cache_key in _HF_TOKENIZER_CACHE:
                print(f"Using cached tokenizer for {model}")
                tokenizer = _HF_TOKENIZER_CACHE[cache_key]
            else:
                print(f"Loading tokenizer for {model}...")
                tokenizer = AutoTokenizer.from_pretrained(model)
                _HF_TOKENIZER_CACHE[cache_key] = tokenizer
            
            # Now check model cache
            if cache_key in _HF_MODEL_CACHE:
                print(f"Using cached model for {model}")
                model_instance = _HF_MODEL_CACHE[cache_key]
            else:
                # Model not in cache, need to load it
                print(f"Loading model {model} (not found in cache)...")
                
                # Set up appropriate model loading parameters
                model_kwargs = {
                    "device_map": device,
                    "torch_dtype": torch.float16,  # Use fp16 for efficiency
                }
                
                # Handle CPU-only mode
                if device == "cpu":
                    model_kwargs = {
                        "low_cpu_mem_usage": True,
                        "torch_dtype": torch.float16,  # Use half precision even on CPU for memory efficiency
                    }
                
                # Special handling for MPS (Apple Silicon)
                if device == "mps":
                    # For models on MPS, enable lower precision
                    model_kwargs["torch_dtype"] = torch.float16
                
                # Add 4-bit quantization for memory-constrained environments
                if os.environ.get("USE_4BIT_QUANTIZATION", "").lower() == "true":
                    try:
                        from transformers import BitsAndBytesConfig
                        model_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        print("Using 4-bit quantization")
                    except ImportError:
                        print("Warning: bitsandbytes not available for 4-bit quantization")
                
                try:
                    # Create offload folder if needed
                    if "offload_folder" in model_kwargs and not os.path.exists(model_kwargs["offload_folder"]):
                        os.makedirs(model_kwargs["offload_folder"], exist_ok=True)
                    
                    # Load the model with specified configuration
                    print(f"Loading model with settings: device={device}, dtype={model_kwargs.get('torch_dtype', 'default')}")
                    model_instance = AutoModelForCausalLM.from_pretrained(
                        model,
                        **model_kwargs
                    )
                    
                    # Cache the loaded model
                    _HF_MODEL_CACHE[cache_key] = model_instance
                    print(f"Model {model} successfully loaded and cached (cache key: {cache_key})")
                    
                except (RuntimeError, ValueError, OSError) as e:
                    if "Invalid buffer size" in str(e) or "CUDA out of memory" in str(e) or "MPS backend out of memory" in str(e):
                        print(f"ERROR: Not enough memory to load model {model}. Trying CPU with minimal memory settings...")
                        # Fall back to CPU with minimal memory settings
                        device = "cpu"
                        os.environ["USE_4BIT_QUANTIZATION"] = "true"  # Force quantization
                        
                        # Make offload directory if it doesn't exist
                        if not os.path.exists("offload_folder"):
                            os.makedirs("offload_folder", exist_ok=True)
                        
                        # Update cache key for fallback configuration
                        cache_key = f"{model}_4bit_cpu_fallback"
                        
                        model_kwargs = {
                            "device_map": "auto",
                            "low_cpu_mem_usage": True,
                            "offload_folder": "offload_folder",
                            "offload_state_dict": True,
                            "torch_dtype": torch.float16,
                        }
                        
                        # Try with 4-bit quantization as last resort
                        try:
                            from transformers import BitsAndBytesConfig
                            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            print("Using 4-bit quantization with CPU fallback")
                        except ImportError:
                            print("Warning: bitsandbytes not available for 4-bit quantization")
                        
                        try:
                            model_instance = AutoModelForCausalLM.from_pretrained(
                                model,
                                **model_kwargs
                            )
                            
                            # Cache the fallback model
                            _HF_MODEL_CACHE[cache_key] = model_instance
                            print(f"Fallback model {model} successfully loaded and cached (cache key: {cache_key})")
                            
                        except Exception as e2:
                            print(f"ERROR: Failed to load model even with minimal memory settings: {e2}")
                            raise RuntimeError(f"Could not load model {model} with any configuration: {e2}")
                    else:
                        raise  # Re-raise the original error if it's not memory-related
            
            # Convert the conversation format to a single text prompt
            prompt = ""
            
            # Format the conversation into a text prompt
            for message in conv:
                role = message.get("role", "")
                content = message.get("content", "")
                
                # Handle different chat templates (adjust as needed for specific models)
                if tokenizer.chat_template:
                    # If the model has a chat template, use it
                    # This is the most robust approach
                    messages_for_template = [{"role": m["role"], "content": m["content"]} for m in conv]
                    prompt = tokenizer.apply_chat_template(messages_for_template, tokenize=False)
                    break  # Skip the manual formatting below
                    
                elif "llama" in model.lower() or "mistral" in model.lower():
                    # Llama/Mistral style formatting
                    if role == "system":
                        prompt += f"<|system|>\n{content}\n"
                    elif role == "user":
                        prompt += f"<|user|>\n{content}\n"
                    elif role == "assistant":
                        prompt += f"<|assistant|>\n{content}\n"
                
                elif "falcon" in model.lower():
                    # Falcon style formatting
                    if role == "system":
                        prompt += f"System: {content}\n"
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
                
                else:
                    # Generic formatting
                    if role == "system":
                        prompt += f"System: {content}\n"
                    elif role == "user":
                        prompt += f"User: {content}\n"
                    elif role == "assistant":
                        prompt += f"Assistant: {content}\n"
            
            # Add the final assistant prefix to prompt the model to generate
            if not tokenizer.chat_template:
                if "llama" in model.lower() or "mistral" in model.lower():
                    prompt += "<|assistant|>\n"
                else:
                    prompt += "Assistant: "
            
            # Tokenize the input
            model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Setup generation parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.0,
                "top_p": 0.95,
                "repetition_penalty": 1.1,
            }
            
            # Add logprobs parameters if requested
            if return_logprobs:
                gen_kwargs["output_scores"] = True
                gen_kwargs["return_dict_in_generate"] = True
            
            # Generate text with or without logprobs
            if return_logprobs:
                generation_output = model_instance.generate(
                    **model_inputs,
                    **gen_kwargs
                )
                
                # Extract the generated tokens
                generated_tokens = generation_output.sequences[0][model_inputs.input_ids.shape[1]:]
                
                # Get the scores (log probabilities)
                scores = generation_output.scores
                
                # Convert scores to token logprobs
                token_logprobs = []
                for i, token_scores in enumerate(scores):
                    # Get the score for the selected token (highest probability)
                    token_id = generated_tokens[i].item()
                    token_score = token_scores[0, token_id].item()  # Use log probabilities
                    token_logprobs.append(token_score)
                
                # Convert tokens to text
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Create token text list
                token_texts = tokenizer.convert_ids_to_tokens(generated_tokens)
                
                # Build the result in the required format
                logprobs = {
                    "content": [
                        {"text": text, "logprob": logprob}
                        for text, logprob in zip(token_texts, token_logprobs)
                    ]
                }
                
                output = {
                    "content": generated_text,
                    "logprobs": logprobs
                }
            else:
                # Without logprobs, just generate the text
                generated_tokens = model_instance.generate(
                    **model_inputs,
                    **gen_kwargs
                )
                generated_text = tokenizer.decode(generated_tokens[0][model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
                output = generated_text
            
            break
        except Exception as e:
            print(f"Error in local model inference: {type(e)}: {e}")
            time.sleep(API_RETRY_SLEEP)
    
    # We no longer clean up the model as we want to keep it cached
    return output

def chat_completion_mistral(model, messages, temperature, max_tokens):
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralException

    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)

    prompts = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            chat_response = client.chat(
                model=model,
                messages=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = chat_response.choices[0].message.content
            break
        except MistralException as e:
            print(type(e), e)
            break

    return output


def http_completion_gemini(model, message, temperature, max_tokens):
    api_key = os.environ["GEMINI_API_KEY"]
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"
        },
    ]

    output = API_ERROR_OUTPUT
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
            json={
                "contents": [{
                    "parts":[
                        {"text": message}
                    ]
                }],
                "safetySettings": safety_settings,
                "generationConfig":{
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                }
            },
        )
    except Exception as e:
        print(f"**API REQUEST ERROR** Reason: {e}.")

    if response.status_code != 200:
        print(f"**API REQUEST ERROR** Reason: status code {response.status_code}.")

    output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return output
    


def chat_completion_cohere(model, messages, temperature, max_tokens):
    import cohere

    co = cohere.Client(os.environ["COHERE_API_KEY"])
    assert len(messages) > 0

    template_map = {"system":"SYSTEM",
                    "assistant":"CHATBOT",
                    "user":"USER"}

    assert messages[-1]["role"] == "user"
    prompt = messages[-1]["content"]

    if len(messages) > 1:
        history = []
        for message in messages[:-1]:
            history.append({"role":template_map[message["role"]], "message":message["content"]})
    else:
        history = None

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = co.chat(
                message=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                chat_history=history,
            )
            output = response.text
            break
        except cohere.core.api_error.ApiError as e:
            print(type(e), e)
            raise
        except Exception as e:
            print(type(e), e)
            break
    
    return output


def chat_completion_together(model, messages, temperature, max_tokens, api_dict=None, return_logprobs=False):
    """
    Makes a request to Together AI's API for text generation.
    
    Args:
        model: The Together AI model ID to use
        messages: The conversation/prompt to send to the model (list of message dicts)
        temperature: Temperature setting for generation
        max_tokens: Maximum number of tokens to generate
        api_dict: Dictionary containing API configuration (api_base)
        return_logprobs: Whether to return logprobs alongside the generated text
        
    Returns:
        If return_logprobs is False, returns the generated text.
        If return_logprobs is True, returns a dict with 'content' and 'logprobs' keys.
    """
    import requests

    # Always get API key from environment variable
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable not set")
        return API_ERROR_OUTPUT
    
    # Get API base URL from config if provided, otherwise use default
    if api_dict:
        api_base = api_dict.get("api_base", "https://api.together.xyz/v1")
    else:
        api_base = os.environ.get("TOGETHER_API_BASE", "https://api.together.xyz/v1")
    
    # Ensure the API base ends with /v1
    if not api_base.endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Prepare the payload for the API request
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Add logprobs if requested
    if return_logprobs:
        # Together API uses integer value 1 for logprobs
        payload["logprobs"] = 1
    
    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            # Send the request to the Together API
            response = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Print the response structure for debugging
            if return_logprobs:
                # print("DEBUG - Together API logprobs response structure:")
                if "choices" in result and len(result["choices"]) > 0 and "logprobs" in result["choices"][0]:
                    logprobs_data = result["choices"][0]["logprobs"]
                    # print(f"Logprobs keys: {logprobs_data.keys()}")
            
            # Extract the generated text
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                
                # Handle logprobs if requested and available
                if return_logprobs and "logprobs" in result["choices"][0]:
                    logprobs_data = result["choices"][0]["logprobs"]
                    
                    # Format logprobs similar to OpenAI format for compatibility
                    tokens_with_logprobs = []
                    
                    # Together API format has parallel arrays for tokens and logprobs
                    if ("tokens" in logprobs_data and 
                        "token_logprobs" in logprobs_data and 
                        len(logprobs_data["tokens"]) == len(logprobs_data["token_logprobs"])):
                        
                        tokens = logprobs_data["tokens"]
                        token_logprobs = logprobs_data["token_logprobs"]
                        
                        for token_text, token_logprob in zip(tokens, token_logprobs):
                            tokens_with_logprobs.append({
                                "text": token_text,
                                "logprob": token_logprob
                            })
                    
                    # Return both content and logprobs
                    output = {
                        "content": content,
                        "logprobs": {"content": tokens_with_logprobs}
                    }
                else:
                    # Return just the content
                    output = content
                
                break
            else:
                print(f"Unexpected response format from Together API: {result}")
                break
                
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            print(f"Response: {response.text if 'response' in locals() else 'No response'}")
            time.sleep(API_RETRY_SLEEP)
        except requests.exceptions.RequestException as req_err:
            print(f"Request error occurred: {req_err}")
            time.sleep(API_RETRY_SLEEP)
        except ValueError as json_err:
            print(f"JSON parsing error: {json_err}")
            print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
            break
        except Exception as e:
            print(f"Error in Together API call: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'No response'}")
            break
    
    return output


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

import json
import re

def write_with_subscores(input_file, output_file, count_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile, open(count_file, 'w') as countfile:
        
        success_counts = {
            'correctness_score': 0,
            'completeness_score': 0,
            'safety_score': 0,
            'conciseness_score': 0,
            'style_score': 0,
        }
        
        for line in infile:
            data = json.loads(line)
            
            for game in data['games']:
                judgment = game['judgment']
                
                # Define regex patterns
                patterns = {
                    'correctness_score': r'Correctness: \(\(([AB<>=]+)\)\)',
                    'completeness_score': r'Completeness: \(\(([AB<>=]+)\)\)',
                    'safety_score': r'Safety: \(\(([AB<>=]+)\)\)',
                    'conciseness_score': r'Conciseness: \(\(([AB<>=]+)\)\)',
                    'style_score': r'Style: \(\(([AB<>=]+)\)\)',
                }                

                for key, pattern in patterns.items():
                    match = re.search(pattern, judgment)
                    if match:
                        success_counts[key] += 1
                    game[key] = match.group(1) if match else ''
                
                # Reorder the dictionary to insert new keys after 'score'
                keys = list(game.keys())
                score_index = keys.index('score')
                new_keys = keys[:score_index+1] + list(patterns.keys()) + keys[score_index+1:]
                new_keys = [k for k in new_keys if k not in patterns]
                
                game = {k: game[k] for k in new_keys}

            # Write the modified data to the output file
            json.dump(data, outfile)
            outfile.write('\n')
        json.dump(success_counts, countfile)
        countfile.write('\n')