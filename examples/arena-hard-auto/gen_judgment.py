import json
import yaml
import argparse
import os
import re
import concurrent.futures

from tqdm import tqdm

from utils import (
    load_questions,
    chat_completion_openai,
    chat_completion_openai_azure,
    chat_completion_anthropic,
    chat_completion_huggingface,
    chat_completion_huggingface_local,
    chat_completion_together,
    load_questions,
    load_model_answers,
    get_endpoint,
    make_config,
)


def get_score(judgment, patterns, pairwise=True):
    """
    Extract scores from the judgment using multiple regex patterns.
    Finds the last match in the string rather than the first.
    
    Args:
        judgment: The text of the judgment
        patterns: List of dicts, each with 'name' and 'pattern' keys
        pairwise: Whether we're doing pairwise comparison
        
    Returns:
        A tuple of (scores_dict, continue_flag) where:
        - scores_dict contains pattern names as keys and matches as values
        - continue_flag indicates whether to continue requesting more tokens
    """
    scores = {}
    continue_flag = False
    
    # If patterns is a single pattern object (for backward compatibility)
    if not isinstance(patterns, list):
        # Legacy mode - single pattern
        matches = patterns.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None, True
        elif len(matches) >= 1:
            # Get the last match instead of the first
            last_match = matches[-1].strip("\n")
            if pairwise:
                return last_match, False
            return int(last_match), False
        else:
            return None, False
    
    # Process each pattern
    for pattern_obj in patterns:
        pattern_name = pattern_obj['name']
        pattern = pattern_obj['pattern']
        
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) == 0:
            # No matches for this pattern, continue requesting more tokens
            continue_flag = True
        elif len(matches) >= 1:
            # Get the last match instead of checking for uniqueness
            last_match = matches[-1].strip("\n")
            scores[pattern_name] = last_match if pairwise else int(last_match)
        else:
            # No valid matches for this pattern
            scores[pattern_name] = None
    
    # If no patterns matched anything, return None
    if not scores:
        return None, continue_flag
    
    # If we're in legacy single-pattern mode and only got one result, return it directly
    if len(patterns) == 1 and len(scores) == 1:
        pattern_name = patterns[0]['name']
        if pattern_name in scores:
            return scores[pattern_name], continue_flag
    
    return scores, continue_flag


def get_score_logprobs(judgment, patterns, logprobs, pairwise=True):
    """
    Similar to get_score but extracts and calculates the average logprob for each matched token
    for multiple patterns. Finds the last match in the string rather than the first.
    
    Args:
        judgment: The text of the judgment
        patterns: List of dicts, each with 'name' and 'pattern' keys
        logprobs: Logprob data from the API (can be object or dict format)
        pairwise: Whether we're doing pairwise comparison
        
    Returns:
        A tuple of (scores_dict, continue_flag) where:
        - scores_dict contains pattern names as keys and dicts with match and logprob as values
        - continue_flag indicates whether to continue requesting more tokens
    """
    scores = {}
    continue_flag = False
    
    # Check that we have valid logprobs
    has_valid_logprobs = (
        (hasattr(logprobs, "content") and isinstance(logprobs.content, list)) or
        (isinstance(logprobs, dict) and "content" in logprobs and isinstance(logprobs["content"], list))
    )
    
    if not has_valid_logprobs:
        print(f"WARNING: Invalid logprobs format: {type(logprobs)}")
        if isinstance(logprobs, dict):
            print(f"Keys: {logprobs.keys()}")
    
    # If patterns is a single pattern object (for backward compatibility)
    if not isinstance(patterns, list):
        # Legacy mode - single pattern
        matches = patterns.findall(judgment)
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) == 0:
            return None, True
        elif len(matches) >= 1:
            # Get the last match instead of the first
            match = matches[-1].strip("\n")
            
            # Calculate average logprob for the matched tokens
            avg_logprob = None
            
            # Process if we have valid logprobs data
            if has_valid_logprobs:
                token_logprobs = calculate_token_logprobs(judgment, match, logprobs)
                if token_logprobs:
                    avg_logprob = sum(token_logprobs) / len(token_logprobs)
            
            # Return score with logprob
            if pairwise:
                return {
                    "match": match,
                    "avg_logprob": avg_logprob
                }, False
            return int(match), False
        else:
            return None, False
    
    # Process each pattern
    for pattern_obj in patterns:
        pattern_name = pattern_obj['name']
        pattern = pattern_obj['pattern']
        
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        
        if len(set(matches)) == 0:
            # No matches for this pattern, continue requesting more tokens
            continue_flag = True
        elif len(matches) >= 1:
            # Get the last match instead of checking for uniqueness
            match = matches[-1].strip("\n")
            
            # Calculate average logprob for the matched tokens
            avg_logprob = None            
            
            # Process if we have valid logprobs data
            if has_valid_logprobs:
                token_logprobs = calculate_token_logprobs(judgment, match, logprobs)
                if token_logprobs:
                    avg_logprob = sum(token_logprobs) / len(token_logprobs)
                    # print(f"DEBUG - Pattern '{pattern_name}' match '{match}' logprob: {avg_logprob}")

            scores[pattern_name] = {
                "match": match if pairwise else int(match),
                "avg_logprob": avg_logprob
            }
        else:
            # No valid matches for this pattern
            scores[pattern_name] = None
    
    # If no patterns matched anything, return None
    if not scores:
        return None, continue_flag
    
    # If we're in legacy single-pattern mode and only got one result, return it directly
    if len(patterns) == 1 and len(scores) == 1:
        pattern_name = patterns[0]['name']
        if pattern_name in scores:
            return scores[pattern_name], continue_flag
    
    return scores, continue_flag


def calculate_token_logprobs(judgment, match, logprobs):
    """Helper function to calculate token logprobs for a match"""
    token_logprobs = []
    
    # Find the match in the full text
    match_start = judgment.find(match)
    if match_start < 0:
        return []
        
    match_end = match_start + len(match)
    
    # Go through tokens and find those within our match
    token_offset = 0
    
    # Support both object-style (OpenAI) and dict-style (Together, HF) formats
    token_list = []
    if hasattr(logprobs, "content") and isinstance(logprobs.content, list):
        # OpenAI-style object with attribute
        token_list = logprobs.content
    elif isinstance(logprobs, dict) and "content" in logprobs and isinstance(logprobs["content"], list):
        # Dictionary-style (Together, HF)
        token_list = logprobs["content"]
    else:
        print(f"WARNING: Unrecognized logprobs format: {type(logprobs)}")
        return []
    
    for i, token in enumerate(token_list):
        # Extract text and logprob from token based on format
        token_text = None
        token_logprob_value = None
        
        # Object style: token has .text and .logprob attributes
        if hasattr(token, "text") and hasattr(token, "logprob"):
            token_text = token.text
            token_logprob_value = token.logprob
        # Dict style: token is a dict with 'text' and 'logprob' keys
        elif isinstance(token, dict) and "text" in token and "logprob" in token:
            token_text = token["text"]
            token_logprob_value = token["logprob"]
        else:
            continue  # Skip this token if format is unknown
        
        token_length = len(token_text)
        
        # Check if this token is within our match
        token_start = token_offset
        token_end = token_start + token_length
        
        # If there's overlap with our match, include this token's logprob
        if (token_start >= match_start and token_start < match_end) or \
           (token_end > match_start and token_end <= match_end) or \
           (token_start <= match_start and token_end >= match_end):
            token_logprobs.append(token_logprob_value)
        
        token_offset += token_length
    
    return token_logprobs


# get answer from model
def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None, return_logprobs=False):
    api_dict = get_endpoint(endpoint_dict["endpoints"])

    if endpoint_dict["api_type"] == "anthropic":
        # Anthropic API doesn't support logprobs
        output = chat_completion_anthropic(model, conv, temperature, max_tokens)
    elif endpoint_dict["api_type"] == "azure":
        # Azure OpenAI API doesn't support logprobs
        output = chat_completion_openai_azure(model, conv, temperature, max_tokens, api_dict)
    elif endpoint_dict["api_type"] == "huggingface":
        # HuggingFace API can now support logprobs
        output = chat_completion_huggingface(model, conv, temperature, max_tokens, return_logprobs)
    elif endpoint_dict["api_type"] == "huggingface_local":
        # Local HuggingFace model with logprobs support
        output = chat_completion_huggingface_local(model, conv, temperature, max_tokens, return_logprobs)
    elif endpoint_dict["api_type"] == "together":
        # Together AI API supports logprobs
        output = chat_completion_together(model, conv, temperature, max_tokens, api_dict, return_logprobs)
    else:
        # OpenAI API supports logprobs
        output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict, return_logprobs)
    
    return output


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]
    return_logprobs = args.get("return_logprobs", False)

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id": question["question_id"],
        "model": answer["model_id"],
        "judge": model,
        "games": []
    }
    
    # Store logprobs if requested
    if return_logprobs:
        output["logprobs"] = []

    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i+1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1: # swap position
                    answer, baseline = baseline, answer

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i+base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]
            
            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        game_logprobs = []
        for _ in range(configs['number_of_judgment_attempts']):
            response = get_answer(
                endpoint_info["model_name"],
                conv,
                configs["temperature"],
                configs["max_tokens"],
                args["endpoint_dict"],
                return_logprobs
            )

            # Handle different response types based on logprobs setting
            current_logprobs = None
            if return_logprobs and isinstance(response, dict):
                new_judgment = response["content"]
                if "logprobs" in response:
                    current_logprobs = response["logprobs"]
                    game_logprobs.append(current_logprobs)
            else:
                new_judgment = response

            judgment += ("\n" + new_judgment)

            # Use different scoring function based on whether we have logprobs
            if return_logprobs and current_logprobs:
                score, try_again = get_score_logprobs(judgment, args["patterns"], current_logprobs)
            else:
                score, try_again = get_score(judgment, args["patterns"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        # Prepare the result structure based on the type of score
        result = {
            "user_prompt": conv[1]["content"],
            "judgment": judgment,
        }
        
        # Handle different score structures
        if score is None:
            # No matches found
            result["score"] = None
        elif isinstance(score, dict) and "match" in score and "avg_logprob" in score:
            # Single pattern with logprobs
            result["score"] = score["match"]
            result["score_logprob"] = score["avg_logprob"]
        elif isinstance(score, dict) and not "match" in score:
            # Multiple patterns
            result["scores"] = {}
            
            # Process each pattern's score
            for pattern_name, pattern_score in score.items():
                if pattern_name == "default" or pattern_name == "overall":
                    # Use the default/overall pattern as the main score 
                    if isinstance(pattern_score, dict) and "match" in pattern_score:
                        result["score"] = pattern_score["match"]
                        result["score_logprob"] = pattern_score["avg_logprob"]
                    else:
                        result["score"] = pattern_score
                        
                        # Calculate logprobs manually for non-dict score
                        if return_logprobs and current_logprobs:
                            # Try to find the pattern in the text with brackets: [[A>>B]]
                            score_text = f"[[{pattern_score}]]"
                            
                            # Use the shared token_logprobs calculation function
                            token_logprobs = calculate_token_logprobs(judgment, pattern_score, current_logprobs)
                            
                            # If no direct match was found, try with the full bracketed format
                            if not token_logprobs:
                                token_logprobs = calculate_token_logprobs(judgment, score_text, current_logprobs)
                            
                            # If we found matching tokens, calculate the average
                            if token_logprobs:
                                result["score_logprob"] = sum(token_logprobs) / len(token_logprobs)
                                print(f"DEBUG: Successfully extracted score_logprob for '{pattern_score}': {result['score_logprob']}")
                            else:
                                print(f"DEBUG: Could not find tokens for '{pattern_score}' in the logprobs")
                
                # Store all pattern scores in the scores dictionary
                if isinstance(pattern_score, dict) and "match" in pattern_score:
                    result["scores"][pattern_name] = {
                        "value": pattern_score["match"],
                        "logprob": pattern_score["avg_logprob"]
                    }
                else:
                    # Calculate logprobs for pattern-specific scores as well
                    if return_logprobs and current_logprobs and pattern_score:
                        # Create pattern_marker based on the pattern name
                        pattern_marker = f"(({pattern_score}))"
                        
                        # Try different approaches to find tokens
                        token_logprobs = calculate_token_logprobs(judgment, pattern_score, current_logprobs)
                        
                        # If no direct match, try with full marker
                        if not token_logprobs:
                            token_logprobs = calculate_token_logprobs(judgment, pattern_marker, current_logprobs)
                        
                        if token_logprobs:
                            result["scores"][pattern_name] = {
                                "value": pattern_score,
                                "logprob": sum(token_logprobs) / len(token_logprobs)
                            }
                        else:
                            result["scores"][pattern_name] = {
                                "value": pattern_score,
                                "logprob": None
                            }
                    else:
                        result["scores"][pattern_name] = {
                            "value": pattern_score,
                            "logprob": None
                        }
        else:
            # Simple string or int result
            result["score"] = score
        
        output["games"].append(result)
        
        # Add raw logprobs for this game if available
        # if return_logprobs and game_logprobs:
        #     output["logprobs"].append(game_logprobs)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting-file", type=str, default="config/judge_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    parser.add_argument("--logprob_judgments", action="store_true", help="Store logprobs for judge models that support it")
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)
    endpoint_list = make_config(args.endpoint_file)

    print(f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
          + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}')
    
    # Print status of logprobs feature
    if args.logprob_judgments:
        endpoint_info = endpoint_list[configs["judge_model"]]
        if endpoint_info["api_type"] not in ["anthropic", "azure"]:
            print(f"Logprobs enabled for judge model: {configs['judge_model']}")
        else:
            print(f"Warning: Logprobs requested but not supported for API type: {endpoint_info['api_type']}")

    # Handle both legacy single pattern and new multiple patterns
    patterns = []
    if "regex_patterns" in configs and configs["regex_patterns"]:
        # New multi-pattern format
        for pattern_config in configs["regex_patterns"]:
            pattern_obj = {
                "name": pattern_config["name"],
                "pattern": re.compile(pattern_config["pattern"])
            }
            patterns.append(pattern_obj)
    elif "regex_pattern" in configs and configs["regex_pattern"]:
        # Legacy single pattern format (for backward compatibility)
        pattern_obj = {
            "name": "default",
            "pattern": re.compile(configs["regex_pattern"])
        }
        patterns.append(pattern_obj)

    question_file = os.path.join("data", configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join("data", configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join("data", configs["bench_name"], "reference_answer")

    questions = load_questions(question_file)
    model_answers = load_model_answers(answer_dir)
    
    # if user choose a set of models, only judge those models
    models = [model for model in configs["model_list"]]
        
    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]
    
    output_files = {}
    if configs["baseline_model"]:
        output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}_judge/{configs['baseline_model']}_base"
    else:
        output_dir = f"data/{configs['bench_name']}/model_judgment/{configs['judge_model']}"
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    endpoint_info = endpoint_list[configs["judge_model"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=endpoint_info["parallel"]) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                question_id = question["question_id"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not question_id in model_answers[model]:
                    print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                    continue

                if model in existing_judgments and question_id in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][question_id]
                if ref_answers:
                    kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                    assert len(kwargs["reference"]) == len(configs["ref_model"])
                else:
                    kwargs["reference"] = None
                if configs["baseline"]:
                    kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
                else:
                    kwargs["baseline_answer"] = None
                kwargs["configs"] = configs
                kwargs["endpoint_dict"] = endpoint_info
                kwargs["output_file"] = output_files[model]
                kwargs["patterns"] = patterns
                
                # Add logprobs parameter if the flag is set and the API supports it
                if args.logprob_judgments and endpoint_info["api_type"] not in ["anthropic", "azure"]:
                    kwargs["return_logprobs"] = True
                    
                future = executor.submit(judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
