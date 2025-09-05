"""
Utilities for working with Arena-Hard(-Auto) data:

- Parse prompts into (question, answer_a, answer_b)
- Extract verdict tokens and map to scores
- Load per-question judgment records from base_processed directories
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Dict, Any, Iterable, Union, List
from pathlib import Path

# Canonical mapping from pairwise tokens to numeric scores
PAIRWISE_TO_SCORE: Dict[str, float] = {
    'A>>B': 1.0,
    'A>B': 2.0,
    'A=B': 3.0,
    'B>A': 4.0,
    'B>>A': 5.0,
}

SCORE_TO_PAIRWISE: Dict[float, str] = {v: k for k, v in PAIRWISE_TO_SCORE.items()}


def verdict_token_to_score(token: str) -> Optional[float]:
    token = token.strip()
    return PAIRWISE_TO_SCORE.get(token)


def extract_verdict_token(record: Dict[str, Any]) -> Optional[str]:
    """Extract a pairwise verdict token from Arena-Hard(-Auto) style records.

    Supports both top-level and nested-in-`games` schemas found in base_processed files.
    Tries these in order for each candidate container (record, then each game):
    - Direct field 'score' with a token like 'A>>B', 'A>B', 'A=B', 'B>A', 'B>>A'
    - Pattern [[A>>B]] style inside 'judgment' text
    - Loose token occurrences in 'judgment'
    """

    def _from_container(container: Dict[str, Any]) -> Optional[str]:
        score_field = container.get('score')
        if isinstance(score_field, str):
            token = score_field.strip()
            if token in PAIRWISE_TO_SCORE:
                return token

        # Numeric fallback (some files store debiased numeric only)
        num = container.get('score_debiased')
        if isinstance(num, (int, float)):
            token_from_num = SCORE_TO_PAIRWISE.get(float(num))
            if token_from_num:
                return token_from_num

        judgment = container.get('judgment')
        if isinstance(judgment, str):
            # Prefer explicit [[A>>B]] bracket form
            m = re.search(r"\[\[(A>>B|B>>A|A>B|B>A|A=B)\]\]", judgment)
            if m:
                return m.group(1)
            # Fallback: loose tokens (avoid matching inside words)
            m2 = re.search(r"\b(A>>B|B>>A|A>B|B>A|A=B)\b", judgment)
            if m2:
                return m2.group(1)
        return None

    # Check top-level first
    token = _from_container(record)
    if token:
        return token

    # Then check nested games (some files store judgments per game)
    games = record.get('games')
    if isinstance(games, Iterable):
        for g in games:
            if isinstance(g, dict):
                token = _from_container(g)
                if token:
                    return token

    return None


def parse_arena_hard_prompt(user_prompt: str) -> Tuple[str, str, str]:
    """Parse Arena-Hard-Auto prompt to extract question and two answers.

    Supports standard markers with robust regex and helpful errors.
    """
    # Normalize line endings just in case
    text = user_prompt

    # Question
    qm = re.search(r"<\|User Prompt\|>\s*\n(.*?)\n\s*<\|The Start of Assistant A\'s Answer\|>", text, re.DOTALL)
    if not qm:
        raise ValueError("Could not extract question from Arena-Hard prompt (missing User Prompt block)")
    question = qm.group(1).strip()

    # Assistant A
    am = re.search(r"<\|The Start of Assistant A\'s Answer\|>\s*\n(.*?)\n\s*<\|The End of Assistant A\'s Answer\|>", text, re.DOTALL)
    if not am:
        raise ValueError("Could not extract Assistant A's answer from Arena-Hard prompt")
    answer_a = am.group(1).strip()

    # Assistant B
    bm = re.search(r"<\|The Start of Assistant B\'s Answer\|>\s*\n(.*?)\n\s*<\|The End of Assistant B\'s Answer\|>", text, re.DOTALL)
    if not bm:
        raise ValueError("Could not extract Assistant B's answer from Arena-Hard prompt")
    answer_b = bm.group(1).strip()

    return question, answer_a, answer_b


# Precompiled regex for extracting final pairwise verdict pattern
ARENA_PAIRWISE_RE = re.compile(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]")


def extract_pairwise_from_judge_response(response: str) -> Optional[str]:
    """Extract the final pairwise verdict token from a judge response.

    Returns the last occurrence of one of the allowed tokens within [[...]] if present.
    """
    if not isinstance(response, str) or not response:
        return None
    matches = ARENA_PAIRWISE_RE.findall(response)
    return matches[-1] if matches else None


def build_pairwise_judge_prompt(question: str, answer_a: str, answer_b: str) -> str:
    """Build a complete judge prompt (system + pairwise content) for Arena-Hard style judging."""
    system_prompt = (
        "Please act as an impartial judge and evaluate the quality of the responses "
        "provided by two AI assistants to the user prompt displayed below. You will be given assistant A's "
        "answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\n"
        "Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\n"
        "When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\n"
        "Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\n"
        "Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\n"
        "After providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n"
        "1. Assistant A is significantly better: [[A>>B]]\n"
        "2. Assistant A is slightly better: [[A>B]]\n"
        "3. Tie, relatively the same: [[A=B]]\n"
        "4. Assistant B is slightly better: [[B>A]]\n"
        "5. Assistant B is significantly better: [[B>>A]]\n\n"
        "IMPORTANT: You must end your response with exactly one of the bracketed patterns above (e.g., [[A>B]]). Do not include any text after the final verdict pattern.\n\n"
        "Example output: \"My final verdict is tie: [[A=B]]\"."
    )

    template = (
        "<|User Prompt|>\n{question}\n\n"
        "<|The Start of Assistant A's Answer|>\n{answer_a}\n<|The End of Assistant A's Answer|>\n\n"
        "<|The Start of Assistant B's Answer|>\n{answer_b}\n<|The End of Assistant B's Answer|>"
    )

    formatted = template.format(question=question, answer_a=answer_a, answer_b=answer_b)
    return system_prompt + "\n\n" + formatted


def load_judgment_data(base_dir: Union[str, Path], question_id: str, model_name: str) -> Dict[str, Any]:
    """Load the processed Arena-Hard record for a given question/model.

    Search strategy:
    - Iterate subdirectories matching '*-setting*'
    - Prefer 'base_processed' subdir; expect per-model JSONL files named '{model_name}.jsonl'
    - Return the first record with matching 'question_id'

    Raises FileNotFoundError if no matching record is found.
    """
    import json

    base_path = Path(base_dir)

    # If user passed a base_processed directory directly, try it first
    if base_path.name == 'base_processed' and base_path.is_dir():
        model_file = base_path / f"{model_name}.jsonl"
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                    except Exception:
                        continue
                    if data.get('question_id') == question_id:
                        return data

    # Otherwise, search within setting directories
    for setting_dir in base_path.glob("*-setting*"):
        base_processed_dir = setting_dir / "base_processed"
        if not base_processed_dir.exists():
            continue
        model_file = base_processed_dir / f"{model_name}.jsonl"
        if not model_file.exists():
            continue
        with open(model_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                except Exception:
                    continue
                if data.get('question_id') == question_id:
                    return data

    raise FileNotFoundError(
        f"Could not find judgment data for question_id={question_id}, model={model_name} under {base_dir}"
    )


def setting_dir_for_judge(judge_name: str) -> Optional[str]:
    """Return expected InDepthAnalysis setting directory for a given judge name.

    Returns None if no explicit mapping is known.
    """
    j = judge_name.strip().lower()
    mapping = {
        'gpt-4o-mini': 'GPT-4o-mini-0718-setting1',
        'gpt-3.5-turbo': 'GPT-3.5-Turbo-0125-setting1',
        'deepseek-r1-32b': 'DeepSeek-R1-32B-setting1',
        'deepseek-r1-32b-gguf': 'DeepSeek-R1-32B-setting1',
        'qwq-32b': 'QwQ-32B-setting1',
        'qwq-32b-gguf': 'QwQ-32B-setting1',
    }
    return mapping.get(j)


def find_sample_data(base_path: Union[str, Path], judge_name: str, max_samples: int = 50):
    """Find sample evaluation data rows for sensitivity measurement.

    Returns a pandas DataFrame with columns: question_id, model, source_dir.
    """
    import json
    import pandas as pd

    base_path = Path(base_path)
    print(f"🔍 Searching for sample data in {base_path}...")

    expected = setting_dir_for_judge(judge_name)
    sample_rows: List[Dict[str, Any]] = []

    candidate_dirs: List[Path] = []
    if expected:
        cand = base_path / expected
        if cand.exists() and cand.is_dir():
            candidate_dirs = [cand]
        else:
            print(f"  ⚠️ Expected data source '{expected}' not found under {base_path}. Falling back to scan.")
    if not candidate_dirs:
        candidate_dirs = [d for d in base_path.glob("*-setting*") if d.is_dir()]

    for setting_dir in candidate_dirs:
        base_processed_dir = setting_dir / "base_processed"
        if not base_processed_dir.exists():
            continue
        print(f"  Found data source: {setting_dir.name}")

        # Load up to ~max_samples across up to 3 model files
        for jsonl_file in list(base_processed_dir.glob("*.jsonl"))[:3]:
            model_name = jsonl_file.stem
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if count >= max_samples // 3:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        qid = obj.get('question_id')
                        if not qid:
                            continue
                        sample_rows.append({
                            'question_id': qid,
                            'model': model_name,
                            'source_dir': str(base_processed_dir)
                        })
                        count += 1
            except Exception as e:
                print(f"    ⚠️ Error reading {jsonl_file}: {e}")
                continue

        if sample_rows:
            break

    if not sample_rows:
        raise ValueError(f"No evaluation data found in {base_path}")

    df = pd.DataFrame(sample_rows)
    print(f"✅ Found {len(df)} sample evaluations across {df['model'].nunique()} models")
    return df
