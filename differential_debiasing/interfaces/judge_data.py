"""
Utilities for locating judge data, loading Arena-Hard evaluations, and preparing
score tables for downstream debiasing analyses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import pandas as pd


_DEFAULT_JUDGE_PATTERNS: Tuple[str, ...] = (
    "QwQ-32B-setting1",
    "DeepSeek-R1-32B-setting1",
    "DeepSeek-R1-32B-setting2",
    "GPT-3.5-Turbo-0125-setting1",
    "GPT-4o-mini-0718-setting1",
)

_JUDGE_CONFIG_MAPPING: Dict[str, str] = {
    # OpenAI models
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo.yaml",
    "gpt-4o-mini": "openai/gpt-4o-mini.yaml",
    # Anthropic models
    "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet.yaml",
    # Local GGUF models
    "qwq-32b-gguf": "local/qwq-32b-gguf.yaml",
    "deepseek-r1-32b-gguf": "local/deepseek-r1-32b-gguf.yaml",
}

_ARENA_SCORE_MAPPING: Dict[str, int] = {
    "": 3,
    "A>>B": 1,
    "A>B": 2,
    "A=B": 3,
    "B>A": 4,
    "B>>A": 5,
    "A<<B": 5,
    "A<B": 4,
    "B=A": 3,
    "B<A": 2,
    "B<<A": 1,
}


def get_judge_config_path(judge_name: str, configs_root: Optional[Path] = None) -> Path:
    """
    Resolve the YAML config path for a judge.

    Parameters
    ----------
    judge_name:
        Name of the judge (e.g. ``gpt-4o-mini``).
    configs_root:
        Optional override for the configs directory. Defaults to
        ``<repo root>/configs/judges``.
    """
    base_dir = configs_root or Path(__file__).resolve().parents[2] / "configs" / "judges"

    config_rel = _JUDGE_CONFIG_MAPPING.get(judge_name)
    if not config_rel:
        raise FileNotFoundError(
            f"Judge config not registered for '{judge_name}'. "
            f"Known judges: {sorted(_JUDGE_CONFIG_MAPPING)}"
        )

    config_path = base_dir / config_rel
    if not config_path.exists():
        raise FileNotFoundError(f"Judge config not found at {config_path}")
    return config_path


def find_judge_data_directories(
    base_path: Path,
    patterns: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """
    Locate judge directories containing Arena-Hard ``base_processed`` JSONL data.
    """
    judge_dirs: Dict[str, Path] = {}
    for pattern in patterns or _DEFAULT_JUDGE_PATTERNS:
        judge_dir = base_path / pattern
        base_processed_dir = judge_dir / "base_processed"
        if (
            judge_dir.exists()
            and judge_dir.is_dir()
            and base_processed_dir.exists()
            and any(base_processed_dir.glob("*.jsonl"))
        ):
            judge_dirs[pattern] = judge_dir
    return judge_dirs


def load_judge_evaluations(base_processed_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load Arena-Hard evaluation records from a ``base_processed`` directory.
    """
    evaluations: Dict[str, List[Dict[str, Any]]] = {}
    for jsonl_file in base_processed_dir.glob("*.jsonl"):
        model_name = jsonl_file.stem
        model_evaluations: List[Dict[str, Any]] = []
        try:
            with open(jsonl_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        model_evaluations.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue

        if model_evaluations:
            evaluations[model_name] = model_evaluations
    return evaluations


def get_arena_score_mapping() -> Dict[str, int]:
    """Expose the Arena-Hard pairwise-to-Likert mapping."""
    return _ARENA_SCORE_MAPPING.copy()


def extract_scores_from_evaluations(
    evaluations: Dict[str, List[Dict[str, Any]]],
    *,
    score_mapping: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Convert Arena-Hard evaluations into a tidy DataFrame of Likert scores.
    """
    mapping = score_mapping or _ARENA_SCORE_MAPPING
    rows: List[Dict[str, Any]] = []

    for model_name, model_evaluations in evaluations.items():
        for eval_data in model_evaluations:
            question_id = eval_data.get("question_id", "")
            games = eval_data.get("games", [])
            for game_idx, game in enumerate(games):
                row = {
                    "model": model_name,
                    "question_id": question_id,
                    "game_index": game_idx,
                }
                factor_fields = [
                    "score",
                    "correctness_score",
                    "completeness_score",
                    "safety_score",
                    "conciseness_score",
                    "style_score",
                ]
                for factor in factor_fields:
                    raw = game.get(factor)
                    if isinstance(raw, str) and raw.strip():
                        row[factor] = float(mapping.get(raw.strip(), 3))
                    else:
                        row[factor] = 3.0
                row.setdefault("overall_score", row.get("score", 3.0))
                rows.append(row)

    return pd.DataFrame(rows)


def load_and_prepare_score_data(
    judge_name: str,
    base_path: Path,
    *,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Convenience helper that loads evaluations for ``judge_name`` and returns
    a DataFrame of scores ready for debiasing analyses.
    """
    judge_dir = base_path / judge_name
    if not judge_dir.exists():
        raise ValueError(f"Judge directory not found: {judge_dir}")

    base_processed_dir = judge_dir / "base_processed"
    if not base_processed_dir.exists():
        raise ValueError(f"Base processed directory not found: {base_processed_dir}")

    evaluations = load_judge_evaluations(base_processed_dir)
    if not evaluations:
        raise ValueError(f"No evaluations found in {base_processed_dir}")

    df = extract_scores_from_evaluations(evaluations)
    if len(df) < min_samples:
        raise ValueError(f"Not enough evaluations for debiasing ({len(df)})")
    return df
