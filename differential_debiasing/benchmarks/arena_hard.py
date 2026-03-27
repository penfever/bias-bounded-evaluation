"""
BenchmarkLoader for Arena-Hard(-Auto) pairwise evaluations.

Wraps existing functions in ``differential_debiasing.interfaces`` without
duplicating logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BenchmarkLoader
from .score_type import ARENA_HARD_PAIRWISE, ScoreType
from ..interfaces.judge_data import (
    load_judge_evaluations,
    extract_scores_from_evaluations,
    find_judge_data_directories,
)
from ..interfaces.arena_hard_utils import extract_verdict_token, verdict_token_to_score


class ArenaHardLoader(BenchmarkLoader):
    """Load Arena-Hard(-Auto) pairwise evaluation data.

    Expects a directory structure like::

        <path>/
            <JudgeName-setting>/
                base_processed/
                    model_a.jsonl
                    model_b.jsonl
                    ...

    Each JSONL record contains a ``games`` list with pairwise verdict tokens
    (``A>>B``, ``A>B``, ``A=B``, ``B>A``, ``B>>A``).

    Parameters
    ----------
    judge_pattern : str or None
        Specific judge directory name to load (e.g. ``"QwQ-32B-setting1"``).
        If *None*, all matching ``*-setting*`` directories are scanned.
    """

    def __init__(self, judge_pattern: Optional[str] = None):
        self._judge_pattern = judge_pattern

    @property
    def name(self) -> str:
        return "arena_hard"

    @property
    def score_type(self) -> ScoreType:
        return ARENA_HARD_PAIRWISE

    def load_scores(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Arena-Hard evaluations from *path*.

        Parameters
        ----------
        path : Path
            Base directory containing judge subdirectories with
            ``base_processed/`` JSONL files.

        Returns
        -------
        pd.DataFrame
            Standardised DataFrame with ``item_id``, ``model``, ``score``
            columns plus any factor score columns present in the data.
        """
        path = Path(path)

        # Determine which judge directories to scan
        if self._judge_pattern:
            patterns = [self._judge_pattern]
        else:
            patterns = None  # scan all matching dirs

        judge_dirs = find_judge_data_directories(path, patterns=patterns)
        if not judge_dirs:
            raise FileNotFoundError(
                f"No Arena-Hard base_processed directories found under {path}"
            )

        frames = []
        for judge_name, judge_dir in judge_dirs.items():
            base_processed = judge_dir / "base_processed"
            evaluations = load_judge_evaluations(base_processed)
            if not evaluations:
                continue
            df = extract_scores_from_evaluations(evaluations)
            df["judge"] = judge_name
            frames.append(df)

        if not frames:
            raise ValueError(f"No evaluation data found under {path}")

        combined = pd.concat(frames, ignore_index=True)

        # Rename to standard columns
        combined = combined.rename(columns={"question_id": "item_id"})

        # Ensure score column is numeric via score_type
        if "score" in combined.columns:
            combined["score"] = combined["score"].apply(
                lambda v: self.score_type.to_numeric(v)
                if isinstance(v, str)
                else float(v)
            )

        self._validate_dataframe(combined)
        return combined

    def aggregate_by_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate using ELO bootstrap when pairwise battles can be constructed.

        Falls back to mean aggregation when the ELO path fails (e.g. missing
        dependencies or insufficient data).
        """
        try:
            from ..interfaces.arena_hard_utils import convert_scores_to_win_rates

            agg = df.groupby("model")["score"].mean().reset_index()
            return convert_scores_to_win_rates(agg, score_column="score")
        except Exception:
            return super().aggregate_by_model(df)
