"""
Abstract base class for benchmark loaders.

A BenchmarkLoader knows how to read evaluation data from a specific benchmark
format and produce a standardised DataFrame with columns:

    item_id  model  score  [... additional columns preserved]

The ``score`` column always contains float values within the range declared
by the loader's ``score_type``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .score_type import ScoreType


REQUIRED_COLUMNS = ("item_id", "model", "score")


class BenchmarkLoader(ABC):
    """Base class for loading benchmark evaluation data."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this benchmark (e.g. 'arena_hard', 'mt_bench')."""
        ...

    @property
    @abstractmethod
    def score_type(self) -> ScoreType:
        """Score schema produced by this benchmark."""
        ...

    @abstractmethod
    def load_scores(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load evaluation data and return a standardised DataFrame.

        The returned DataFrame **must** contain at least the columns
        ``item_id``, ``model``, and ``score`` (float).  Additional columns
        are preserved.

        Parameters
        ----------
        path : Path
            Root directory or file for the benchmark data.
        **kwargs
            Loader-specific options.
        """
        ...

    def aggregate_by_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-item scores to per-model summaries.

        Default implementation: group by ``model`` and take the mean of
        ``score``.  Override for benchmark-specific aggregation (e.g. ELO
        from pairwise battles).

        Returns a DataFrame with columns ``model`` and ``score``.
        """
        return df.groupby("model")["score"].mean().reset_index()

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Raise if required columns are missing from *df*."""
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"BenchmarkLoader '{self.name}' produced a DataFrame missing "
                f"required columns: {missing}"
            )
