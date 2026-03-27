"""
Generic JSONL benchmark loader.

Handles arbitrary JSONL files with configurable field names and score types.
Works out-of-the-box with any JSONL that has an item identifier, a model
name, and a numeric (or string-mapped) score field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BenchmarkLoader
from .score_type import NUMERIC_1_10, ScoreType


class GenericJsonlLoader(BenchmarkLoader):
    """Load scores from a flat JSONL file.

    Parameters
    ----------
    score_type : ScoreType
        Score schema for the data.  Defaults to ``NUMERIC_1_10``.
    item_id_field : str
        JSONL key containing the per-item identifier.
    model_field : str
        JSONL key containing the model name.
    score_field : str
        JSONL key containing the raw score value.
    loader_name : str
        Name returned by the ``name`` property.
    """

    def __init__(
        self,
        score_type: ScoreType = NUMERIC_1_10,
        item_id_field: str = "item_id",
        model_field: str = "model",
        score_field: str = "score",
        loader_name: str = "generic_jsonl",
    ):
        self._score_type = score_type
        self._item_id_field = item_id_field
        self._model_field = model_field
        self._score_field = score_field
        self._loader_name = loader_name

    @property
    def name(self) -> str:
        return self._loader_name

    @property
    def score_type(self) -> ScoreType:
        return self._score_type

    def load_scores(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load a JSONL file and return a standardised DataFrame.

        Parameters
        ----------
        path : Path
            Path to the JSONL file.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``item_id``, ``model``, ``score`` columns.
            Additional fields from the JSONL are preserved.
        """
        path = Path(path)
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            raise ValueError(f"No records found in {path}")

        df = pd.DataFrame(records)

        # Rename configured fields to standard column names
        rename_map = {}
        if self._item_id_field != "item_id" and self._item_id_field in df.columns:
            rename_map[self._item_id_field] = "item_id"
        if self._model_field != "model" and self._model_field in df.columns:
            rename_map[self._model_field] = "model"
        if self._score_field != "score" and self._score_field in df.columns:
            rename_map[self._score_field] = "score"

        if rename_map:
            df = df.rename(columns=rename_map)

        # Synthesize item_id from index if not present
        if "item_id" not in df.columns:
            df["item_id"] = [str(i) for i in range(len(df))]

        # Synthesize model if not present
        if "model" not in df.columns:
            df["model"] = "unknown"

        # Convert scores to numeric
        if "score" not in df.columns:
            raise ValueError(
                f"Score field '{self._score_field}' not found in JSONL. "
                f"Available fields: {sorted(records[0].keys())}"
            )

        df["score"] = df["score"].apply(self._score_type.to_numeric)

        self._validate_dataframe(df)
        return df
