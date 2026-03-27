"""Tests for the benchmarks package: ScoreType, BenchmarkLoader, and loaders."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from differential_debiasing.benchmarks import (
    ARENA_HARD_PAIRWISE,
    BINARY_PASS_FAIL,
    LIKERT_1_5,
    NUMERIC_1_10,
    BenchmarkLoader,
    GenericJsonlLoader,
    ScoreType,
    get_loader,
    register_loader,
)


# ---------------------------------------------------------------------------
# ScoreType tests
# ---------------------------------------------------------------------------

class TestScoreType:
    def test_score_range(self):
        assert ARENA_HARD_PAIRWISE.score_range == (1.0, 5.0)
        assert BINARY_PASS_FAIL.score_range == (0.0, 1.0)
        assert NUMERIC_1_10.score_range == (1.0, 10.0)

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="min.*must be less than max"):
            ScoreType(type="numeric", min=5.0, max=5.0)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown score type"):
            ScoreType(type="freeform", min=0, max=1)

    def test_to_numeric_passthrough(self):
        assert NUMERIC_1_10.to_numeric(7) == 7.0
        assert NUMERIC_1_10.to_numeric(3.5) == 3.5
        assert NUMERIC_1_10.to_numeric("4") == 4.0

    def test_to_numeric_value_map(self):
        assert ARENA_HARD_PAIRWISE.to_numeric("A>>B") == 1.0
        assert ARENA_HARD_PAIRWISE.to_numeric("B>>A") == 5.0
        assert ARENA_HARD_PAIRWISE.to_numeric("A=B") == 3.0

    def test_to_numeric_binary(self):
        assert BINARY_PASS_FAIL.to_numeric("pass") == 1.0
        assert BINARY_PASS_FAIL.to_numeric("fail") == 0.0
        assert BINARY_PASS_FAIL.to_numeric("true") == 1.0
        assert BINARY_PASS_FAIL.to_numeric("false") == 0.0
        assert BINARY_PASS_FAIL.to_numeric("yes") == 1.0
        assert BINARY_PASS_FAIL.to_numeric("no") == 0.0

    def test_to_numeric_case_insensitive(self):
        assert BINARY_PASS_FAIL.to_numeric("PASS") == 1.0
        assert BINARY_PASS_FAIL.to_numeric("Pass") == 1.0

    def test_to_numeric_unknown_label(self):
        with pytest.raises(ValueError, match="Unknown label"):
            ARENA_HARD_PAIRWISE.to_numeric("C>>D")

    def test_to_numeric_unconvertible(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            NUMERIC_1_10.to_numeric(None)

    def test_arena_hard_value_map_completeness(self):
        """Arena-Hard value_map should include all standard + alternate tokens."""
        expected_tokens = {"A>>B", "A>B", "A=B", "B>A", "B>>A",
                          "A<<B", "A<B", "B=A", "B<A", "B<<A"}
        assert expected_tokens == set(ARENA_HARD_PAIRWISE.value_map.keys())

    def test_predefined_constants_are_frozen(self):
        with pytest.raises(AttributeError):
            NUMERIC_1_10.min = 0

    def test_is_discrete(self):
        assert LIKERT_1_5.is_discrete is True
        assert BINARY_PASS_FAIL.is_discrete is True
        custom = ScoreType(type="numeric", min=0.0, max=1.0, is_discrete=False)
        assert custom.is_discrete is False


# ---------------------------------------------------------------------------
# GenericJsonlLoader tests
# ---------------------------------------------------------------------------

class TestGenericJsonlLoader:
    def _write_jsonl(self, records, tmpdir):
        path = tmpdir / "test.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return path

    def test_basic_load(self, tmp_path):
        records = [
            {"item_id": "q1", "model": "m1", "score": 7},
            {"item_id": "q2", "model": "m1", "score": 8},
            {"item_id": "q1", "model": "m2", "score": 5},
        ]
        path = self._write_jsonl(records, tmp_path)
        loader = GenericJsonlLoader()
        df = loader.load_scores(path)

        assert set(df.columns) >= {"item_id", "model", "score"}
        assert len(df) == 3
        assert df["score"].dtype == float

    def test_custom_field_names(self, tmp_path):
        records = [
            {"qid": "q1", "agent": "a1", "rating": 9},
            {"qid": "q2", "agent": "a1", "rating": 6},
        ]
        path = self._write_jsonl(records, tmp_path)
        loader = GenericJsonlLoader(
            item_id_field="qid",
            model_field="agent",
            score_field="rating",
        )
        df = loader.load_scores(path)

        assert "item_id" in df.columns
        assert "model" in df.columns
        assert "score" in df.columns
        assert len(df) == 2

    def test_missing_item_id_synthesized(self, tmp_path):
        records = [{"model": "m1", "score": 5}]
        path = self._write_jsonl(records, tmp_path)
        loader = GenericJsonlLoader()
        df = loader.load_scores(path)
        assert "item_id" in df.columns

    def test_missing_model_synthesized(self, tmp_path):
        records = [{"item_id": "q1", "score": 5}]
        path = self._write_jsonl(records, tmp_path)
        loader = GenericJsonlLoader()
        df = loader.load_scores(path)
        assert "model" in df.columns
        assert df["model"].iloc[0] == "unknown"

    def test_missing_score_field_raises(self, tmp_path):
        records = [{"item_id": "q1", "model": "m1", "value": 5}]
        path = self._write_jsonl(records, tmp_path)
        loader = GenericJsonlLoader()
        with pytest.raises(ValueError, match="Score field"):
            loader.load_scores(path)

    def test_binary_score_type(self, tmp_path):
        records = [
            {"item_id": "q1", "model": "m1", "score": "pass"},
            {"item_id": "q2", "model": "m1", "score": "fail"},
        ]
        path = self._write_jsonl(records, tmp_path)
        loader = GenericJsonlLoader(score_type=BINARY_PASS_FAIL)
        df = loader.load_scores(path)

        assert df["score"].tolist() == [1.0, 0.0]

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loader = GenericJsonlLoader()
        with pytest.raises(ValueError, match="No records"):
            loader.load_scores(path)

    def test_score_type_property(self):
        loader = GenericJsonlLoader(score_type=LIKERT_1_5)
        assert loader.score_type is LIKERT_1_5
        assert loader.score_type.score_range == (1.0, 5.0)

    def test_name_property(self):
        loader = GenericJsonlLoader(loader_name="my_bench")
        assert loader.name == "my_bench"


# ---------------------------------------------------------------------------
# Aggregate tests
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_default_aggregate_by_model(self):
        loader = GenericJsonlLoader()
        df = pd.DataFrame({
            "item_id": ["q1", "q2", "q1", "q2"],
            "model": ["m1", "m1", "m2", "m2"],
            "score": [7.0, 8.0, 5.0, 6.0],
        })
        agg = loader.aggregate_by_model(df)
        assert set(agg.columns) == {"model", "score"}
        assert len(agg) == 2
        m1_score = agg.loc[agg["model"] == "m1", "score"].iloc[0]
        assert m1_score == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_loader_generic(self):
        loader = get_loader("generic_jsonl")
        assert loader.name == "generic_jsonl"

    def test_get_loader_arena_hard(self):
        loader = get_loader("arena_hard")
        assert loader.name == "arena_hard"
        assert loader.score_type is ARENA_HARD_PAIRWISE

    def test_get_loader_unknown(self):
        with pytest.raises(KeyError, match="Unknown benchmark loader"):
            get_loader("nonexistent")

    def test_register_custom_loader(self):
        class DummyLoader(BenchmarkLoader):
            @property
            def name(self):
                return "dummy"

            @property
            def score_type(self):
                return NUMERIC_1_10

            def load_scores(self, path, **kwargs):
                return pd.DataFrame({"item_id": [], "model": [], "score": []})

        register_loader("dummy", DummyLoader)
        loader = get_loader("dummy")
        assert loader.name == "dummy"
