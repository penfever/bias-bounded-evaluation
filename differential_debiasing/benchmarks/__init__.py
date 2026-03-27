"""
Benchmark loaders for the differential debiasing pipeline.

Provides a unified interface for loading evaluation data from different
benchmark formats (Arena-Hard, MT-Bench, custom JSONL, etc.) into a
standardised DataFrame with ``item_id``, ``model``, ``score`` columns.
"""

from .score_type import (
    ScoreType,
    ARENA_HARD_PAIRWISE,
    BINARY_PASS_FAIL,
    LIKERT_1_5,
    NUMERIC_1_10,
)
from .base import BenchmarkLoader
from .arena_hard import ArenaHardLoader
from .generic_jsonl import GenericJsonlLoader

from typing import Dict, Type


LOADER_REGISTRY: Dict[str, Type[BenchmarkLoader]] = {
    "arena_hard": ArenaHardLoader,
    "generic_jsonl": GenericJsonlLoader,
}


def get_loader(name: str, **kwargs) -> BenchmarkLoader:
    """Instantiate a benchmark loader by name.

    Parameters
    ----------
    name : str
        Registered loader name (e.g. ``"arena_hard"``, ``"generic_jsonl"``).
    **kwargs
        Passed to the loader constructor.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in LOADER_REGISTRY:
        raise KeyError(
            f"Unknown benchmark loader '{name}'. "
            f"Available: {sorted(LOADER_REGISTRY.keys())}"
        )
    return LOADER_REGISTRY[name](**kwargs)


def register_loader(name: str, loader_cls: Type[BenchmarkLoader]) -> None:
    """Register a custom benchmark loader.

    Parameters
    ----------
    name : str
        Name to register under.
    loader_cls : type
        BenchmarkLoader subclass.
    """
    LOADER_REGISTRY[name] = loader_cls


__all__ = [
    "BenchmarkLoader",
    "ScoreType",
    "ArenaHardLoader",
    "GenericJsonlLoader",
    "get_loader",
    "register_loader",
    "LOADER_REGISTRY",
    "ARENA_HARD_PAIRWISE",
    "BINARY_PASS_FAIL",
    "LIKERT_1_5",
    "NUMERIC_1_10",
]
