"""
Score type definitions for benchmark loaders.

Represents the schema of scores produced by a benchmark — type, range, and
optional mappings from string labels to numeric values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ScoreType:
    """Describes a finite-range, real-valued score schema.

    Parameters
    ----------
    type : str
        One of "likert", "numeric", "binary", "categorical", "pairwise".
    min : float
        Minimum numeric value in the score range.
    max : float
        Maximum numeric value in the score range.
    labels : dict or None
        Optional mapping from score values to human-readable descriptions.
        E.g. {"1": "Poor", "5": "Excellent"} for Likert scales.
    value_map : dict or None
        Optional mapping from string labels to float values, used for
        categorical, binary, and pairwise score types.
        E.g. {"pass": 1.0, "fail": 0.0} or {"A>>B": 1.0, "A>B": 2.0, ...}.
    is_discrete : bool
        Whether scores are restricted to integer values.
    """

    type: str
    min: float
    max: float
    labels: Optional[Dict[str, str]] = field(default=None, repr=False)
    value_map: Optional[Dict[str, float]] = field(default=None, repr=False)
    is_discrete: bool = False

    def __post_init__(self):
        if self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be less than max ({self.max})")
        if self.type not in ("likert", "numeric", "binary", "categorical", "pairwise"):
            raise ValueError(
                f"Unknown score type '{self.type}'. "
                "Must be one of: likert, numeric, binary, categorical, pairwise."
            )

    @property
    def score_range(self) -> Tuple[float, float]:
        """Return (min, max) tuple for use with DifferentialDebias."""
        return (self.min, self.max)

    def to_numeric(self, raw) -> float:
        """Convert a raw score value to a float.

        If *value_map* is defined and *raw* is a string key in the map, the
        mapped float is returned.  Otherwise ``float(raw)`` is attempted.

        Raises
        ------
        ValueError
            If the raw value cannot be converted.
        """
        if self.value_map is not None and isinstance(raw, str):
            cleaned = raw.strip()
            if cleaned in self.value_map:
                return self.value_map[cleaned]
            # Try case-insensitive lookup
            for key, val in self.value_map.items():
                if key.lower() == cleaned.lower():
                    return val
            raise ValueError(
                f"Unknown label '{raw}' for score type '{self.type}'. "
                f"Valid labels: {sorted(self.value_map.keys())}"
            )
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"Cannot convert '{raw}' (type {type(raw).__name__}) to numeric score."
            )


# ---------------------------------------------------------------------------
# Predefined score types
# ---------------------------------------------------------------------------

ARENA_HARD_PAIRWISE = ScoreType(
    type="pairwise",
    min=1.0,
    max=5.0,
    labels={
        "1": "A significantly better (A>>B)",
        "2": "A slightly better (A>B)",
        "3": "Tie (A=B)",
        "4": "B slightly better (B>A)",
        "5": "B significantly better (B>>A)",
    },
    value_map={
        "A>>B": 1.0,
        "A>B": 2.0,
        "A=B": 3.0,
        "B>A": 4.0,
        "B>>A": 5.0,
        # Alternate notation
        "A<<B": 5.0,
        "A<B": 4.0,
        "B=A": 3.0,
        "B<A": 2.0,
        "B<<A": 1.0,
    },
    is_discrete=True,
)

LIKERT_1_5 = ScoreType(
    type="likert",
    min=1.0,
    max=5.0,
    labels={
        "1": "Strongly disagree / Very poor",
        "2": "Disagree / Poor",
        "3": "Neutral / Acceptable",
        "4": "Agree / Good",
        "5": "Strongly agree / Excellent",
    },
    is_discrete=True,
)

NUMERIC_1_10 = ScoreType(
    type="numeric",
    min=1.0,
    max=10.0,
    is_discrete=True,
)

BINARY_PASS_FAIL = ScoreType(
    type="binary",
    min=0.0,
    max=1.0,
    value_map={
        "pass": 1.0,
        "fail": 0.0,
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "1": 1.0,
        "0": 0.0,
    },
    is_discrete=True,
)
