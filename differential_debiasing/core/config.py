"""
Lightweight configuration helpers for differential debiasing.

The original project carried several legacy estimators (factor analysis,
empirical triggers, conformal hybrids, etc.). After pruning those paths we
retain a much smaller surface area, so the configuration layer mirrors the
reduced feature set.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


SUPPORTED_METHODS = {
    "schematic_adherence",
    "psychometric_reliability",
    "abb",
    "combined_abb",
    "profile_enhanced_abb",
    "fixed",
}


@dataclass
class DebiasConfig:
    """Configuration for differential debiasing."""

    tau: float = 0.5
    delta: float = 0.05
    sensitivity_method: str = "schematic_adherence"

    # Estimator-specific knobs
    factor_columns: Optional[list] = None
    target_column: str = "overall_score"
    sensitivity_value: Optional[float] = None  # Used when sensitivity_method == "fixed"

    # Runtime options
    use_average_case: bool = True
    random_seed: Optional[int] = None

    # Output naming
    score_field: str = "score"
    debiased_field: str = "debiased_score"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DebiasConfig":
        return cls(**config_dict)

    def validate(self) -> None:
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        if self.sensitivity_method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown sensitivity method '{self.sensitivity_method}'. "
                f"Supported: {sorted(SUPPORTED_METHODS)}"
            )
        if (
            self.sensitivity_method == "fixed"
            and self.sensitivity_value is not None
            and self.sensitivity_value <= 0
        ):
            raise ValueError("sensitivity_value must be positive when using the fixed estimator")


class ConfigManager:
    """Manager for loading/saving simple presets."""

    def __init__(self):
        self.presets = self._load_presets()

    def _load_presets(self) -> Dict[str, DebiasConfig]:
        return {
            "schematic_default": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="schematic_adherence",
            ),
            "psychometric_default": DebiasConfig(
                tau=0.75,
                delta=0.05,
                sensitivity_method="psychometric_reliability",
            ),
            "abb_default": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="abb",
            ),
            "combined_abb": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="combined_abb",
            ),
            "profile_enhanced": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="profile_enhanced_abb",
            ),
            "fixed_1pt0": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="fixed",
                sensitivity_value=1.0,
            ),
        }

    def get_preset(self, name: str) -> DebiasConfig:
        if name not in self.presets:
            available = ", ".join(sorted(self.presets))
            raise KeyError(f"Unknown preset '{name}'. Available: {available}")
        return self.presets[name]
