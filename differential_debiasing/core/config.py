"""
Configuration system for differential debiasing
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class DebiasConfig:
    """Configuration for differential debiasing."""
    
    # Core bias parameters
    tau: float = 0.5
    delta: float = 0.05
    
    # Sensitivity estimation
    sensitivity_method: str = "factor_analysis"
    
    # Factor analysis specific
    factor_columns: Optional[list] = None
    target_column: str = "score"
    use_factor_analyzer: bool = False
    n_factors: Optional[int] = None
    
    # Empirical specific
    bias_triggers: Optional[list] = None
    
    # Cross-validation specific
    n_folds: int = 5
    
    # Historical specific
    percentile: float = 95
    window_size: Optional[int] = None
    min_samples: int = 10
    
    # Domain-specific
    domain: str = "general"
    scale_type: str = "1-10"
    sensitivity_value: Optional[float] = None
    
    # Psychometric reliability specific
    alpha_weight: float = 1/3
    clr_weight: float = 1/3
    htmt_weight: float = 1/3
    clr_max: float = 2.0
    htmt_threshold: float = 0.85
    min_items_per_factor: int = 2
    
    # Schematic adherence specific
    use_polynomial: bool = True
    use_interactions: bool = True
    use_clustering: bool = False
    n_clusters: int = 3
    cluster_column: Optional[str] = None
    min_samples_per_cluster: int = 10
    regularization_alpha: float = 0.01
    
    # Combined sensitivity specific  
    combination_alpha: float = 0.5  # Weight between psychometric vs schematic
    combination_gamma: float = 1.0  # Calibration constant
    
    # Processing options
    use_average_case: bool = True
    random_seed: Optional[int] = None
    
    # Data processing
    score_field: str = "score"
    debiased_field: str = "debiased_score"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DebiasConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        
        if not 0 < self.delta < 1:
            raise ValueError("delta must be in (0, 1)")
        
        valid_methods = ["factor_analysis", "empirical", "cross_validation", "historical", "domain_specific", 
                        "psychometric_reliability", "schematic_adherence", "combined"]
        if self.sensitivity_method not in valid_methods:
            raise ValueError(f"Unknown sensitivity method: {self.sensitivity_method}")
        
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")
        
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        
        if self.sensitivity_value is not None and self.sensitivity_value <= 0:
            raise ValueError("sensitivity_value must be positive")


class ConfigManager:
    """Manager for loading and saving configurations."""
    
    def __init__(self):
        self.presets = self._load_presets()
    
    def _load_presets(self) -> Dict[str, DebiasConfig]:
        """Load preset configurations."""
        return {
            "conservative": DebiasConfig(
                tau=0.3,
                delta=0.01,
                sensitivity_method="factor_analysis",
                use_average_case=True
            ),
            "moderate": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="factor_analysis",
                use_average_case=True
            ),
            "permissive": DebiasConfig(
                tau=0.7,
                delta=0.1,
                sensitivity_method="factor_analysis",
                use_average_case=True
            ),
            "academic": DebiasConfig(
                tau=0.4,
                delta=0.05,
                sensitivity_method="factor_analysis",
                domain="academic",
                use_average_case=True
            ),
            "creative": DebiasConfig(
                tau=0.6,
                delta=0.05,
                sensitivity_method="empirical",
                domain="creative",
                use_average_case=True
            ),
            "medical": DebiasConfig(
                tau=0.2,
                delta=0.01,
                sensitivity_method="domain_specific",
                domain="medical",
                use_average_case=True
            ),
            "empirical_testing": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="empirical",
                bias_triggers=["verbose_vs_concise", "formal_vs_casual"],
                use_average_case=True
            ),
            "historical_analysis": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="historical",
                percentile=95,
                window_size=50,
                use_average_case=True
            ),
            "cross_validation": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="cross_validation",
                n_folds=10,
                use_average_case=True
            ),
            "psychometric_reliability": DebiasConfig(
                tau=1.0,
                delta=0.05,
                sensitivity_method="psychometric_reliability",
                factor_columns=None,  # Auto-detect
                alpha_weight=1/3,
                clr_weight=1/3,
                htmt_weight=1/3,
                clr_max=2.0,
                htmt_threshold=0.85,
                min_items_per_factor=2,
                use_average_case=True
            ),
            "schematic_adherence": DebiasConfig(
                tau=1.0,
                delta=0.05,
                sensitivity_method="schematic_adherence",
                factor_columns=None,  # Auto-detect
                target_column="overall_score",
                use_polynomial=True,
                use_interactions=True,
                use_clustering=False,
                regularization_alpha=0.01,
                use_average_case=True
            ),
            "combined_balanced": DebiasConfig(
                tau=1.0,
                delta=0.05,
                sensitivity_method="combined",
                factor_columns=None,  # Auto-detect
                target_column="overall_score",
                combination_alpha=0.5,  # Equal weighting
                combination_gamma=1.0,
                # Psychometric settings
                alpha_weight=1/3,
                clr_weight=1/3,
                htmt_weight=1/3,
                # Schematic settings
                use_polynomial=True,
                use_interactions=True,
                use_average_case=True
            ),
            "combined_psychometric_focused": DebiasConfig(
                tau=0.4,
                delta=0.05,
                sensitivity_method="combined",
                factor_columns=None,
                target_column="overall_score",
                combination_alpha=0.8,  # Favor psychometric reliability
                combination_gamma=1.0,
                alpha_weight=1/3,
                clr_weight=1/3,
                htmt_weight=1/3,
                use_polynomial=True,
                use_interactions=True,
                use_average_case=True
            ),
            "combined_schematic_focused": DebiasConfig(
                tau=0.5,
                delta=0.05,
                sensitivity_method="combined",
                factor_columns=None,
                target_column="overall_score",
                combination_alpha=0.2,  # Favor schematic adherence
                combination_gamma=1.0,
                alpha_weight=1/3,
                clr_weight=1/3,
                htmt_weight=1/3,
                use_polynomial=True,
                use_interactions=True,
                use_clustering=True,
                n_clusters=3,
                use_average_case=True
            ),
            "arena_hard_auto": DebiasConfig(
                tau=0.4,
                delta=0.05,
                sensitivity_method="combined",
                factor_columns=["correctness_score", "completeness_score", "safety_score", "conciseness_score", "style_score"],
                target_column="overall_score",
                combination_alpha=0.6,  # Slightly favor psychometric
                combination_gamma=0.8,  # Conservative calibration
                use_polynomial=True,
                use_interactions=True,
                use_average_case=True
            )
        }
    
    def get_preset(self, name: str) -> DebiasConfig:
        """Get a preset configuration."""
        if name not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        return self.presets[name]
    
    def list_presets(self) -> list:
        """List available preset names."""
        return list(self.presets.keys())
    
    def load_config(self, filepath: Union[str, Path]) -> DebiasConfig:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
                config_dict = yaml.safe_load(f)
            elif filepath.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {filepath.suffix}")
        
        config = DebiasConfig.from_dict(config_dict)
        config.validate()
        return config
    
    def save_config(self, config: DebiasConfig, filepath: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        config_dict = config.to_dict()
        
        # Remove None values for cleaner output
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        with open(filepath, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def create_config_template(self, filepath: Union[str, Path], format: str = "yaml") -> None:
        """Create a configuration template file."""
        template_config = DebiasConfig()
        self.save_config(template_config, filepath, format)
    
    def validate_config_file(self, filepath: Union[str, Path]) -> bool:
        """Validate a configuration file."""
        try:
            config = self.load_config(filepath)
            config.validate()
            return True
        except Exception as e:
            print(f"Config validation failed: {e}")
            return False
    
    def merge_configs(self, base_config: DebiasConfig, override_dict: Dict[str, Any]) -> DebiasConfig:
        """Merge configuration with override dictionary."""
        config_dict = base_config.to_dict()
        
        # Update with overrides
        for key, value in override_dict.items():
            if key in config_dict:
                config_dict[key] = value
            else:
                print(f"Warning: Unknown configuration key '{key}' ignored")
        
        merged_config = DebiasConfig.from_dict(config_dict)
        merged_config.validate()
        return merged_config


def create_example_configs():
    """Create example configuration files."""
    config_manager = ConfigManager()
    
    # Create example directory
    example_dir = Path("examples/configs")
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all presets as example configs
    for preset_name, config in config_manager.presets.items():
        config_path = example_dir / f"{preset_name}.yaml"
        config_manager.save_config(config, config_path)
    
    # Create a template config
    template_path = example_dir / "template.yaml"
    config_manager.create_config_template(template_path)
    
    print(f"Created example configs in {example_dir}")


if __name__ == "__main__":
    create_example_configs()