"""
Unit test for Combined A-BB estimator mixing static and dynamic measurements.
"""

from ..test_data_helpers import (
    create_synthetic_judge_function,
    load_and_prepare_score_data,
)
from differential_debiasing.sensitivity.combined_abb_sensitivity import (
    CombinedABBSensitivity,
)


def test_combined_static_dynamic_pipeline():
    """Combined estimator should blend static and dynamic signals."""
    df = load_and_prepare_score_data(n_samples=180, random_seed=7)
    judge_function = create_synthetic_judge_function(df)

    estimator = CombinedABBSensitivity(
        static_estimators=["psychometric_reliability", "schematic_adherence"],
        dynamic_generators=["hamming", "formatting"],
        combination_strategy="conservative",
        num_neighbors=3,
        judge_function=judge_function,
        random_seed=11,
    )

    factor_columns = [
        col for col in df.columns if col.endswith("_score") and col != "overall_score"
    ]
    estimator.fit(df, factor_columns=factor_columns)

    sensitivity = estimator.estimate(score_range=4.0)
    breakdown = estimator.get_measurement_breakdown()

    assert sensitivity >= 0.0
    assert set(breakdown.keys()) >= {
        "static_measurements",
        "dynamic_measurements",
        "combination_strategy",
    }
    assert breakdown["combination_strategy"] == "conservative"
    assert breakdown["static_measurements"]
    assert breakdown["dynamic_measurements"]
