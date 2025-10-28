"""
Integration test for Combined A-BB sensitivity estimation using synthetic data.

The original version of this test relied on archived scripts that queried
real judges and accessed researcher-specific datasets. To keep the regression
suite deterministic, we now exercise the pipeline with synthetic judgments.
"""

from ..test_data_helpers import (
    create_synthetic_judge_function,
    load_and_prepare_score_data,
)
from differential_debiasing.sensitivity.combined_abb_sensitivity import (
    CombinedABBSensitivity,
)


def test_combined_sensitivity_with_synthetic_judge():
    """Ensure Combined A-BB runs end-to-end on synthetic data."""
    df = load_and_prepare_score_data(n_samples=120, random_seed=1234)
    judge_function = create_synthetic_judge_function(df)

    estimator = CombinedABBSensitivity(
        static_estimators=["psychometric_reliability", "schematic_adherence"],
        dynamic_generators=["hamming"],
        combination_strategy="conservative",
        num_neighbors=2,
        judge_function=judge_function,
        random_seed=99,
    )

    factor_columns = [
        col for col in df.columns if col.endswith("_score") and col != "overall_score"
    ]
    estimator.fit(df, factor_columns=factor_columns)

    sensitivity = estimator.estimate(score_range=4.0)
    breakdown = estimator.get_measurement_breakdown()

    assert sensitivity >= 0.0
    assert breakdown["static_measurements"]
    assert breakdown["dynamic_measurements"]
