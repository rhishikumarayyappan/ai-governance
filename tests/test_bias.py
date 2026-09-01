"""Phase 1, Week 2, Component 2 — tests for the BiasTestSuite.

Five tests, as specified in docs/BUILD_PLAN.md and the session brief.
"""

import numpy as np
import pandas as pd

from governance.testing.bias import BiasTestSuite

_FIXED_ORDER = [
    "demographic_parity_difference",
    "equalized_odds_difference",
    "equal_opportunity_difference",
    "predictive_parity_difference",
    "individual_fairness_score",
]


def _dpd(results):
    return next(r for r in results if r.metric_name == "demographic_parity_difference")


def test_perfect_fairness():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    sensitive_features = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    results = BiasTestSuite().run(y_true, y_pred, sensitive_features, "group")

    assert len(results) == 5
    dpd = _dpd(results)
    assert abs(dpd.value - 0.0) <= 0.001
    assert dpd.status == "pass"


def test_clear_bias():
    # Group A (8 people): all predicted positive. Group B (8): all predicted negative.
    y_pred = np.array([1] * 8 + [0] * 8)
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1])
    sensitive_features = pd.Series(["A"] * 8 + ["B"] * 8)

    results = BiasTestSuite().run(y_true, y_pred, sensitive_features, "group")

    dpd = _dpd(results)
    assert dpd.value > 0.5  # A gets 100% positive, B gets 0% -> gap ~1.0
    assert dpd.status == "fail"


def test_always_returns_five_results_in_fixed_order():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    sensitive_features = pd.Series(["A", "B"] * 4)

    results = BiasTestSuite().run(y_true, y_pred, sensitive_features, "group")

    assert len(results) == 5
    assert [r.metric_name for r in results] == _FIXED_ORDER


def test_custom_threshold_override_changes_outcome():
    # Group A: 12/25 predicted positive = 0.48. Group B: 10/25 = 0.40. Gap = 0.08.
    y_pred = np.array([1] * 12 + [0] * 13 + [1] * 10 + [0] * 15)
    y_true = np.array([1, 0] * 25)
    sensitive_features = pd.Series(["A"] * 25 + ["B"] * 25)

    default_result = _dpd(
        BiasTestSuite().run(y_true, y_pred, sensitive_features, "group")
    )
    assert abs(default_result.value - 0.08) <= 0.001
    assert default_result.status == "warn"  # 0.08 is <= 0.10 default but > 0.07

    strict_result = _dpd(
        BiasTestSuite({"demographic_parity_difference": 0.05}).run(
            y_true, y_pred, sensitive_features, "group"
        )
    )
    assert strict_result.status == "fail"  # 0.08 > 0.05 custom threshold


def test_detail_field_contains_per_group_rates_as_floats():
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    sensitive_features = pd.Series(
        ["male", "male", "male", "male", "female", "female", "female", "female"]
    )

    results = BiasTestSuite().run(y_true, y_pred, sensitive_features, "gender")

    detail = _dpd(results).detail
    assert isinstance(detail, dict)
    assert len(detail) >= 1
    for rate in detail.values():
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
