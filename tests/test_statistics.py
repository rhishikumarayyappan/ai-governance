"""Tests for governance/testing/statistics.py — Phase 2 Component 2.1.

Six tests, one per behaviour the significance layer must get right. Every
parametric test is cross-checked against a hand calculation or scipy's own
output; the permutation test is checked for statistical convergence, not just
that it runs.

The parametric sanity tests pass ``cross_check=False`` so they are not slowed
by a 1,000-iteration permutation they are not testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency, norm

from governance.testing.statistics import ALPHA, significance_test


def _labels(pos: int, total: int) -> list[int]:
    """`pos` ones followed by `total - pos` zeros."""
    return [1] * pos + [0] * (total - pos)


# --------------------------------------------------------------------------- #
# Test 1 — Chi-squared on known data with an obvious association
# --------------------------------------------------------------------------- #
def test_chi_squared_on_known_strong_association():
    # Group A: 90% positive, Group B: 10% positive, 50 per group.
    y_pred = np.array(_labels(45, 50) + _labels(5, 50))
    sensitive = pd.Series(["A"] * 50 + ["B"] * 50)

    result = significance_test(
        y_pred, y_pred, sensitive, method="chi_squared", cross_check=False
    )

    assert result.test_used == "chi_squared"
    assert result.p_value < 0.001          # highly significant
    assert result.significant is True
    assert result.assumptions_met is True  # every expected cell count is 25
    assert result.sample_sizes == {"A": 50, "B": 50}

    # Matches scipy's own chi2_contingency to 10 dp (orientation-invariant).
    scipy_chi2, scipy_p, _, _ = chi2_contingency(
        pd.crosstab(sensitive, pd.Series(y_pred))
    )
    assert result.statistic == pytest.approx(scipy_chi2, abs=1e-10)
    assert result.p_value == pytest.approx(scipy_p, abs=1e-10)


# --------------------------------------------------------------------------- #
# Test 2 — Fisher's exact auto-selected on a small sample
# --------------------------------------------------------------------------- #
def test_fishers_exact_auto_selected_on_small_sample():
    # 8 per group, clear association. Row/col totals force an expected cell
    # count below 5, so chi-squared is invalid and auto must fall back.
    y_pred = np.array(_labels(7, 8) + _labels(1, 8))
    sensitive = pd.Series(["A"] * 8 + ["B"] * 8)

    result = significance_test(
        y_pred, y_pred, sensitive, method="auto", cross_check=False
    )

    assert result.test_used == "fishers_exact"
    # chi-squared's requirement was NOT met — that is why we fell back — and the
    # result says so rather than silently presenting a Fisher p as "assumptions ok".
    assert result.assumptions_met is False
    assert "expected cell count" in result.detail["auto_selection"]
    assert "assumption_notes" in result.detail
    # p-value is real and in range.
    assert 0.0 < result.p_value <= 1.0
    assert result.p_value < ALPHA  # this association is strong enough to detect


# --------------------------------------------------------------------------- #
# Test 3 — genuinely no association returns a high p-value
# --------------------------------------------------------------------------- #
def test_no_association_returns_high_p_value():
    rng = np.random.default_rng(20260902)
    n = 300
    # Outcome drawn independently of group for both groups.
    y_pred = np.concatenate([rng.integers(0, 2, n), rng.integers(0, 2, n)])
    sensitive = pd.Series(["A"] * n + ["B"] * n)

    result = significance_test(
        y_pred, y_pred, sensitive, method="auto", random_state=1
    )

    assert result.p_value > 0.05
    assert result.significant is False
    # The assumption-free cross-check agrees there is nothing here.
    assert result.detail["permutation_p_value"] > 0.05


# --------------------------------------------------------------------------- #
# Test 4 — two-proportion z-test matches a manual calculation
# --------------------------------------------------------------------------- #
def test_z_test_matches_manual_calculation():
    # Group A: 60/100 positive.  Group B: 45/100 positive.  Round numbers.
    y_pred = np.array(_labels(60, 100) + _labels(45, 100))
    sensitive = pd.Series(["A"] * 100 + ["B"] * 100)

    # Hand calculation.
    p1, p2 = 0.60, 0.45
    n1 = n2 = 100
    p_pooled = (60 + 45) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
    z_expected = (p1 - p2) / se
    p_expected = 2 * norm.sf(abs(z_expected))

    result = significance_test(
        y_pred, y_pred, sensitive, method="z_test", cross_check=False
    )

    assert result.test_used == "z_test"
    assert result.statistic == pytest.approx(z_expected, abs=1e-12)
    assert result.p_value == pytest.approx(p_expected, abs=1e-12)
    assert result.assumptions_met is True  # 100 * 0.525 = 52.5 > 5 everywhere


# --------------------------------------------------------------------------- #
# Test 5 — permutation test converges across different seeds
# --------------------------------------------------------------------------- #
def test_permutation_test_converges_across_seeds():
    rng = np.random.default_rng(7)
    a = rng.choice([0, 1], 150, p=[0.4, 0.6])
    b = rng.choice([0, 1], 150, p=[0.55, 0.45])
    y_pred = np.concatenate([a, b])
    sensitive = pd.Series(["A"] * 150 + ["B"] * 150)

    p_a = significance_test(
        y_pred, y_pred, sensitive, method="permutation", random_state=1
    ).p_value
    p_b = significance_test(
        y_pred, y_pred, sensitive, method="permutation", random_state=999
    ).p_value

    # 1,000 iterations should put the two Monte-Carlo estimates within ~0.02.
    assert abs(p_a - p_b) < 0.02


# --------------------------------------------------------------------------- #
# Test 6 — auto mode selects the right test for each shape
# --------------------------------------------------------------------------- #
def test_auto_mode_selects_correct_test():
    # (a) large, balanced, 2 groups -> chi-squared
    big_y = np.array(_labels(120, 200) + _labels(95, 200))
    big_g = pd.Series(["A"] * 200 + ["B"] * 200)
    assert (
        significance_test(big_y, big_y, big_g, method="auto", cross_check=False).test_used
        == "chi_squared"
    )

    # (b) small, 2 groups -> Fisher's exact
    small_y = np.array(_labels(6, 8) + _labels(1, 8))
    small_g = pd.Series(["A"] * 8 + ["B"] * 8)
    assert (
        significance_test(small_y, small_y, small_g, method="auto", cross_check=False).test_used
        == "fishers_exact"
    )

    # (c) 3 groups -> permutation
    three_y = np.array(_labels(30, 60) + _labels(25, 60) + _labels(40, 60))
    three_g = pd.Series(["A"] * 60 + ["B"] * 60 + ["C"] * 60)
    assert (
        significance_test(three_y, three_y, three_g, method="auto", random_state=1).test_used
        == "permutation"
    )


# --------------------------------------------------------------------------- #
# Guard rails (not one of the six, but cheap and load-bearing)
# --------------------------------------------------------------------------- #
def test_non_binary_prediction_is_rejected():
    with pytest.raises(ValueError, match="binary"):
        significance_test(
            np.array([0, 1, 2, 1]),
            np.array([0.1, 0.9, 0.4, 0.7]),
            pd.Series(["A", "A", "B", "B"]),
        )


def test_two_group_only_methods_reject_more_than_two_groups():
    y = np.array([1, 0, 1, 0, 1, 0])
    g = pd.Series(["A", "A", "B", "B", "C", "C"])
    with pytest.raises(ValueError, match="permutation"):
        significance_test(y, y, g, method="chi_squared")
