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

from governance.testing.statistics import (
    ALPHA,
    METRIC_WRAPPERS,
    BootstrapResult,
    MetricTensionResult,
    SimpsonsParadoxResult,
    apply_multiple_comparisons_correction,
    assess_reliability,
    bootstrap_confidence_interval,
    demographic_parity_wrapper,
    detect_metric_tensions,
    detect_simpsons_paradox,
    equal_opportunity_wrapper,
    equalized_odds_wrapper,
    overall_accuracy_floor_wrapper,
    predictive_parity_wrapper,
    significance_test,
)


class _FakeBiasResult:
    """Duck-types a BiasTestResult for detect_metric_tensions (which reads only
    .metric_name / .status / .value and never imports from bias.py)."""

    def __init__(self, metric_name, status, value=0.0):
        self.metric_name = metric_name
        self.status = status
        self.value = value


def _five_results(**status_by_metric):
    """Build 5 fake bias results; any metric not named defaults to 'pass'."""
    order = (
        "demographic_parity_difference",
        "equalized_odds_difference",
        "equal_opportunity_difference",
        "predictive_parity_difference",
        "overall_accuracy_floor",
    )
    return [_FakeBiasResult(m, status_by_metric.get(m, "pass")) for m in order]


def _labelled(group_pos_counts):
    """group_pos_counts: list of (group, stratum, n_positive, n_total).
    Returns (y_pred, sensitive_features, stratify_by) as aligned arrays."""
    yp, sf, sb = [], [], []
    for group, stratum, n_pos, n_tot in group_pos_counts:
        yp.extend([1] * n_pos + [0] * (n_tot - n_pos))
        sf.extend([group] * n_tot)
        sb.extend([stratum] * n_tot)
    return np.array(yp), pd.Series(sf), pd.Series(sb)


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


# =========================================================================== #
# Bootstrap confidence intervals — Phase 2, gap 1.3
# =========================================================================== #
def _grouped_predictions(n_per_group, p_a, p_b, seed):
    """n_per_group rows in group A at positive-rate p_a, likewise B at p_b."""
    rng = np.random.default_rng(seed)
    y_pred = np.concatenate(
        [
            rng.choice([0, 1], n_per_group, p=[1 - p_a, p_a]),
            rng.choice([0, 1], n_per_group, p=[1 - p_b, p_b]),
        ]
    )
    groups = pd.Series(["A"] * n_per_group + ["B"] * n_per_group)
    return y_pred, groups


# --------------------------------------------------------------------------- #
# Test 1 — the CI always contains its own point estimate
# --------------------------------------------------------------------------- #
def test_ci_contains_point_estimate():
    y_pred, groups = _grouped_predictions(400, p_a=0.7, p_b=0.4, seed=1)
    result = bootstrap_confidence_interval(
        demographic_parity_wrapper, y_pred, y_pred, groups,
        n_iterations=400, random_state=7,
    )
    assert result.ci_lower <= result.point_estimate <= result.ci_upper
    assert result.n_valid_iterations == 400


# --------------------------------------------------------------------------- #
# Test 2 — narrow CI on large, stable data (and no false reliability warning)
# --------------------------------------------------------------------------- #
def test_narrow_ci_on_large_sample():
    # 1,500 per group, a wide and consistent gap → tight binomial variance.
    y_pred, groups = _grouped_predictions(1500, p_a=0.9, p_b=0.1, seed=2)
    result = bootstrap_confidence_interval(
        demographic_parity_wrapper, y_pred, y_pred, groups,
        n_iterations=1000, random_state=2,
    )
    assert (result.ci_upper - result.ci_lower) < 0.05
    assert result.reliability_warning is None
    assert result.n_skipped_single_group == 0


# --------------------------------------------------------------------------- #
# Test 3 — wide CI on small, noisy data (bootstrap is sample-size sensitive)
# --------------------------------------------------------------------------- #
def test_wide_ci_on_small_sample():
    small_pred, small_groups = _grouped_predictions(10, p_a=0.9, p_b=0.1, seed=3)
    large_pred, large_groups = _grouped_predictions(1500, p_a=0.9, p_b=0.1, seed=3)

    small = bootstrap_confidence_interval(
        demographic_parity_wrapper, small_pred, small_pred, small_groups,
        n_iterations=1000, random_state=3,
    )
    large = bootstrap_confidence_interval(
        demographic_parity_wrapper, large_pred, large_pred, large_groups,
        n_iterations=1000, random_state=3,
    )
    small_width = small.ci_upper - small.ci_lower
    large_width = large.ci_upper - large.ci_lower

    assert small_width > large_width * 3   # dramatically wider, not marginally
    assert small_width > 0.15


# --------------------------------------------------------------------------- #
# Test 4 — all five metric wrappers produce valid BootstrapResults
# --------------------------------------------------------------------------- #
def test_all_five_wrappers_work():
    rng = np.random.default_rng(4)
    n = 400
    groups = pd.Series(["A"] * n + ["B"] * n)
    y_true = rng.integers(0, 2, 2 * n)
    # Predictions: mostly right, with a group-dependent error skew.
    flip = rng.random(2 * n) < np.where(groups.to_numpy() == "A", 0.15, 0.30)
    y_pred = np.where(flip, 1 - y_true, y_true)

    for name, wrapper in METRIC_WRAPPERS.items():
        result = bootstrap_confidence_interval(
            wrapper, y_true, y_pred, groups, n_iterations=300, random_state=4,
        )
        assert np.isfinite(result.point_estimate), name
        assert result.ci_lower <= result.point_estimate <= result.ci_upper, name
        assert result.n_valid_iterations > 0, name
        assert set(result.bootstrap_distribution_summary) == {"min", "max", "std"}


# --------------------------------------------------------------------------- #
# Test 5 — a fixed seed produces byte-identical results
# --------------------------------------------------------------------------- #
def test_fixed_seed_is_reproducible():
    y_pred, groups = _grouped_predictions(300, p_a=0.65, p_b=0.45, seed=5)
    kwargs = dict(n_iterations=250, random_state=99)

    a = bootstrap_confidence_interval(
        demographic_parity_wrapper, y_pred, y_pred, groups, **kwargs
    )
    b = bootstrap_confidence_interval(
        demographic_parity_wrapper, y_pred, y_pred, groups, **kwargs
    )
    assert a.point_estimate == b.point_estimate
    assert a.ci_lower == b.ci_lower
    assert a.ci_upper == b.ci_upper
    assert a.bootstrap_distribution_summary == b.bootstrap_distribution_summary


# --------------------------------------------------------------------------- #
# Test 6 — single-group resamples are skipped, tracked, and flagged
# --------------------------------------------------------------------------- #
def test_single_group_resample_is_handled():
    # 38 in A, 2 in B → a meaningful fraction of resamples will contain no B.
    groups = pd.Series(["A"] * 38 + ["B"] * 2)
    y_pred = np.array([0] * 19 + [1] * 19 + [1, 0])

    result = bootstrap_confidence_interval(
        demographic_parity_wrapper, y_pred, y_pred, groups,
        n_iterations=1000, random_state=1,
    )

    # did not crash, and produced a usable CI from the surviving iterations
    assert result.n_valid_iterations > 0
    assert np.isfinite(result.ci_lower) and np.isfinite(result.ci_upper)

    # skips tracked and reconciled
    assert result.n_skipped_single_group > 0
    assert result.skip_breakdown["single_group"] == result.n_skipped_single_group
    assert (
        result.n_valid_iterations
        + sum(result.skip_breakdown.values())
        == result.n_iterations
    )

    # the flag fired because the skipped fraction is well over 5%
    assert result.reliability_warning is not None
    assert "skipped" in result.reliability_warning


# =========================================================================== #
# Multiple-comparisons correction — Phase 2, gap 1.1
# =========================================================================== #
def _mk_bootstrap(
    ci_lower, ci_upper, std, n_valid=1000, n_skipped_single_group=0
) -> BootstrapResult:
    """Minimal BootstrapResult for reliability tests — only the fields
    assess_reliability reads need to be meaningful."""
    return BootstrapResult(
        point_estimate=(ci_lower + ci_upper) / 2,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=0.95,
        n_iterations=1000,
        bootstrap_distribution_summary={"min": ci_lower, "max": ci_upper, "std": std},
        n_valid_iterations=n_valid,
        n_skipped_single_group=n_skipped_single_group,
        reliability_warning=None,
        skip_breakdown={},
    )


# --------------------------------------------------------------------------- #
# Test 1 — Bonferroni on a known example
# --------------------------------------------------------------------------- #
def test_bonferroni_known_example():
    # 5 tests at alpha 0.05 → corrected alpha exactly 0.01.
    # p-values straddle 0.01: two below, three above.
    p_values = [0.001, 0.008, 0.02, 0.04, 0.30]
    result = apply_multiple_comparisons_correction(
        p_values, method="bonferroni", alpha=0.05
    )
    assert result.corrected_alpha == 0.01
    assert result.n_comparisons == 5
    assert result.significant == [True, True, False, False, False]
    # Bonferroni's decision is exactly p < corrected_alpha, element-wise.
    assert result.significant == [p < 0.01 for p in p_values]


# --------------------------------------------------------------------------- #
# Test 2 — Benjamini-Hochberg against the original 1995 paper
# --------------------------------------------------------------------------- #
def test_benjamini_hochberg_matches_1995_paper():
    # Benjamini & Hochberg (1995), "Controlling the False Discovery Rate",
    # JRSS-B 57(1):289-300, Table 1 — the Neuhaus et al. cardiac-trial
    # p-values. At alpha=0.05 the paper's BH procedure rejects the 4 smallest
    # hypotheses (vs 3 for Bonferroni).
    p_values = [
        0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
        0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000,
    ]
    result = apply_multiple_comparisons_correction(
        p_values, method="benjamini_hochberg", alpha=0.05
    )
    assert sum(result.significant) == 4
    assert result.significant[:4] == [True, True, True, True]
    assert result.significant[4:] == [False] * 11
    # corrected_alpha is a per-rank list, original order, aligned to p_values.
    assert isinstance(result.corrected_alpha, list)
    assert len(result.corrected_alpha) == 15
    assert result.corrected_alpha[0] == pytest.approx(1 / 15 * 0.05)
    assert result.corrected_alpha[-1] == pytest.approx(15 / 15 * 0.05)


# --------------------------------------------------------------------------- #
# Test 3 — BH is at least as permissive as Bonferroni on the same data
# --------------------------------------------------------------------------- #
def test_bh_is_less_conservative_than_bonferroni():
    # BH only diverges from Bonferroni when there is a *staircase* of small-ish
    # p-values that the step-up procedure can climb — not for any five numbers.
    # Here Bonferroni flags 1 (only p < 0.01); BH climbs to rank 4 and flags 4.
    p_values = [0.001, 0.012, 0.025, 0.04, 0.2]
    bonf = apply_multiple_comparisons_correction(p_values, method="bonferroni")
    bh = apply_multiple_comparisons_correction(
        p_values, method="benjamini_hochberg"
    )
    # BH never flags fewer than Bonferroni, and here flags strictly more.
    assert sum(bh.significant) >= sum(bonf.significant)
    assert sum(bh.significant) > sum(bonf.significant)
    # every Bonferroni rejection is also a BH rejection
    for b, h in zip(bonf.significant, bh.significant):
        assert not b or h


# --------------------------------------------------------------------------- #
# Test 4 — BH step-up: a hypothesis can be significant with p above its own crit
# --------------------------------------------------------------------------- #
def test_bh_step_up_can_reject_above_own_critical_value():
    # n=4, alpha=0.05. Sorted p = [0.001, 0.04, 0.045, 0.05].
    # Largest rank k with p_(k) <= (k/4)*0.05: rank 4, 0.05 <= 0.05 → k=4.
    # So ALL four are significant — including rank 2 (p=0.04 > crit 0.025) and
    # rank 3 (p=0.045 > crit 0.0375). This is correct BH, not a bug.
    p_values = [0.001, 0.04, 0.045, 0.05]
    result = apply_multiple_comparisons_correction(
        p_values, method="benjamini_hochberg", alpha=0.05
    )
    assert result.significant == [True, True, True, True]
    assert p_values[1] > result.corrected_alpha[1]   # 0.04 > 0.025
    assert result.significant[1] is True             # yet still rejected


# --------------------------------------------------------------------------- #
# Test 5 — invalid p-values raise, never get silently coerced
# --------------------------------------------------------------------------- #
def test_invalid_p_values_raise():
    with pytest.raises(ValueError, match="non-finite"):
        apply_multiple_comparisons_correction([0.01, float("nan"), 0.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        apply_multiple_comparisons_correction([0.01, 1.5])
    with pytest.raises(ValueError, match="non-empty"):
        apply_multiple_comparisons_correction([])


# =========================================================================== #
# Reliability scoring — Phase 2, gap 1.8
# =========================================================================== #
# --------------------------------------------------------------------------- #
# Test 6 — a small group blocks the verdict
# --------------------------------------------------------------------------- #
def test_small_group_size_blocks_verdict():
    boot = _mk_bootstrap(ci_lower=0.08, ci_upper=0.12, std=0.01)
    assessment = assess_reliability(boot, sample_sizes={"A": 500, "B": 15})
    assert assessment.tier == "insufficient_data"
    assert assessment.blocks_verdict is True
    assert any("only 15 samples" in r for r in assessment.reasons)


# --------------------------------------------------------------------------- #
# Test 7 — a high single-group skip rate blocks the verdict
# --------------------------------------------------------------------------- #
def test_high_skip_rate_blocks_verdict():
    # groups are adequately sized, but 12% of resamples collapsed to one group
    boot = _mk_bootstrap(
        ci_lower=0.05, ci_upper=0.15, std=0.02,
        n_valid=880, n_skipped_single_group=120,
    )
    assessment = assess_reliability(boot, sample_sizes={"A": 200, "B": 40})
    assert assessment.tier == "insufficient_data"
    assert assessment.blocks_verdict is True
    assert any("collapsed to a single group" in r for r in assessment.reasons)


# --------------------------------------------------------------------------- #
# Test 8 — a wide CI is unstable but not blocked
# --------------------------------------------------------------------------- #
def test_wide_ci_is_unstable_not_blocked():
    boot = _mk_bootstrap(ci_lower=0.02, ci_upper=0.30, std=0.03)  # width 0.28
    assessment = assess_reliability(boot, sample_sizes={"A": 300, "B": 280})
    assert assessment.tier == "unstable"
    assert assessment.blocks_verdict is False
    assert any("confidence interval spans" in r for r in assessment.reasons)


# --------------------------------------------------------------------------- #
# Test 9 — clean data is reliable, with a positive confirmation
# --------------------------------------------------------------------------- #
def test_clean_data_is_reliable():
    boot = _mk_bootstrap(ci_lower=0.09, ci_upper=0.11, std=0.008)
    assessment = assess_reliability(boot, sample_sizes={"A": 1000, "B": 950})
    assert assessment.tier == "reliable"
    assert assessment.blocks_verdict is False
    assert len(assessment.reasons) == 1
    assert "passed" in assessment.reasons[0]


# --------------------------------------------------------------------------- #
# Test 10 — multiple failing rules all appear in reasons
# --------------------------------------------------------------------------- #
def test_multiple_reasons_accumulate():
    # fails Rule 1 (group of 12) AND Rule 3 (CI width 0.4)
    boot = _mk_bootstrap(ci_lower=0.0, ci_upper=0.4, std=0.02)
    assessment = assess_reliability(boot, sample_sizes={"A": 400, "B": 12})
    assert assessment.tier == "insufficient_data"   # the more severe of the two
    assert any("only 12 samples" in r for r in assessment.reasons)
    assert any("confidence interval spans" in r for r in assessment.reasons)
    assert len(assessment.reasons) >= 2


# =========================================================================== #
# Simpson's paradox detection — Phase 2, gaps 1.5 / 9.9
# =========================================================================== #
# --------------------------------------------------------------------------- #
# Test 1 — masked bias: aggregate passes, one stratum fails, same direction
# --------------------------------------------------------------------------- #
def test_simpsons_masked_bias_detected():
    # Stratum X (n=60): A 0.80 vs B 0.30 → gap 0.50, FAIL, A favoured.
    # Stratum Y (n=600): A 0.50 vs B 0.51 → gap 0.01, pass (tiny, B edge — below
    #   the 0.10 reversal bar so it does not count as a reversal).
    # Aggregate: A 0.527 vs B 0.491 → gap ~0.036, PASS.
    y_pred, sf, sb = _labelled([
        ("A", "X", 24, 30), ("B", "X", 9, 30),
        ("A", "Y", 150, 300), ("B", "Y", 153, 300),
    ])
    result = detect_simpsons_paradox(y_pred, y_pred, sf, sb)

    assert isinstance(result, SimpsonsParadoxResult)
    assert result.aggregate_status == "pass"
    assert result.paradox_detected is True
    assert result.paradox_type == "masked_bias"
    x = next(s for s in result.stratum_results if s["stratum_label"] == "X")
    assert x["status"] == "fail" and x["excluded"] is False


# --------------------------------------------------------------------------- #
# Test 2 — no paradox: aggregate and every stratum agree
# --------------------------------------------------------------------------- #
def test_simpsons_no_paradox_on_consistent_data():
    y_pred, sf, sb = _labelled([
        ("A", "X", 165, 300), ("B", "X", 150, 300),   # gap 0.05, pass
        ("A", "Y", 156, 300), ("B", "Y", 150, 300),   # gap 0.02, pass
    ])
    result = detect_simpsons_paradox(y_pred, y_pred, sf, sb)

    assert result.paradox_detected is False
    assert result.paradox_type is None
    assert "No Simpson's paradox" in result.explanation
    assert "masking" not in result.explanation and "reverses" not in result.explanation


# --------------------------------------------------------------------------- #
# Test 3 — a below-min stratum is shown but cannot trigger the flag
# --------------------------------------------------------------------------- #
def test_simpsons_small_stratum_excluded_not_dropped():
    # Stratum Z (n=20) would fail hard (A all positive, B all negative) but is
    # below min_stratum_size=30, so it must not trigger paradox_detected.
    y_pred, sf, sb = _labelled([
        ("A", "X", 150, 300), ("B", "X", 150, 300),   # consistent, pass
        ("A", "Z", 10, 10), ("B", "Z", 0, 10),        # gap 1.0 but n=20
    ])
    result = detect_simpsons_paradox(y_pred, y_pred, sf, sb)

    z = next(s for s in result.stratum_results if s["stratum_label"] == "Z")
    assert z["excluded"] is True
    assert z["metric_value"] is not None          # still computed, for transparency
    assert z["metric_value"] > 0.10               # it WOULD have failed
    assert result.paradox_detected is False       # ...but it cannot trigger


# --------------------------------------------------------------------------- #
# Test 4 — genuine sign reversal (and reversal skipped for a custom metric_fn)
# --------------------------------------------------------------------------- #
def test_simpsons_sign_reversal_detected():
    # Aggregate favours A; stratum Y favours B; both numerically fail.
    y_pred, sf, sb = _labelled([
        ("A", "X", 135, 150), ("B", "X", 75, 150),    # A 0.90 vs B 0.50
        ("A", "Y", 15, 50), ("B", "Y", 35, 50),       # A 0.30 vs B 0.70
    ])
    result = detect_simpsons_paradox(y_pred, y_pred, sf, sb)
    assert result.paradox_type == "reversal"
    assert "reverses" in result.explanation

    # Same data, custom metric_fn → reversal must NOT be evaluated.
    custom = detect_simpsons_paradox(
        y_pred, y_pred, sf, sb, metric_fn=equalized_odds_wrapper
    )
    assert custom.paradox_type != "reversal"
    assert "not evaluated for the supplied custom metric" in custom.explanation


# --------------------------------------------------------------------------- #
# Test 5 — explanation is alarming only when a paradox is real
# --------------------------------------------------------------------------- #
def test_simpsons_explanation_tone():
    masked = _labelled([
        ("A", "X", 24, 30), ("B", "X", 9, 30),
        ("A", "Y", 150, 300), ("B", "Y", 153, 300),
    ])
    r_paradox = detect_simpsons_paradox(masked[0], masked[0], masked[1], masked[2])
    assert r_paradox.paradox_detected is True
    assert len(r_paradox.explanation) > 40
    assert "masking bias" in r_paradox.explanation

    clean = _labelled([
        ("A", "X", 156, 300), ("B", "X", 150, 300),
        ("A", "Y", 153, 300), ("B", "Y", 150, 300),
    ])
    r_clean = detect_simpsons_paradox(clean[0], clean[0], clean[1], clean[2])
    assert r_clean.paradox_detected is False
    assert "No Simpson's paradox" in r_clean.explanation


# =========================================================================== #
# Metric tension detection — Phase 2, gap 1.9
# =========================================================================== #
def _base_rate_data(rate_a, rate_b, n=400):
    """y_true with group A at positive rate rate_a, B at rate_b."""
    a_pos = round(rate_a * n)
    b_pos = round(rate_b * n)
    y_true = np.array([1] * a_pos + [0] * (n - a_pos) + [1] * b_pos + [0] * (n - b_pos))
    sf = pd.Series(["A"] * n + ["B"] * n)
    return y_true, sf


# --------------------------------------------------------------------------- #
# Test 1 — base rates differ + DP fails while PP passes → tension detected
# --------------------------------------------------------------------------- #
def test_tension_detected_when_base_rates_differ():
    y_true, sf = _base_rate_data(0.70, 0.30)
    results = _five_results(
        demographic_parity_difference="fail",
        predictive_parity_difference="pass",
    )
    out = detect_metric_tensions(y_true, sf, results)

    assert isinstance(out, MetricTensionResult)
    assert out.base_rates_differ_significantly is True
    assert len(out.tensions) == 1
    assert out.tensions[0]["pattern"] == "demographic_parity_fails_predictive_parity_passes"
    assert "base-rate" in out.tensions[0]["explanation"] or "base rate" in out.tensions[0]["explanation"]
    assert out.unexplained_disagreement is False


# --------------------------------------------------------------------------- #
# Test 2 — same metric pattern, similar base rates → no tension, flagged genuine
# --------------------------------------------------------------------------- #
def test_no_tension_when_base_rates_similar():
    y_true, sf = _base_rate_data(0.50, 0.50)
    results = _five_results(
        demographic_parity_difference="fail",
        predictive_parity_difference="pass",
    )
    out = detect_metric_tensions(y_true, sf, results)

    assert out.base_rates_differ_significantly is False
    assert out.tensions == []                       # not manufactured
    assert out.unexplained_disagreement is True     # the boolean downstream branches on
    assert "genuine finding" in out.fairness_definition_note


# --------------------------------------------------------------------------- #
# Test 3 — all metrics agree → no tensions, no unexplained disagreement
# --------------------------------------------------------------------------- #
def test_no_tension_when_all_metrics_agree():
    y_true, sf = _base_rate_data(0.70, 0.30)         # base rates differ...
    all_pass = detect_metric_tensions(y_true, sf, _five_results())
    assert all_pass.tensions == []
    assert all_pass.unexplained_disagreement is False

    all_fail = detect_metric_tensions(
        y_true, sf,
        _five_results(
            demographic_parity_difference="fail",
            equalized_odds_difference="fail",
            equal_opportunity_difference="fail",
            predictive_parity_difference="fail",
            overall_accuracy_floor="fail",
        ),
    )
    assert all_fail.tensions == []
    assert all_fail.unexplained_disagreement is False


# --------------------------------------------------------------------------- #
# Test 4 — base_rates reflects the real positive rate per group
# --------------------------------------------------------------------------- #
def test_base_rates_are_computed_correctly():
    # A: 30 positive of 50 → 0.6.  B: 10 positive of 50 → 0.2.
    y_true = np.array([1] * 30 + [0] * 20 + [1] * 10 + [0] * 40)
    sf = pd.Series(["A"] * 50 + ["B"] * 50)
    out = detect_metric_tensions(y_true, sf, _five_results())

    assert out.base_rates == {"A": 0.6, "B": 0.2}
    assert out.base_rate_difference == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Test 5 — fairness_definition_note names the passing and failing metrics
# --------------------------------------------------------------------------- #
def test_fairness_definition_note_names_metrics():
    y_true, sf = _base_rate_data(0.70, 0.30)
    results = _five_results(
        demographic_parity_difference="pass",
        equal_opportunity_difference="pass",
        equalized_odds_difference="fail",
        predictive_parity_difference="fail",
        overall_accuracy_floor="fail",
    )
    note = detect_metric_tensions(y_true, sf, results).fairness_definition_note

    satisfies, _, rest = note.partition("It does not satisfy:")
    assert "demographic_parity_difference" in satisfies
    assert "equal_opportunity_difference" in satisfies
    assert "predictive_parity_difference" in rest
    assert "GAP_CHECKLIST 9.13" in note        # the overall_accuracy_floor caveat
