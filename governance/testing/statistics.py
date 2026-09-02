"""Statistical rigour layer — Phase 2 (Component 2.1).

Two capabilities, both operating on a single protected attribute:

1. **Significance testing** (``significance_test``) — *could this
   demographic-parity gap in the model's predictions plausibly be random noise?*
   Four tests: chi-squared, Fisher's exact, two-proportion z-test, permutation;
   an ``auto`` selector; an always-on permutation cross-check.

2. **Bootstrap confidence intervals** (``bootstrap_confidence_interval`` +
   five metric wrappers) — *if we resampled this test set many times, where
   would the true gap most likely fall?* Generic over any
   ``(y_true, y_pred, sensitive_features) -> float`` metric.

This module is a **sibling** of ``bias.py`` — it is called alongside
``BiasTestSuite``, never wraps it, and never imports from it (and ``bias.py``
never imports from here). Every fairlearn / sklearn metric used here is imported
directly from its library so the numbers match exactly what ``BiasTestSuite``
reports.

3. **Multiple-comparison correction** (``apply_multiple_comparisons_correction``)
   — five simultaneous fairness tests inflate the family-wise false-positive
   rate from 5% to ≈23%. Bonferroni (conservative, default) and
   Benjamini-Hochberg FDR (step-up), both hand-implemented and auditable.

4. **Reliability scoring** (``assess_reliability``) — three tiers
   (reliable / unstable / insufficient_data). ``insufficient_data`` blocks a
   compliance verdict entirely; ``unstable`` degrades confidence but still
   reports.

This module is a **sibling** of ``bias.py`` — it is called alongside
``BiasTestSuite``, never wraps it, and never imports from it (and ``bias.py``
never imports from here). Every fairlearn / sklearn metric used here is imported
directly from its library so the numbers match exactly what ``BiasTestSuite``
reports.

Explicitly NOT in scope here (later pieces of Phase 2 / Phase 4):
  * wiring these results onto ``TestResult`` rows — a separate Phase 2 step
  * continuous or multi-class model outputs — Phase 4 (``regression.py`` etc.)
  * general R×C chi-squared — would be its own component with its own tests

See docs/BUILD_PLAN.md → "PHASE 2" and docs/GAP_CHECKLIST.md → Category 1
(1.4 significance, 1.3 confidence intervals, 1.1 correction, 1.8 reliability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from scipy.stats import chi2_contingency, fisher_exact, norm
from sklearn.metrics import accuracy_score, precision_score

__all__ = [
    "SignificanceResult",
    "significance_test",
    "VALID_METHODS",
    "BootstrapResult",
    "bootstrap_confidence_interval",
    "demographic_parity_wrapper",
    "equalized_odds_wrapper",
    "equal_opportunity_wrapper",
    "predictive_parity_wrapper",
    "individual_fairness_wrapper",
    "METRIC_WRAPPERS",
    "CorrectionResult",
    "apply_multiple_comparisons_correction",
    "ReliabilityAssessment",
    "assess_reliability",
    "CORRECTION_METHODS",
]

# --------------------------------------------------------------------------- #
# Module constants
# --------------------------------------------------------------------------- #
ALPHA = 0.05                     # raw significance level (before any correction)
MIN_EXPECTED_CELL_COUNT = 5      # standard chi-squared validity rule
Z_TEST_MIN_COUNT = 5            # n * p and n * (1 - p) must exceed this per group
DEFAULT_N_PERMUTATIONS = 1_000
POSITIVE_LABEL = 1              # y_pred is assumed binary {0, 1}
_FP_TOLERANCE = 1e-12          # float slack for "as large or larger" comparison

VALID_METHODS = ("auto", "chi_squared", "fishers_exact", "z_test", "permutation")
_PRIMARY_METHODS = ("chi_squared", "fishers_exact", "z_test", "permutation")
_TWO_GROUP_ONLY = ("chi_squared", "fishers_exact", "z_test")


@dataclass
class SignificanceResult:
    """Outcome of one significance test on one protected attribute.

    Attributes
    ----------
    test_used : str
        The primary test actually run — one of ``_PRIMARY_METHODS``. When
        ``method="auto"`` this is the test the selector chose.
    statistic : float
        Test-dependent:
          * ``chi_squared``  → the χ² statistic
          * ``fishers_exact``→ the odds ratio
          * ``z_test``       → the z statistic
          * ``permutation``  → the observed demographic-parity difference
    p_value : float
        Two-sided p-value from the primary test.
    significant : bool
        ``p_value < ALPHA`` (0.05). This is the raw call — multiple-comparison
        correction happens in a later component and is not applied here.
    assumptions_met : bool
        Whether the primary test was used inside its region of validity:
          * ``chi_squared``  → every expected cell count ≥ 5
          * ``z_test``       → n·p̂ and n·(1−p̂) both > 5 for each group
          * ``fishers_exact``/``permutation`` when chosen *explicitly* → True
            (both are assumption-free / exact)
          * ``fishers_exact`` reached via ``auto`` fallback → **False**, because
            the fallback only happens *because* chi-squared's assumptions failed;
            ``detail["assumption_notes"]`` explains, and notes that the Fisher
            p-value itself is exact and trustworthy.
        A ``False`` here means: treat ``p_value`` and ``significant`` with
        suspicion — usually biased toward over-stating significance.
    sample_sizes : dict
        ``{group_label: n}`` in first-appearance order.
    detail : dict
        Populated as available:
          * ``group_positive_rates``     – {group: P(pred == positive)}
          * ``expected_frequencies``     – 2×2 list, whenever chi-squared is
            computed (as the primary test or as the ``auto`` decision input)
          * ``permutation_p_value``      – secondary cross-check. Populated on
            **every** call unless ``cross_check=False`` (not just ``auto`` mode).
          * ``permutation_observed_statistic`` / ``n_permutations``
          * ``cross_check_divergence``   – |primary p − permutation p|, when both
            exist and the primary test is not itself the permutation test
          * ``assumption_notes``         – human-readable string on any failure
    """

    test_used: str
    statistic: float
    p_value: float
    significant: bool
    assumptions_met: bool
    sample_sizes: dict
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.test_used not in _PRIMARY_METHODS:
            raise ValueError(
                f"SignificanceResult.test_used must be one of {_PRIMARY_METHODS}, "
                f"got {self.test_used!r}"
            )
        self.statistic = float(self.statistic)
        self.p_value = float(self.p_value)
        self.significant = bool(self.significant)
        self.assumptions_met = bool(self.assumptions_met)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def _validate_and_prepare(y_pred, sensitive_features, method):
    if method not in VALID_METHODS:
        raise ValueError(
            f"method must be one of {VALID_METHODS}, got {method!r}"
        )

    y_pred = np.asarray(y_pred)
    observed_labels = set(np.unique(y_pred).tolist())
    if not observed_labels <= {0, 1}:
        raise ValueError(
            "significance_test expects a binary y_pred with labels in {0, 1}; "
            f"found labels {sorted(observed_labels)}. Continuous or multi-class "
            "outputs are handled by Phase 4 metrics, not this component."
        )
    y_pred = y_pred.astype(int)

    if not isinstance(sensitive_features, pd.Series):
        sensitive_features = pd.Series(list(sensitive_features))
    sensitive_features = sensitive_features.reset_index(drop=True)

    if len(sensitive_features) != len(y_pred):
        raise ValueError(
            f"y_pred (n={len(y_pred)}) and sensitive_features "
            f"(n={len(sensitive_features)}) must be the same length"
        )

    groups = [str(g) for g in pd.unique(sensitive_features)]
    if len(groups) < 2:
        raise ValueError(
            "significance testing needs at least two groups in "
            f"sensitive_features; found {len(groups)}: {groups}"
        )

    group_str = sensitive_features.astype(str).to_numpy()
    return y_pred, group_str, groups


def _group_stats(y_pred, group_str, groups):
    sample_sizes: dict[str, int] = {}
    positive_rates: dict[str, float] = {}
    for g in groups:
        mask = group_str == g
        n = int(mask.sum())
        sample_sizes[g] = n
        positive_rates[g] = (
            float(np.mean(y_pred[mask] == POSITIVE_LABEL)) if n else 0.0
        )
    return sample_sizes, positive_rates


def _contingency_2x2(y_pred, group_str, groups):
    """Rows = the two groups (first-appearance order);
    columns = [predicted positive, predicted negative]."""
    rows = []
    for g in groups[:2]:
        mask = group_str == g
        pos = int(np.sum(y_pred[mask] == POSITIVE_LABEL))
        neg = int(mask.sum()) - pos
        rows.append([pos, neg])
    return np.array(rows, dtype=float)


# --------------------------------------------------------------------------- #
# Individual tests
# --------------------------------------------------------------------------- #
def _run_chi_squared(table):
    """scipy.stats.chi2_contingency with its defaults (Yates correction on for
    2×2) so results match a direct chi2_contingency(crosstab) call.

    Returns (statistic, p_value, expected_frequencies, assumptions_met, note).
    """
    try:
        chi2, p, _dof, expected = chi2_contingency(table)
    except ValueError as exc:
        # Degenerate table (a zero row/column total → zero expected cell).
        return (
            0.0,
            1.0,
            [[0.0, 0.0], [0.0, 0.0]],
            False,
            f"chi-squared could not be computed on a degenerate table: {exc}",
        )
    min_expected = float(np.min(expected))
    assumptions_met = min_expected >= MIN_EXPECTED_CELL_COUNT
    note = (
        None
        if assumptions_met
        else (
            f"chi-squared assumption violated: smallest expected cell count is "
            f"{min_expected:.2f} (< {MIN_EXPECTED_CELL_COUNT})"
        )
    )
    return float(chi2), float(p), expected.tolist(), assumptions_met, note


def _run_fishers_exact(table):
    """scipy.stats.fisher_exact, two-sided. Strictly 2×2.

    Returns (odds_ratio, p_value).
    """
    result = fisher_exact(table, alternative="two-sided")
    odds_ratio, p_value = float(result[0]), float(result[1])
    return odds_ratio, p_value


def _run_z_test(table, groups):
    """Two-proportion z-test, computed manually.

    table rows are the two groups, columns [positive, negative].
    Returns (z, p_value, assumptions_met, note).
    """
    (pos_a, neg_a), (pos_b, neg_b) = table
    n_a, n_b = pos_a + neg_a, pos_b + neg_b

    if n_a == 0 or n_b == 0:
        return 0.0, 1.0, False, "a group has zero samples; z-test undefined"

    p1, p2 = pos_a / n_a, pos_b / n_b
    p_pooled = (pos_a + pos_b) / (n_a + n_b)
    se = np.sqrt(p_pooled * (1.0 - p_pooled) * (1.0 / n_a + 1.0 / n_b))

    # Normal-approximation validity: n·p̂ and n·(1−p̂) > 5 for BOTH groups,
    # using the pooled proportion (the value used in the SE).
    checks = [
        n_a * p_pooled, n_a * (1.0 - p_pooled),
        n_b * p_pooled, n_b * (1.0 - p_pooled),
    ]
    assumptions_met = all(c > Z_TEST_MIN_COUNT for c in checks)
    note = (
        None
        if assumptions_met
        else (
            "z-test normal approximation weak: min(n·p̂, n·(1−p̂)) across groups "
            f"is {min(checks):.2f} (need > {Z_TEST_MIN_COUNT})"
        )
    )

    if se == 0.0:
        # Both groups have an identical, degenerate prediction rate (all 0 or
        # all 1). No variance → no test.
        return 0.0, 1.0, False, "pooled standard error is zero; z-test undefined"

    z = (p1 - p2) / se
    p_value = 2.0 * norm.sf(abs(z))  # sf == 1 - cdf, more accurate in the tail
    return float(z), float(p_value), assumptions_met, note


def _permutation_p_value(y_true, y_pred, group_str, n_permutations, rng):
    """Assumption-free test on the demographic-parity difference.

    1. observed = |max group positive-rate − min group positive-rate|
       (fairlearn.metrics.demographic_parity_difference, so it matches
       BiasTestSuite exactly — for 2 or more groups)
    2. shuffle the group labels, breaking any real group↔outcome association
    3. recompute the difference
    4. repeat n_permutations times
    5. p = (1 + #{shuffled ≥ observed}) / (1 + n_permutations)

    Step 5 uses add-one smoothing (the standard permutation-test estimator):
    it can never report p == 0, which would be an indefensible "this could not
    happen by chance" claim in a compliance document.

    Returns (observed_statistic, p_value).
    """
    observed = float(
        demographic_parity_difference(
            y_true, y_pred, sensitive_features=group_str
        )
    )
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(group_str)
        stat = demographic_parity_difference(
            y_true, y_pred, sensitive_features=shuffled
        )
        if stat >= observed - _FP_TOLERANCE:
            at_least_as_extreme += 1
    p_value = (1 + at_least_as_extreme) / (1 + n_permutations)
    return observed, float(p_value)


# --------------------------------------------------------------------------- #
# auto selection
# --------------------------------------------------------------------------- #
def _select_auto(table, n_groups):
    """Return (chosen_method, expected_frequencies_or_None, decision_note)."""
    if n_groups > 2:
        return (
            "permutation",
            None,
            "permutation test selected: more than two groups "
            "(chi-squared / Fisher / z-test are two-group only in this module)",
        )
    try:
        _chi2, _p, _dof, expected = chi2_contingency(table)
    except ValueError:
        return (
            "fishers_exact",
            None,
            "Fisher's exact selected: contingency table is degenerate for "
            "chi-squared",
        )
    min_expected = float(np.min(expected))
    if min_expected < MIN_EXPECTED_CELL_COUNT:
        return (
            "fishers_exact",
            expected.tolist(),
            f"Fisher's exact selected: smallest expected cell count is "
            f"{min_expected:.2f} (< {MIN_EXPECTED_CELL_COUNT}), so chi-squared "
            f"is unreliable. Fisher's exact has no minimum-sample assumption.",
        )
    return (
        "chi_squared",
        expected.tolist(),
        f"chi-squared selected: smallest expected cell count is "
        f"{min_expected:.2f} (≥ {MIN_EXPECTED_CELL_COUNT})",
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def significance_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    method: str = "auto",
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    random_state: int | None = None,
    cross_check: bool = True,
) -> SignificanceResult:
    """Test whether the demographic-parity gap in ``y_pred`` across the groups
    in ``sensitive_features`` is distinguishable from chance.

    Parameters
    ----------
    y_true
        Accepted for interface parity with the rest of the testing engine and
        for future outcome-conditioned tests. The demographic-parity
        significance test does not use it; ``None`` is tolerated.
    y_pred
        Binary predictions, labels in ``{0, 1}``. Anything else raises
        ``ValueError`` — continuous / multi-class outputs are Phase 4's job.
    sensitive_features
        Group label per row. Needs at least two distinct values.
    method
        One of ``VALID_METHODS``. ``"auto"`` picks:
          * > 2 groups                     → ``permutation``
          * 2 groups, any expected cell < 5 → ``fishers_exact``
          * 2 groups, otherwise            → ``chi_squared``
        ``"chi_squared"``, ``"fishers_exact"`` and ``"z_test"`` are two-group
        only here; calling them with > 2 groups raises ``ValueError`` pointing
        at ``permutation``.
    n_permutations
        Iterations for the permutation test / cross-check. Default 1,000.
    random_state
        Seed for the permutation shuffling. Pass a fixed int in tests for
        reproducibility; leave ``None`` in production for true randomness
        (per BUILD_PLAN).
    cross_check
        When ``True`` (default) a permutation p-value is computed and stored in
        ``detail["permutation_p_value"]`` on **every** call, whatever the
        primary test — an assumption-free second opinion that is most valuable
        exactly when a caller has picked a parametric test explicitly. Set
        ``False`` only as a deliberate, review-visible opt-out.

    Returns
    -------
    SignificanceResult
    """
    y_pred, group_str, groups = _validate_and_prepare(
        y_pred, sensitive_features, method
    )
    n_groups = len(groups)
    sample_sizes, positive_rates = _group_stats(y_pred, group_str, groups)

    if method in _TWO_GROUP_ONLY and n_groups > 2:
        raise ValueError(
            f"method={method!r} supports exactly two groups in this module; "
            f"got {n_groups} ({groups}). Use method='permutation' for more than "
            f"two groups."
        )

    rng = np.random.default_rng(random_state)
    detail: dict = {"group_positive_rates": positive_rates}

    # 2×2 table is only meaningful / needed for the two-group parametric tests.
    table = _contingency_2x2(y_pred, group_str, groups) if n_groups == 2 else None

    # ---- pick the primary test ------------------------------------------- #
    if method == "auto":
        test_used, expected_from_auto, decision_note = _select_auto(table, n_groups)
        detail["auto_selection"] = decision_note
        if expected_from_auto is not None:
            detail["expected_frequencies"] = expected_from_auto
    else:
        test_used = method
        expected_from_auto = None

    auto_fisher_fallback = (
        method == "auto"
        and test_used == "fishers_exact"
        and n_groups == 2
    )

    # ---- run the primary test ------------------------------------------- #
    assumption_notes: list[str] = []

    if test_used == "chi_squared":
        stat, p_value, expected, assumptions_met, note = _run_chi_squared(table)
        detail["expected_frequencies"] = expected
        if note:
            assumption_notes.append(note)

    elif test_used == "fishers_exact":
        stat, p_value = _run_fishers_exact(table)
        if auto_fisher_fallback:
            # Reached only because chi-squared's assumptions failed. Report that
            # as assumptions_met=False so a caller who ignores the flag is not
            # misled — while making clear the Fisher p-value itself is exact.
            assumptions_met = False
            assumption_notes.append(
                "Primary preference (chi-squared) was invalid here; auto "
                "substituted Fisher's exact, which is exact and needs no "
                "minimum sample size — this p_value is trustworthy."
            )
        else:
            assumptions_met = True

    elif test_used == "z_test":
        stat, p_value, assumptions_met, note = _run_z_test(table, groups)
        if note:
            assumption_notes.append(note)

    elif test_used == "permutation":
        stat, p_value = _permutation_p_value(
            y_true, y_pred, group_str, n_permutations, rng
        )
        assumptions_met = True
        detail["permutation_observed_statistic"] = stat
        detail["n_permutations"] = n_permutations

    else:  # pragma: no cover - guarded by _validate_and_prepare
        raise ValueError(f"unhandled method {test_used!r}")

    # ---- always-on permutation cross-check ----------------------------- #
    if cross_check and test_used != "permutation":
        # Fresh generator so the cross-check is independent of any permutation
        # primary (there is none on this branch) and reproducible from the seed.
        cc_rng = np.random.default_rng(random_state)
        observed, perm_p = _permutation_p_value(
            y_true, y_pred, group_str, n_permutations, cc_rng
        )
        detail["permutation_p_value"] = perm_p
        detail["permutation_observed_statistic"] = observed
        detail["n_permutations"] = n_permutations
        detail["cross_check_divergence"] = abs(p_value - perm_p)
    elif test_used == "permutation":
        detail["permutation_p_value"] = p_value

    if assumption_notes:
        detail["assumption_notes"] = " ".join(assumption_notes)

    return SignificanceResult(
        test_used=test_used,
        statistic=stat,
        p_value=p_value,
        significant=p_value < ALPHA,
        assumptions_met=assumptions_met,
        sample_sizes=sample_sizes,
        detail=detail,
    )


# =========================================================================== #
# Bootstrap confidence intervals (Phase 2 — gap 1.3)
# =========================================================================== #

DEFAULT_BOOTSTRAP_ITERATIONS = 1_000
SINGLE_GROUP_SKIP_THRESHOLD = 0.05   # skipped fraction above this flags the CI

# metric_fn contract: (y_true, y_pred, sensitive_features) -> float
MetricFn = Callable[..., float]


@dataclass
class BootstrapResult:
    """A bootstrap 95% (or other) confidence interval for one fairness metric.

    Attributes
    ----------
    point_estimate : float
        ``metric_fn`` evaluated once on the full, unresampled data. This is the
        number a report shows; ``nan`` only if the metric could not be computed
        even on the full data.
    ci_lower, ci_upper : float
        The ``(1-c)/2`` and ``1-(1-c)/2`` percentiles of the bootstrap
        distribution. ``nan`` / ``nan`` when every iteration was skipped.
    confidence_level : float
        As requested (default 0.95).
    n_iterations : int
        Iterations requested.
    bootstrap_distribution_summary : dict
        ``{"min", "max", "std"}`` over the *valid* resample values — for
        diagnosing an oddly shaped distribution, not for reporting. (No mean or
        median on purpose: ``point_estimate`` is unambiguously "the" number.)
    n_valid_iterations : int
        Iterations that produced a usable value.
    n_skipped_single_group : int
        Resamples that by chance contained only one group (metric undefined).
    reliability_warning : str | None
        ``None`` unless the skipped fraction exceeds
        ``SINGLE_GROUP_SKIP_THRESHOLD`` (5%) or every iteration failed. When set,
        the text is specific: how many were skipped, why, and what it means.
        Component 2.3's reliability scoring consumes this.
    skip_breakdown : dict
        ``{reason: count}`` for every skipped iteration — ``"single_group"``,
        ``"non_finite_value"``, or an exception type name
        (e.g. ``"ValueError"``). Always present (may be empty). Lets a developer
        chasing a high skip rate tell the causes apart without re-running.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_iterations: int
    bootstrap_distribution_summary: dict
    n_valid_iterations: int
    n_skipped_single_group: int
    reliability_warning: str | None
    skip_breakdown: dict = field(default_factory=dict)


def _bump(d: dict, key: str) -> None:
    d[key] = d.get(key, 0) + 1


def bootstrap_confidence_interval(
    metric_fn: MetricFn,
    y_true,
    y_pred,
    sensitive_features,
    n_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    confidence_level: float = 0.95,
    random_state: int | None = None,
) -> BootstrapResult:
    """Percentile bootstrap CI for any fairness metric.

    ``metric_fn`` must accept ``(y_true, y_pred, sensitive_features)`` and return
    a single float. Use the five wrappers in this module (or
    ``METRIC_WRAPPERS``) for the BiasTestSuite metrics.

    Resampling draws **one** vector of row indices with replacement per
    iteration and applies it to all three arrays, so each synthetic row keeps
    its real ``(group, prediction, label)`` triple — the joint relationship the
    metric measures is preserved. ``sensitive_features`` is converted to a
    positional array first, so a pandas index cannot misalign the draw.

    Degenerate resamples (only one group present) and any ``metric_fn`` failure
    are **skipped, not fatal** — counted in ``skip_breakdown``. If more than 5%
    of iterations are skipped the result carries a ``reliability_warning``; if
    all fail, the CI is ``nan`` with a specific warning rather than an exception.

    Parameters
    ----------
    random_state : int | None
        Fixed int → identical result every call (seeding is wired through a
        single ``np.random.default_rng``). ``None`` → true randomness.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must be in (0, 1), got {confidence_level}"
        )
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be >= 1, got {n_iterations}")

    yp = np.asarray(y_pred)
    n = len(yp)
    if n == 0:
        raise ValueError("cannot bootstrap an empty dataset")
    yt = None if y_true is None else np.asarray(y_true)
    sf = np.asarray(sensitive_features)
    if len(sf) != n or (yt is not None and len(yt) != n):
        raise ValueError(
            "y_true, y_pred and sensitive_features must all be the same length"
        )

    # ---- point estimate on the full data -------------------------------- #
    try:
        point_estimate = float(metric_fn(yt, yp, sf))
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        point_estimate = float("nan")
        point_note = f"point estimate could not be computed ({type(exc).__name__})"
    else:
        point_note = None

    # ---- the bootstrap loop ------------------------------------------- #
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    skip_breakdown: dict[str, int] = {}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        sf_bs = sf[idx]
        if np.unique(sf_bs).size < 2:
            _bump(skip_breakdown, "single_group")
            continue
        try:
            value = float(
                metric_fn(None if yt is None else yt[idx], yp[idx], sf_bs)
            )
        except Exception as exc:  # noqa: BLE001 - one bad resample must not kill the CI
            _bump(skip_breakdown, type(exc).__name__)
            continue
        if not np.isfinite(value):
            _bump(skip_breakdown, "non_finite_value")
            continue
        values.append(value)

    n_valid = len(values)
    n_skipped_total = n_iterations - n_valid
    n_skipped_single_group = skip_breakdown.get("single_group", 0)

    lower_pct = (1.0 - confidence_level) / 2.0 * 100.0
    upper_pct = 100.0 - lower_pct

    if n_valid == 0:
        ci_lower = ci_upper = float("nan")
        summary = {"min": float("nan"), "max": float("nan"), "std": float("nan")}
        reliability_warning = (
            f"all {n_iterations} iterations failed — CI could not be computed, "
            f"sample size or class balance likely insufficient for this metric "
            f"(skip breakdown: {skip_breakdown})"
        )
    else:
        arr = np.asarray(values, dtype=float)
        ci_lower, ci_upper = (
            float(x) for x in np.percentile(arr, [lower_pct, upper_pct])
        )
        summary = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }
        skipped_fraction = n_skipped_total / n_iterations
        if skipped_fraction > SINGLE_GROUP_SKIP_THRESHOLD:
            reliability_warning = (
                f"{n_skipped_total}/{n_iterations} resamples ({skipped_fraction:.1%}) "
                f"were skipped — CI may be unreliable due to small sample size or "
                f"group imbalance (skip breakdown: {skip_breakdown})"
            )
        else:
            reliability_warning = None

    # Fold in a point-estimate failure note, if any.
    if point_note:
        reliability_warning = (
            point_note if reliability_warning is None
            else f"{point_note}; {reliability_warning}"
        )

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_iterations=n_iterations,
        bootstrap_distribution_summary=summary,
        n_valid_iterations=n_valid,
        n_skipped_single_group=n_skipped_single_group,
        reliability_warning=reliability_warning,
        skip_breakdown=skip_breakdown,
    )


# --------------------------------------------------------------------------- #
# Metric wrappers — adapt each BiasTestSuite metric to the metric_fn shape.
# These call fairlearn / sklearn directly; they never import from bias.py.
# --------------------------------------------------------------------------- #
def _require_y_true(y_true, name: str):
    if y_true is None:
        raise ValueError(f"{name} needs y_true — it conditions on the true label")
    return np.asarray(y_true)


def demographic_parity_wrapper(y_true, y_pred, sensitive_features) -> float:
    """|selection-rate gap| across groups. y_true is unused (tolerates None)."""
    yt = y_pred if y_true is None else y_true
    return float(
        abs(
            demographic_parity_difference(
                yt, y_pred, sensitive_features=np.asarray(sensitive_features)
            )
        )
    )


def equalized_odds_wrapper(y_true, y_pred, sensitive_features) -> float:
    yt = _require_y_true(y_true, "equalized_odds_wrapper")
    return float(
        abs(
            equalized_odds_difference(
                yt, y_pred, sensitive_features=np.asarray(sensitive_features)
            )
        )
    )


def equal_opportunity_wrapper(y_true, y_pred, sensitive_features) -> float:
    yt = _require_y_true(y_true, "equal_opportunity_wrapper")
    return float(
        abs(
            equal_opportunity_difference(
                yt, y_pred, sensitive_features=np.asarray(sensitive_features)
            )
        )
    )


def predictive_parity_wrapper(y_true, y_pred, sensitive_features) -> float:
    """max - min group precision. Mirrors bias.py exactly, including
    ``zero_division=0`` (a group with no predicted positives scores 0.0, it
    does not raise)."""
    yt = _require_y_true(y_true, "predictive_parity_wrapper")
    yp = np.asarray(y_pred)
    groups = np.asarray(sensitive_features)
    precisions = []
    for g in pd.unique(pd.Series(groups)):
        mask = groups == g
        precisions.append(
            float(precision_score(yt[mask], yp[mask], zero_division=0))
        )
    return float(max(precisions) - min(precisions)) if precisions else 0.0


def individual_fairness_wrapper(y_true, y_pred, sensitive_features) -> float:
    """Overall accuracy (0-1). NOT a difference and NOT absolute-valued —
    matches bias.py's inverted-threshold metric 5 (fail below 0.80)."""
    yt = _require_y_true(y_true, "individual_fairness_wrapper")
    frame = MetricFrame(
        metrics=accuracy_score,
        y_true=yt,
        y_pred=np.asarray(y_pred),
        sensitive_features=np.asarray(sensitive_features),
    )
    return float(frame.overall)


METRIC_WRAPPERS: dict[str, MetricFn] = {
    "demographic_parity_difference": demographic_parity_wrapper,
    "equalized_odds_difference": equalized_odds_wrapper,
    "equal_opportunity_difference": equal_opportunity_wrapper,
    "predictive_parity_difference": predictive_parity_wrapper,
    "individual_fairness_score": individual_fairness_wrapper,
}


# =========================================================================== #
# Multiple-comparisons correction (Phase 2 — gap 1.1)
# =========================================================================== #

CORRECTION_METHODS = ("bonferroni", "benjamini_hochberg")


@dataclass
class CorrectionResult:
    """Outcome of correcting a family of p-values for multiple testing.

    Attributes
    ----------
    method : str
        ``"bonferroni"`` or ``"benjamini_hochberg"``.
    original_alpha : float
        The uncorrected significance level (e.g. 0.05).
    corrected_alpha : float | list[float]
        **Bonferroni** — a single scalar, ``alpha / n``: the bar is lowered once,
        uniformly, and every p-value is compared against it.
        **Benjamini-Hochberg** — a list, one value per p-value in the *original*
        order: each comparison's critical value is ``(rank / n) * alpha`` where
        ``rank`` is that p-value's 1-indexed ascending rank. There is no single
        number because the BH threshold depends on rank.
    p_values : list[float]
        Echoed back, original order, for traceability.
    significant : list[bool]
        Per p-value, original order, after correction.
        **Bonferroni** — exactly ``p < corrected_alpha``.
        **Benjamini-Hochberg** — the *step-up* decision: find the largest rank
        ``k`` with ``p_(k) <= (k/n) * alpha``, then every hypothesis of rank
        ``<= k`` is significant. Consequence: ``significant[i]`` can be ``True``
        even when ``p_values[i] > corrected_alpha[i]`` — this is correct BH
        behaviour, not a bug, and is the most common way BH is implemented wrong.
    n_comparisons : int
        ``len(p_values)`` — the family size the correction was computed against.
    """

    method: str
    original_alpha: float
    corrected_alpha: float | list[float]
    p_values: list[float]
    significant: list[bool]
    n_comparisons: int


def _validate_p_values(p_values, alpha) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    arr = np.asarray(list(p_values), dtype=float)
    if arr.size == 0:
        raise ValueError("p_values must be a non-empty sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "p_values contains a non-finite value (nan/inf). A metric that could "
            "not produce a valid p-value is indeterminate and must be handled "
            "before correction, not passed through."
        )
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(
            f"every p-value must be in [0, 1]; got {arr.tolist()}"
        )
    return arr


def apply_multiple_comparisons_correction(
    p_values: list[float],
    method: str = "bonferroni",
    alpha: float = 0.05,
) -> CorrectionResult:
    """Correct a family of p-values so the *family-wise* (Bonferroni) or *false
    discovery* (Benjamini-Hochberg) error rate is controlled at ``alpha``.

    Both procedures are implemented directly (no library shortcut) so every step
    is auditable. BH is validated in the test suite against Benjamini & Hochberg
    (1995), *JRSS-B* 57(1):289-300, Table 1.
    """
    if method not in CORRECTION_METHODS:
        raise ValueError(
            f"method must be one of {CORRECTION_METHODS}, got {method!r}"
        )
    arr = _validate_p_values(p_values, alpha)
    n = arr.size
    p_list = arr.tolist()

    if method == "bonferroni":
        corrected_alpha: float = alpha / n
        significant = [p < corrected_alpha for p in p_list]
        return CorrectionResult(
            method=method,
            original_alpha=alpha,
            corrected_alpha=corrected_alpha,
            p_values=p_list,
            significant=significant,
            n_comparisons=n,
        )

    # ---- Benjamini-Hochberg -------------------------------------------- #
    order = np.argsort(arr, kind="stable")           # indices, ascending p
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)               # 1-indexed rank per original i
    per_rank_critical = [(int(ranks[i]) / n) * alpha for i in range(n)]

    p_sorted = arr[order]
    k = 0
    for i in range(n, 0, -1):                        # largest rank first
        if p_sorted[i - 1] <= (i / n) * alpha:
            k = i
            break
    significant = [int(ranks[i]) <= k for i in range(n)]

    return CorrectionResult(
        method=method,
        original_alpha=alpha,
        corrected_alpha=per_rank_critical,
        p_values=p_list,
        significant=significant,
        n_comparisons=n,
    )


# =========================================================================== #
# Reliability scoring (Phase 2 — gap 1.8)
# =========================================================================== #
#
# NOTE — this rule set was revised from the original BUILD_PLAN v2.0 draft
# during Component 2.3 implementation (2026-09-02) and BUILD_PLAN.md was updated
# to match. The draft folded "bootstrap SD > 0.05" into "insufficient_data" and
# used "n < 100 per group" plus an N-run reseed. Corrections:
#   * SD > 0.05 -> "unstable", not "insufficient_data". A large SD is measured
#     uncertainty on an estimate that WAS computed — categorically different from
#     data that cannot support any estimate.
#   * SD read from the single 1,000-iteration bootstrap distribution already
#     computed in Component 2.2 — an N-run reseed would multiply the ~4s/metric
#     cost by N for a largely redundant signal.
#   * min group size hard floor is 30, not 100. The 30-99 range that is still
#     too imprecise is caught by the CI-width and SD rules anyway.

RELIABILITY_TIERS = ("reliable", "unstable", "insufficient_data")

CI_WIDTH_UNSTABLE_THRESHOLD = 0.15
BOOTSTRAP_SD_UNSTABLE_THRESHOLD = 0.05
SINGLE_GROUP_SKIP_BLOCK_THRESHOLD = 0.05   # skipped fraction that blocks a verdict


@dataclass
class ReliabilityAssessment:
    """Whether a fairness result can support a compliance verdict.

    Attributes
    ----------
    tier : str
        Exactly one of ``RELIABILITY_TIERS``. Resolved by severity:
        ``insufficient_data`` outranks ``unstable`` outranks ``reliable``.
    reasons : list[str]
        Plain-language explanation of **every** rule that fired (not just the
        one that set the tier), so a compliance report shows the full picture.
        For a ``reliable`` result, a single positive confirmation line.
    blocks_verdict : bool
        ``True`` if and only if ``tier == "insufficient_data"``. When ``True``
        the compliance mapper must produce "indeterminate — insufficient sample"
        rather than pass or fail.
    """

    tier: str
    reasons: list[str]
    blocks_verdict: bool


def assess_reliability(
    bootstrap_result: BootstrapResult,
    sample_sizes: dict,
    min_group_size: int = 30,
) -> ReliabilityAssessment:
    """Assign a reliability tier to one metric's bootstrap result.

    Rules, all evaluated (reasons accumulate); tier is the most severe that
    fired:

    ==  ==================================================  ==================
    #   Condition                                           Tier
    ==  ==================================================  ==================
    0   ``n_valid_iterations == 0`` (every resample failed)  insufficient_data
    1   any ``sample_sizes[g] < min_group_size``             insufficient_data
    2   ``n_skipped_single_group / n_valid_iterations`` > 5%  insufficient_data
    3   ``ci_upper - ci_lower`` > 0.15                        unstable
    4   ``bootstrap SD`` > 0.05                               unstable
    5   none of the above                                    reliable
    ==  ==================================================  ==================
    """
    if not isinstance(sample_sizes, dict) or not sample_sizes:
        raise ValueError("sample_sizes must be a non-empty {group: count} dict")
    if min_group_size < 1:
        raise ValueError(f"min_group_size must be >= 1, got {min_group_size}")

    reasons: list[str] = []
    insufficient = False
    unstable = False

    n_valid = bootstrap_result.n_valid_iterations
    n_skipped_sg = bootstrap_result.n_skipped_single_group
    ci_width = bootstrap_result.ci_upper - bootstrap_result.ci_lower
    sd = bootstrap_result.bootstrap_distribution_summary.get("std")

    # Rule 0 — nothing was computed.
    if n_valid == 0:
        insufficient = True
        reasons.append(
            "All bootstrap resamples failed — no estimate could be computed, "
            "so no verdict is possible."
        )

    # Rule 1 — a group is below the hard floor.
    for group, size in sample_sizes.items():
        if size < min_group_size:
            insufficient = True
            reasons.append(
                f"Group '{group}' has only {int(size)} samples (minimum "
                f"{min_group_size} required for reliable estimation)."
            )

    # Rule 2 — resamples keep collapsing to one group.
    if n_valid > 0:
        skip_fraction = n_skipped_sg / n_valid
        if skip_fraction > SINGLE_GROUP_SKIP_BLOCK_THRESHOLD:
            insufficient = True
            reasons.append(
                f"{skip_fraction:.1%} of bootstrap resamples collapsed to a "
                f"single group — sample size or class balance is insufficient "
                f"for a reliable estimate."
            )

    # Rule 3 — the interval is too wide to act on.
    if np.isfinite(ci_width) and ci_width > CI_WIDTH_UNSTABLE_THRESHOLD:
        unstable = True
        reasons.append(
            f"95% confidence interval spans {ci_width:.3f} — too wide for a "
            f"precise verdict."
        )

    # Rule 4 — the estimate moves a lot across resamples.
    if sd is not None and np.isfinite(sd) and sd > BOOTSTRAP_SD_UNSTABLE_THRESHOLD:
        unstable = True
        reasons.append(
            f"Bootstrap distribution standard deviation ({sd:.3f}) indicates "
            f"high estimate variability."
        )

    if insufficient:
        tier = "insufficient_data"
    elif unstable:
        tier = "unstable"
    else:
        tier = "reliable"
        reasons = [
            f"All reliability checks passed: every group has at least "
            f"{min_group_size} samples, under 5% of bootstrap resamples failed, "
            f"the 95% CI width is within {CI_WIDTH_UNSTABLE_THRESHOLD}, and the "
            f"bootstrap SD is within {BOOTSTRAP_SD_UNSTABLE_THRESHOLD}."
        ]

    return ReliabilityAssessment(
        tier=tier,
        reasons=reasons,
        blocks_verdict=(tier == "insufficient_data"),
    )
