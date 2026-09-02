"""Statistical rigour layer — Component 2.1: significance testing.

Answers one question for a single protected attribute: *could this
demographic-parity gap in the model's predictions plausibly be random noise?*

This module is a **sibling** of ``bias.py`` — it is called alongside
``BiasTestSuite``, never wraps it, and never imports from it (and ``bias.py``
never imports from here). The fairlearn metric used for the permutation test is
imported directly from fairlearn so the permuted statistic matches exactly what
``BiasTestSuite`` reports.

Scope of THIS component (Phase 2, Component 2.1 — significance only):
  * four tests — chi-squared, Fisher's exact, two-proportion z-test, permutation
  * an ``auto`` selector that picks the right one by group count and cell size
  * an always-on permutation cross-check attached to every result

Explicitly NOT in scope here (later pieces of Phase 2 / Phase 4):
  * bootstrap confidence intervals, Bonferroni / Benjamini-Hochberg correction,
    reliability scoring  — later in Component 2.1
  * continuous or multi-class model outputs — Phase 4 (``regression.py`` etc.)
  * general R×C chi-squared — would be its own component with its own tests

See docs/BUILD_PLAN.md → "PHASE 2 → Component 2.1 → Significance Testing" and
docs/GAP_CHECKLIST.md → Category 1, gap 1.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from scipy.stats import chi2_contingency, fisher_exact, norm

__all__ = ["SignificanceResult", "significance_test", "VALID_METHODS"]

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
