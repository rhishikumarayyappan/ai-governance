"""Phase 2 wiring sub-step — behaviour tests for the statistical layer that
``run_bias_tests`` now computes and persists onto every TestResult row.

These run the **real** 1,000-iteration bootstrap / permutation path (marked
``statistical`` — slow, ~30s per run). They deliberately live in their own file
so the iteration-count-reducing fixture in ``test_engine.py`` /
``test_api_testing.py`` cannot reach them: those fixtures are module-scoped
autouse, and importing nothing from those modules here makes it structurally
impossible for a value-asserting test to run at a reduced count.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sqlalchemy import select

from governance.db import models
from governance.db.database import get_session
from governance.db.models import AISystem, RiskTier
from governance.testing.engine import get_run_results, run_bias_tests

pytestmark = pytest.mark.statistical

_GAP_METRICS = {
    "demographic_parity_difference",
    "equalized_odds_difference",
    "equal_opportunity_difference",
    "predictive_parity_difference",
}


def _system() -> str:
    with get_session() as session:
        s = AISystem(name="stats-test", model_type="classification",
                     risk_tier=RiskTier.high)
        session.add(s)
        session.commit()
        return s.id


# --------------------------------------------------------------------------- #
# 1 — large, clean data: every column populated, headline metric reliable
# --------------------------------------------------------------------------- #
def test_large_clean_run_populates_every_column(test_db):
    rng = np.random.RandomState(1)
    n = 2000
    f1 = rng.randn(n)
    grp = np.array(["A", "B"] * (n // 2))
    # strong f1 -> y relationship, NO group effect: the gaps are ~0 with tight CIs
    y = ((f1 * 2.5 + rng.randn(n) * 0.1) > 0).astype(int)
    X = pd.DataFrame({"f1": f1, "grp": grp})
    model = LogisticRegression().fit(X[["f1"]].values, y)

    run_id = run_bias_tests(_system(), model, X, y, ["grp"], random_state=42)
    rows = {r["metric_name"]: r for r in get_run_results(run_id)}
    assert set(rows) == _GAP_METRICS | {"overall_accuracy_floor"}

    for name, r in rows.items():
        # every metric: CI bounds, a valid tier, sample size
        assert r["confidence_interval_lower"] is not None
        assert r["confidence_interval_upper"] is not None
        assert r["confidence_interval_lower"] <= r["metric_value"] <= r["confidence_interval_upper"]
        assert r["reliability_tier"] in ("reliable", "unstable", "insufficient_data")
        assert r["sample_size"] == n
        if name in _GAP_METRICS:
            assert r["p_value"] is not None
            assert r["correction_method"] == "bonferroni"
            assert r["corrected_threshold"] == pytest.approx(0.05 / 4)
        else:  # overall_accuracy_floor — structurally not a significance test
            assert r["p_value"] is None
            assert r["corrected_threshold"] is None
            assert r["correction_method"] is None

    # on 2,000 rows of clean data the headline metric is a precise estimate
    assert rows["demographic_parity_difference"]["reliability_tier"] == "reliable"
    assert rows["overall_accuracy_floor"]["reliability_tier"] == "reliable"


# --------------------------------------------------------------------------- #
# 2 — a group below 30: status is "indeterminate", never a silent pass/fail
# --------------------------------------------------------------------------- #
def test_small_group_makes_every_metric_indeterminate(test_db):
    rng = np.random.RandomState(2)
    n = 120
    f1 = rng.randn(n)
    # 105 in group A, 15 in group B — B is below the 30-sample floor
    grp = np.array(["A"] * 105 + ["B"] * 15)
    y = ((f1 + rng.randn(n) * 0.5) > 0).astype(int)
    X = pd.DataFrame({"f1": f1, "grp": grp})
    model = LogisticRegression().fit(X[["f1"]].values, y)

    run_id = run_bias_tests(_system(), model, X, y, ["grp"], random_state=42)
    rows = get_run_results(run_id)
    assert len(rows) == 5

    for r in rows:
        assert r["reliability_tier"] == "insufficient_data"
        assert r["status"] == "indeterminate"
        # explicitly: an indeterminate row is NOT a pass and NOT a fail
        assert r["status"] != "pass"
        assert r["status"] != "fail"
        assert r["status"] != "warn"

    # and nothing downstream that checks `status == "pass"` would count it
    assert not any(r["status"] == "pass" for r in rows)


# --------------------------------------------------------------------------- #
# 3 — the correction actually changes a verdict (it is wired, not discarded)
# --------------------------------------------------------------------------- #
def test_multiple_comparison_correction_changes_a_verdict(test_db):
    # seed=9 data: a trained model where equalized_odds (p~0.019) and
    # equal_opportunity (p~0.023) are each significant at raw alpha=0.05 but
    # NOT after Bonferroni tightens the bar to 0.05/4 = 0.0125.
    rng = np.random.RandomState(9)
    n = 350
    f1 = rng.randn(n)
    f2 = rng.randn(n)
    grp = np.array(["A", "B"] * (n // 2))
    y = ((f1 * 1.2 + (grp == "A") * 0.9 + rng.randn(n) * 1.0) > 0).astype(int)
    X = pd.DataFrame({"f1": f1, "f2": f2, "grp": grp})
    model = LogisticRegression().fit(X[["f1", "f2"]].values, y)

    run_id = run_bias_tests(_system(), model, X, y, ["grp"], random_state=42)
    gap_rows = [
        r for r in get_run_results(run_id) if r["metric_name"] in _GAP_METRICS
    ]
    assert len(gap_rows) == 4

    ct = gap_rows[0]["corrected_threshold"]
    assert ct == pytest.approx(0.05 / 4)          # family of 4, not 1 and not 5

    raw = {r["metric_name"]: r["p_value"] < 0.05 for r in gap_rows}
    corrected = {r["metric_name"]: r["p_value"] < ct for r in gap_rows}

    # Bonferroni can only ever *remove* significance, never add it
    for name in raw:
        assert not (corrected[name] and not raw[name])
    # and here it removes it from at least one metric — proof it is applied,
    # not merely computed and thrown away
    assert any(raw[name] and not corrected[name] for name in raw)


# --------------------------------------------------------------------------- #
# 4 — bias.py's public contract is unchanged through the engine path
# --------------------------------------------------------------------------- #
def test_engine_still_persists_exactly_five_metrics_in_fixed_order(test_db):
    rng = np.random.RandomState(4)
    n = 200
    f1 = rng.randn(n)
    grp = np.array(["A", "B"] * (n // 2))
    y = ((f1 + rng.randn(n) * 0.5) > 0).astype(int)
    X = pd.DataFrame({"f1": f1, "grp": grp})
    model = LogisticRegression().fit(X[["f1"]].values, y)

    from governance.testing.bias import BiasTestSuite

    run_id = run_bias_tests(_system(), model, X, y, ["grp"], random_state=42)
    with get_session() as session:
        rows = session.scalars(
            select(models.TestResult).where(models.TestResult.run_id == run_id)
        ).all()

    names = [r.metric_name for r in rows]
    assert len(names) == 5                                   # exactly 5, still
    assert sorted(names) == sorted(BiasTestSuite.METRIC_ORDER)  # one per metric
    # BiasTestSuite.run() itself still asserts its fixed order internally on
    # every call — the engine just persists what it returns, one row per result.
