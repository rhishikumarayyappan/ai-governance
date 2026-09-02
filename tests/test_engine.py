"""Phase 1, Week 2, Component 3 — tests for the engine orchestrator.

Four tests, as specified in the session brief. Every test uses the function-
scoped `test_db` fixture, so each starts with a completely clean database and
nothing leaks between them.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sqlalchemy import func, select

from governance.db import models  # noqa: imported as module so pytest doesn't
from governance.db.database import get_session  # try to collect TestRun/TestResult
from governance.db.models import AISystem, RiskTier
from governance.testing import engine as _engine
from governance.testing.engine import get_run_results, run_bias_tests


@pytest.fixture(autouse=True)
def _fast_statistics(monkeypatch):
    """Drop the engine's bootstrap / permutation iteration counts from 1,000 to
    100 for the duration of each test in THIS module.

    Exists purely for suite speed on structural / plumbing tests — the ones that
    verify "5 results, valid status, all columns populated", not a specific
    statistical value. **Never rely on this in a test that asserts a particular
    CI bound, p-value, or corrected threshold** — those live in
    ``tests/test_engine_statistics.py``, which does not import this fixture and
    runs the real 1,000-iteration path. The production ``run_bias_tests`` code
    path is completely unaffected; only these two module-level constants are
    patched, and only inside this test module.

    100 is verified to give the *same* ``reliability_tier`` as 1,000 for the
    fixtures here (all "unstable" — 140 rows of noise genuinely can't support a
    tight estimate), and single-group resample skips stay at 0 with a 70/70
    split, so the 5% skip threshold is never in play.
    """
    monkeypatch.setattr(_engine, "_BOOTSTRAP_ITERATIONS", 100)
    monkeypatch.setattr(_engine, "_PERMUTATION_ITERATIONS", 100)


def _make_system(name: str = "test-system") -> str:
    with get_session() as session:
        system = AISystem(
            name=name,
            model_type="classification",
            risk_tier=RiskTier.high,
            sector="finance",
            owner="test@example.com",
        )
        session.add(system)
        session.commit()
        return system.id


@pytest.fixture
def toy_data():
    """140 rows (70 per group), 3 numeric features, 1 protected attribute.

    Sized so both groups clear the 30-sample reliability floor — a 20-row run
    genuinely cannot support a verdict, so since Phase 2 the engine would flag
    every metric "indeterminate" on the old fixture. This is a realistic-enough
    toy, not a workaround.
    """
    rng = np.random.RandomState(0)
    n = 140
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "f3": rng.randn(n),
            "grp": np.array(["A", "B"] * (n // 2)),
        }
    )
    y = rng.randint(0, 2, size=n)
    model = LogisticRegression().fit(X[["f1", "f2", "f3"]].values, y)
    return model, X, y


def test_happy_path_end_to_end(test_db, toy_data):
    model, X, y = toy_data
    system_id = _make_system()

    run_id = run_bias_tests(
        system_id=system_id,
        model_source=model,
        X_test=X,
        y_test=y,
        protected_attributes=["grp"],
    )

    assert isinstance(run_id, str)

    with get_session() as session:
        run = session.get(models.TestRun, run_id)
        assert run is not None
        assert run.status == "complete"
        assert run.completed_at is not None
        results = session.scalars(
            select(models.TestResult).where(models.TestResult.run_id == run_id)
        ).all()

    assert len(results) == 5
    assert all(r.status in ("pass", "warn", "fail") for r in results)


def test_invalid_system_id_raises_and_writes_nothing(test_db):
    with pytest.raises(ValueError):
        run_bias_tests(
            system_id="does-not-exist",
            model_source=LogisticRegression(),
            X_test=pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "grp": ["A", "B", "A", "B"]}),
            y_test=np.array([0, 1, 1, 0]),
            protected_attributes=["grp"],
        )

    with get_session() as session:
        assert session.scalar(select(func.count()).select_from(models.TestRun)) == 0
        assert session.scalar(select(func.count()).select_from(models.TestResult)) == 0


def test_missing_protected_column_is_skipped_not_fatal(test_db, toy_data):
    model, X, y = toy_data
    system_id = _make_system()

    run_id = run_bias_tests(
        system_id=system_id,
        model_source=model,
        X_test=X,
        y_test=y,
        protected_attributes=["nonexistent_column"],
    )

    assert isinstance(run_id, str)
    with get_session() as session:
        run = session.get(models.TestRun, run_id)
        assert run.status == "complete"  # missing column skipped, not fatal
        count = session.scalar(
            select(func.count())
            .select_from(models.TestResult)
            .where(models.TestResult.run_id == run_id)
        )
        assert count == 0  # nothing to save — the only attribute was skipped


def test_get_run_results_returns_five_well_formed_dicts(test_db, toy_data):
    model, X, y = toy_data
    system_id = _make_system()

    run_id = run_bias_tests(
        system_id=system_id,
        model_source=model,
        X_test=X,
        y_test=y,
        protected_attributes=["grp"],
    )

    results = get_run_results(run_id)

    assert isinstance(results, list)
    assert len(results) == 5
    for r in results:
        assert {"metric_name", "metric_value", "threshold", "status", "detail"} <= r.keys()
        assert isinstance(r["metric_value"], float)
