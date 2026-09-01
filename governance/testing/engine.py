"""Engine orchestrator — Component 3 of the Phase 1 testing engine.

The single entry point that runs a full bias test for one AI system and writes
the results to SQLite. This is the ONLY module allowed to create TestRun and
TestResult rows — nothing else writes those tables directly.

See docs/BUILD_PLAN.md -> "Component 3 — Engine Orchestrator".
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from governance.db.database import get_session
from governance.db.models import AISystem, TestResult, TestRun
from governance.testing.adapters import load_adapter
from governance.testing.bias import BiasTestSuite

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = ("complete", "failed")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mark_run_failed(run_id: str, message: str) -> None:
    """Force a TestRun into 'failed' state.

    Uses its own fresh session so it succeeds even if the caller's session is in
    a broken transaction. Idempotent: a run already in a terminal state is left
    untouched, so the first (most specific) error message wins.
    """
    try:
        with get_session() as session:
            run = session.get(TestRun, run_id)
            if run is None or run.status in _TERMINAL_STATUSES:
                return
            config = dict(run.config or {})
            config["error"] = message
            run.config = config  # reassign so SQLAlchemy sees the JSON change
            run.status = "failed"
            run.completed_at = _now()
            session.commit()
    except Exception:  # noqa: BLE001 - safety net; must never mask the real error
        logger.exception("Could not mark TestRun %s as failed", run_id)


def run_bias_tests(
    system_id: str,
    model_source: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    protected_attributes: list[str],
    config: dict | None = None,
) -> str:
    """Run the full bias test suite for one AI system and persist everything.

    The ONLY function permitted to create TestRun and TestResult rows. Opens a
    TestRun immediately (status "running") so the run is trackable from the first
    moment, then loads the model, runs BiasTestSuite once per protected
    attribute, saves every metric result, and closes the TestRun as "complete".
    On any failure the TestRun is set to "failed" (error recorded in
    config["error"], completed_at set) before the exception is re-raised — a run
    is never left permanently in "running".

    Parameters
    ----------
    system_id : str
        UUID of an existing AISystem row. Verified against SQLite before any
        TestRun is created; raises ValueError (naming the id) if absent, and no
        rows are written in that case.
    model_source : Any
        Anything load_adapter() accepts — a fitted estimator, a .pkl path, or a
        prediction-endpoint URL.
    X_test : pd.DataFrame
        Feature frame. Must contain every column in protected_attributes. All
        protected columns are dropped before predict() so the model never sees
        them as features during the fairness measurement.
    y_test : np.ndarray
        1D ground-truth labels, same length as X_test.
    protected_attributes : list[str]
        Column names in X_test to test as sensitive features. Each gets its own
        full 5-metric run. A name not present in X_test is logged as a warning
        and skipped — it does not fail the run.
    config : dict | None
        Optional dict recorded verbatim on the TestRun for audit. Defaults to {}.

    Returns
    -------
    str
        The run_id (UUID) of the TestRun.

    Raises
    ------
    ValueError
        If system_id matches no AISystem (raised before any DB write).
    Exception
        Any model-load or test-execution error, re-raised after the TestRun is
        marked "failed".
    """
    # ---- Stage A: validate system_id, then open the TestRun ----------------
    with get_session() as session:
        system = session.get(AISystem, system_id)
        if system is None:
            raise ValueError(
                f"Cannot run tests: no AISystem exists with id '{system_id}'"
            )

        run_id = str(uuid.uuid4())
        run = TestRun(
            id=run_id,
            system_id=system_id,
            status="running",
            config=dict(config or {}),
            started_at=_now(),
            completed_at=None,
        )
        session.add(run)
        session.commit()  # run is visible in the DB before any testing begins

    # ---- Stages B-D, wrapped in the top-level safety net (Stage E) ---------
    try:
        # Stage B: load the model
        try:
            adapter = load_adapter(model_source)
        except Exception as exc:
            _mark_run_failed(run_id, f"model load failed: {exc}")
            raise

        # Stage C: run the suite once per protected attribute
        for attribute_name in protected_attributes:
            if attribute_name not in X_test.columns:
                logger.warning(
                    f"Column {attribute_name} not found in X_test — skipping"
                )
                continue

            # Drop ALL protected columns so the model never sees them as
            # features during the fairness measurement.
            features = X_test.drop(columns=protected_attributes, errors="ignore")
            y_pred = adapter.predict(features.values)
            sensitive = X_test[attribute_name]

            results = BiasTestSuite().run(y_test, y_pred, sensitive, attribute_name)

            with get_session() as session:
                session.add_all(
                    [
                        TestResult(
                            id=str(uuid.uuid4()),
                            run_id=run_id,
                            module="bias",
                            metric_name=r.metric_name,
                            metric_value=r.value,
                            threshold=r.threshold,
                            status=r.status,
                            detail=r.detail,
                        )
                        for r in results
                    ]
                )
                session.commit()

        # Stage D: close the run
        with get_session() as session:
            run = session.get(TestRun, run_id)
            run.status = "complete"
            run.completed_at = _now()
            session.commit()

    except Exception as exc:
        # Stage E: safety net — a run must never stay "running".
        _mark_run_failed(run_id, str(exc))
        raise

    return run_id


def get_run_results(run_id: str) -> list[dict]:
    """Return all TestResult rows for run_id as plain dicts.

    One dict per result with keys: id, run_id, module, metric_name,
    metric_value, threshold, status, detail. Returns [] for an unknown run_id —
    never raises (the API layer decides whether that is a 404).
    """
    with get_session() as session:
        rows = session.scalars(
            select(TestResult).where(TestResult.run_id == run_id)
        ).all()
        return [
            {
                "id": row.id,
                "run_id": row.run_id,
                "module": row.module,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "threshold": row.threshold,
                "status": row.status,
                "detail": row.detail,
            }
            for row in rows
        ]
