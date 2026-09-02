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
from governance.testing.statistics import (
    METRIC_WRAPPERS,
    apply_multiple_comparisons_correction,
    assess_reliability,
    bootstrap_confidence_interval,
    permutation_p_value,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = ("complete", "failed")

# --- Phase 2 statistical layer configuration --------------------------------- #
# The 4 group-comparison fairness metrics. These get a permutation p-value and
# multiple-comparison correction as ONE family, per protected attribute.
# overall_accuracy_floor is deliberately excluded — shuffling group labels
# cannot change overall accuracy, so a permutation p-value for it is
# structurally meaningless (there is no group-comparison hypothesis to test).
_FAIRNESS_GAP_METRICS = (
    "demographic_parity_difference",
    "equalized_odds_difference",
    "equal_opportunity_difference",
    "predictive_parity_difference",
)
_INDETERMINATE = "indeterminate"
_BOOTSTRAP_ITERATIONS = 1_000
_PERMUTATION_ITERATIONS = 1_000
_CORRECTION_METHOD = "bonferroni"      # conservative, controls family-wise error
_RELIABILITY_MIN_GROUP = 30
_DEFAULT_RANDOM_STATE = 42             # pinned so a re-run reproduces the numbers


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


def _compute_attribute_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    bias_results: list,
    *,
    random_state: int,
) -> dict[str, dict]:
    """Phase 2 statistical enrichment for ONE protected attribute.

    For every one of the 5 metrics: a bootstrap 95% CI and a reliability tier.
    For the 4 group-comparison fairness metrics only: a permutation p-value and
    a multiple-comparison-corrected significance threshold, corrected across
    those 4 together (one hypothesis family per attribute — never mixed with a
    different attribute's metrics).

    Returns ``{metric_name: {...}}`` with keys matching the new TestResult
    columns plus ``status`` (which is ``"indeterminate"`` when reliability
    scoring says the data cannot support a verdict, otherwise the raw
    pass/warn/fail from BiasTestSuite).
    """
    sf = pd.Series(sensitive).reset_index(drop=True)
    sample_sizes = {str(k): int(v) for k, v in sf.value_counts().items()}
    n_total = int(len(sf))

    out: dict[str, dict] = {}
    gap_p_values: dict[str, float] = {}

    for r in bias_results:
        name = r.metric_name
        wrapper = METRIC_WRAPPERS[name]

        boot = bootstrap_confidence_interval(
            wrapper, y_true, y_pred, sf,
            n_iterations=_BOOTSTRAP_ITERATIONS, random_state=random_state,
        )
        reliability = assess_reliability(
            boot, sample_sizes, min_group_size=_RELIABILITY_MIN_GROUP
        )
        out[name] = {
            "confidence_interval_lower": boot.ci_lower,
            "confidence_interval_upper": boot.ci_upper,
            "reliability_tier": reliability.tier,
            "sample_size": n_total,
            # populated below for the 4 fairness-gap metrics; left NULL for
            # overall_accuracy_floor — not a group-comparison metric, so a
            # permutation p-value is structurally meaningless (shuffling group
            # labels cannot change overall accuracy).
            "p_value": None,
            "corrected_threshold": None,
            "correction_method": None,
            "_raw_status": r.status,
            "_blocks_verdict": reliability.blocks_verdict,
        }

        if name in _FAIRNESS_GAP_METRICS:
            _observed, p = permutation_p_value(
                wrapper, y_true, y_pred, sf,
                n_permutations=_PERMUTATION_ITERATIONS, random_state=random_state,
            )
            gap_p_values[name] = p

    # ---- correct the fairness-gap p-values as one family ------------------ #
    if gap_p_values:
        ordered = [m for m in _FAIRNESS_GAP_METRICS if m in gap_p_values]
        correction = apply_multiple_comparisons_correction(
            [gap_p_values[m] for m in ordered], method=_CORRECTION_METHOD
        )
        ca = correction.corrected_alpha  # Bonferroni -> scalar; BH -> per-rank list
        for i, m in enumerate(ordered):
            out[m]["p_value"] = gap_p_values[m]
            out[m]["correction_method"] = correction.method
            out[m]["corrected_threshold"] = (
                float(ca) if isinstance(ca, (int, float)) else float(ca[i])
            )

    # ---- resolve the persisted status ---------------------------------- #
    for d in out.values():
        blocks = d.pop("_blocks_verdict")
        raw = d.pop("_raw_status")
        d["status"] = _INDETERMINATE if blocks else raw

    return out


def run_bias_tests(
    system_id: str,
    model_source: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    protected_attributes: list[str],
    config: dict | None = None,
    *,
    random_state: int = _DEFAULT_RANDOM_STATE,
) -> str:
    """Run the full bias test suite for one AI system and persist everything.

    The ONLY function permitted to create TestRun and TestResult rows. Opens a
    TestRun immediately (status "running") so the run is trackable from the first
    moment, then loads the model, runs BiasTestSuite once per protected
    attribute, computes the Phase 2 statistical layer for that attribute
    (bootstrap 95% CI + reliability tier for all 5 metrics; permutation p-value
    + multiple-comparison correction for the 4 fairness-gap metrics as one
    family), saves every metric result with those columns populated, and closes
    the TestRun as "complete". On any failure the TestRun is set to "failed"
    (error recorded in config["error"], completed_at set) before the exception
    is re-raised — a run is never left permanently in "running".

    When reliability scoring returns tier="insufficient_data" for a metric, that
    row's persisted ``status`` is ``"indeterminate"`` — the data cannot support
    a pass/warn/fail verdict. Downstream code must check ``status == "pass"``
    explicitly, never ``status != "fail"``.

    The statistical layer is on for every run. It adds real runtime — roughly
    9 x 1,000-iteration resampling loops per protected attribute, seconds on
    small data and minutes on large. ``random_state`` is pinned by default so a
    re-run of the same test reproduces the same CIs and p-values.

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

            # Phase 2: bootstrap CI + reliability for all 5, permutation p-value
            # + multiple-comparison correction for the 4 fairness-gap metrics.
            stats = _compute_attribute_statistics(
                y_test, y_pred, sensitive, results, random_state=random_state
            )

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
                            status=stats[r.metric_name]["status"],
                            detail=r.detail,
                            confidence_interval_lower=stats[r.metric_name][
                                "confidence_interval_lower"
                            ],
                            confidence_interval_upper=stats[r.metric_name][
                                "confidence_interval_upper"
                            ],
                            p_value=stats[r.metric_name]["p_value"],
                            corrected_threshold=stats[r.metric_name][
                                "corrected_threshold"
                            ],
                            correction_method=stats[r.metric_name][
                                "correction_method"
                            ],
                            reliability_tier=stats[r.metric_name][
                                "reliability_tier"
                            ],
                            sample_size=stats[r.metric_name]["sample_size"],
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

    One dict per result. Keys: id, run_id, module, metric_name, metric_value,
    threshold, status, detail, plus the Phase 2 statistical columns —
    confidence_interval_lower, confidence_interval_upper, p_value,
    corrected_threshold, correction_method, reliability_tier, sample_size (the
    last three of those are NULL for overall_accuracy_floor). Returns [] for an
    unknown run_id — never raises (the API layer decides whether that is a 404).
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
                "confidence_interval_lower": row.confidence_interval_lower,
                "confidence_interval_upper": row.confidence_interval_upper,
                "p_value": row.p_value,
                "corrected_threshold": row.corrected_threshold,
                "correction_method": row.correction_method,
                "reliability_tier": row.reliability_tier,
                "sample_size": row.sample_size,
            }
            for row in rows
        ]
