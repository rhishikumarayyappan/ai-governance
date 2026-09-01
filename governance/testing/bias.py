"""Bias & fairness test suite — Component 2 of the Phase 1 testing engine.

`BiasTestSuite.run()` always returns exactly 5 `BiasTestResult` objects, in a
fixed order. Every downstream component (the engine orchestrator, the compliance
mapper, the PDF report) depends on that shape and that order — do not change
either without updating them.

See docs/BUILD_PLAN.md → "Component 2 — BiasTestSuite".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import accuracy_score, precision_score

__all__ = ["BiasTestResult", "BiasTestSuite"]

_VALID_STATUSES = ("pass", "warn", "fail")


@dataclass
class BiasTestResult:
    """The single output unit of the whole testing engine — one per metric.

    `value` is always stored rounded to 4 decimal places (never a raw float).
    `status` is always exactly one of "pass", "warn", "fail".
    """

    metric_name: str
    value: float
    threshold: float
    status: str
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Never store raw floating point — always 4 dp.
        self.value = round(float(self.value), 4)
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"BiasTestResult.status must be one of {_VALID_STATUSES}, "
                f"got {self.status!r}"
            )

    @property
    def passed(self) -> bool:
        """True only when status is 'pass'. 'warn' and 'fail' both return False."""
        return self.status == "pass"


class BiasTestSuite:
    """Runs the 5 fairness metrics against a set of predictions."""

    # Class attribute (not set in __init__) so it can be inspected without
    # instantiating the class.
    DEFAULT_THRESHOLDS = {
        "demographic_parity_difference": 0.10,
        "equalized_odds_difference": 0.10,
        "equal_opportunity_difference": 0.10,
        "predictive_parity_difference": 0.10,
        "individual_fairness_score": 0.80,
    }

    # The fixed order run() returns results in. Never reorder — other components
    # index into this list positionally.
    METRIC_ORDER = (
        "demographic_parity_difference",
        "equalized_odds_difference",
        "equal_opportunity_difference",
        "predictive_parity_difference",
        "individual_fairness_score",
    )

    def __init__(self, thresholds: dict | None = None) -> None:
        merged = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            merged.update(thresholds)  # custom values override, defaults fill gaps
        self.thresholds = merged

    # ------------------------------------------------------------------ #
    # Threshold logic — centralised here, never duplicated in a metric.
    # ------------------------------------------------------------------ #
    def _get_status(self, metric_name: str, value: float) -> str:
        threshold = self.thresholds[metric_name]

        if metric_name == "individual_fairness_score":
            # INVERTED THRESHOLD — fail below, pass above. No warn band.
            return "pass" if value >= threshold else "fail"

        # Metrics 1-4: it is a gap, lower is better.
        if value <= threshold * 0.7:
            return "pass"
        if value <= threshold:
            return "warn"
        return "fail"

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        y_true,
        y_pred,
        sensitive_features,
        attribute_name: str,
    ) -> list[BiasTestResult]:
        """Return exactly 5 BiasTestResult objects in METRIC_ORDER."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if not isinstance(sensitive_features, pd.Series):
            sensitive_features = pd.Series(sensitive_features)
        groups = sensitive_features.to_numpy()

        results = [
            self._demographic_parity_difference(y_true, y_pred, sensitive_features, groups),
            self._equalized_odds_difference(y_true, y_pred, sensitive_features, attribute_name),
            self._equal_opportunity_difference(y_true, y_pred, sensitive_features, attribute_name),
            self._predictive_parity_difference(y_true, y_pred, groups, attribute_name),
            self._individual_fairness_score(y_true, y_pred, sensitive_features, attribute_name),
        ]

        # Guard the contract: exactly 5, fixed order.
        assert [r.metric_name for r in results] == list(self.METRIC_ORDER), (
            "BiasTestSuite.run() produced results out of the fixed order"
        )
        return results

    # ------------------------------------------------------------------ #
    # Metric 1 — Demographic Parity Difference (fairlearn)
    # ------------------------------------------------------------------ #
    def _demographic_parity_difference(self, y_true, y_pred, sensitive_features, groups):
        name = "demographic_parity_difference"
        raw = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        value = round(abs(float(raw)), 4)  # magnitude only, direction doesn't matter

        # detail MUST carry the per-group positive prediction rates — the
        # compliance report needs "men 0.71, women 0.52", not just the gap.
        per_group_rates = {
            str(g): round(float(np.mean(y_pred[groups == g])), 4)
            for g in pd.unique(pd.Series(groups))
        }

        return BiasTestResult(
            metric_name=name,
            value=value,
            threshold=self.thresholds[name],
            status=self._get_status(name, value),
            detail=per_group_rates,
        )

    # ------------------------------------------------------------------ #
    # Metric 2 — Equalized Odds Difference (fairlearn)
    # ------------------------------------------------------------------ #
    def _equalized_odds_difference(self, y_true, y_pred, sensitive_features, attribute_name):
        name = "equalized_odds_difference"
        raw = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        value = round(abs(float(raw)), 4)
        return BiasTestResult(
            metric_name=name,
            value=value,
            threshold=self.thresholds[name],
            status=self._get_status(name, value),
            detail={"attribute_name": attribute_name},
        )

    # ------------------------------------------------------------------ #
    # Metric 3 — Equal Opportunity Difference (fairlearn)
    # ------------------------------------------------------------------ #
    def _equal_opportunity_difference(self, y_true, y_pred, sensitive_features, attribute_name):
        name = "equal_opportunity_difference"
        raw = equal_opportunity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        value = round(abs(float(raw)), 4)
        return BiasTestResult(
            metric_name=name,
            value=value,
            threshold=self.thresholds[name],
            status=self._get_status(name, value),
            detail={"attribute_name": attribute_name},
        )

    # ------------------------------------------------------------------ #
    # Metric 4 — Predictive Parity Difference (manual, sklearn)
    # No fairlearn function exists — gap in precision between groups.
    # ------------------------------------------------------------------ #
    def _predictive_parity_difference(self, y_true, y_pred, groups, attribute_name):
        name = "predictive_parity_difference"
        precisions = []
        for g in pd.unique(pd.Series(groups)):
            mask = groups == g
            precisions.append(
                float(precision_score(y_true[mask], y_pred[mask], zero_division=0))
            )
        value = round(max(precisions) - min(precisions), 4) if precisions else 0.0
        return BiasTestResult(
            metric_name=name,
            value=value,
            threshold=self.thresholds[name],
            status=self._get_status(name, value),
            detail={"attribute_name": attribute_name},
        )

    # ------------------------------------------------------------------ #
    # Metric 5 — Individual Fairness Score (fairlearn MetricFrame)
    # INVERTED THRESHOLD — fail below, pass above.
    # ------------------------------------------------------------------ #
    def _individual_fairness_score(self, y_true, y_pred, sensitive_features, attribute_name):
        name = "individual_fairness_score"
        frame = MetricFrame(
            metrics=accuracy_score,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )
        # Individual fairness proxy = overall consistency (accuracy) of the model.
        value = round(float(frame.overall), 4)
        # INVERTED THRESHOLD — fail below, pass above
        status = self._get_status(name, value)
        return BiasTestResult(
            metric_name=name,
            value=value,
            threshold=self.thresholds[name],
            status=status,
            detail={
                "attribute_name": attribute_name,
                "by_group": {
                    str(k): round(float(v), 4) for k, v in frame.by_group.items()
                },
            },
        )
