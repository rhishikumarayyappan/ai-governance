"""Phase 1 Week 3 — validation of the bias engine against published benchmarks.

Runs the EXISTING BiasTestSuite (governance/testing/bias.py) against three
public datasets and checks the headline metric falls inside the range the
literature reports. No governance/ code is modified in Week 3 — this script
only exercises what is already built.

Datasets
--------
1. UCI Adult Income      — sex bias, demographic_parity_difference ∈ [0.15, 0.25]
2. ProPublica COMPAS     — race bias, equalized_odds_difference   ∈ [0.15, 0.25]
3. UCI German Credit     — age bias, demographic_parity_difference ∈ [0.05, 0.20]

Run:  poetry run python notebooks/validate_phase1.py
"""

from __future__ import annotations

import io
import json
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from governance.testing.bias import BiasTestSuite

RANDOM_STATE = 42
COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)


def _fit_predict(X_train, y_train, X_test, numeric, categorical, model):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)


def _summarise(results, headline_metric):
    rows = []
    headline = None
    for r in results:
        rows.append(
            f"    {r.metric_name:<32} {r.value:>8.4f}   ({r.status})"
        )
        if r.metric_name == headline_metric:
            headline = r.value
    return headline, "\n".join(rows)


# --------------------------------------------------------------------------- #
# 1. UCI Adult Income
# --------------------------------------------------------------------------- #
def validate_adult():
    from fairlearn.datasets import fetch_adult

    data = fetch_adult(as_frame=True)
    df = data.frame.dropna()
    y = (df["class"].astype(str).str.contains(">50K")).astype(int).to_numpy()

    sensitive = df["sex"].astype(str)
    numeric = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    categorical = [
        "workclass", "marital-status", "occupation", "relationship", "race",
        "native-country",
    ]
    X = df[numeric + categorical]

    X_tr, X_te, y_tr, y_te, _, s_te = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    y_pred = _fit_predict(
        X_tr, y_tr, X_te, numeric, categorical,
        LogisticRegression(max_iter=2000),
    )
    results = BiasTestSuite().run(y_te, y_pred, s_te.reset_index(drop=True), "sex")
    return _summarise(results, "demographic_parity_difference")


# --------------------------------------------------------------------------- #
# 2. ProPublica COMPAS
# --------------------------------------------------------------------------- #
def validate_compas():
    raw = urllib.request.urlopen(COMPAS_URL, timeout=60).read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))

    # ProPublica's own filtering (Angwin et al. 2016, "How We Analyzed…").
    df = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ]
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()

    # Ground truth: did the person actually reoffend within two years.
    y = df["two_year_recid"].astype(int).to_numpy()
    sensitive = df["race"].astype(str)

    numeric = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    categorical = ["c_charge_degree", "sex"]
    X = df[numeric + categorical]

    X_tr, X_te, y_tr, y_te, _, s_te = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    y_pred = _fit_predict(
        X_tr, y_tr, X_te, numeric, categorical,
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    )
    results = BiasTestSuite().run(y_te, y_pred, s_te.reset_index(drop=True), "race")
    return _summarise(results, "equalized_odds_difference")


# --------------------------------------------------------------------------- #
# 3. UCI German Credit (via OpenML)
# --------------------------------------------------------------------------- #
def validate_german():
    from sklearn.datasets import fetch_openml

    data = fetch_openml("credit-g", version=1, as_frame=True)
    df = data.frame.copy()
    # target: good = 1 (credit granted / low risk), bad = 0
    y = (df["class"].astype(str) == "good").astype(int).to_numpy()

    # Protected attribute: age dichotomised at 25 (Kamiran & Calders 2009 convention).
    age = pd.to_numeric(df["age"])
    sensitive = np.where(age > 25, "older", "younger")
    sensitive = pd.Series(sensitive)

    feature_cols = [c for c in df.columns if c not in ("class", "age")]
    categorical = [c for c in feature_cols if str(df[c].dtype) in ("category", "object")]
    numeric = [c for c in feature_cols if c not in categorical]
    X = df[feature_cols]

    X_tr, X_te, y_tr, y_te, _, s_te = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    y_pred = _fit_predict(
        X_tr, y_tr, X_te, numeric, categorical,
        LogisticRegression(max_iter=2000),
    )
    results = BiasTestSuite().run(y_te, y_pred, s_te.reset_index(drop=True), "age")
    return _summarise(results, "demographic_parity_difference")


CASES = [
    ("UCI Adult Income", "sex", "demographic_parity_difference", (0.15, 0.25), validate_adult),
    ("ProPublica COMPAS", "race", "equalized_odds_difference", (0.15, 0.25), validate_compas),
    ("UCI German Credit", "age", "demographic_parity_difference", (0.05, 0.20), validate_german),
]


def main():
    print("=" * 70)
    print("PHASE 1 WEEK 3 — BENCHMARK VALIDATION")
    print("=" * 70)
    summary = []
    for name, attr, metric, (lo, hi), fn in CASES:
        print(f"\n### {name}  (protected: {attr})")
        headline, table = fn()
        print(table)
        in_range = lo <= headline <= hi
        verdict = "PASS" if in_range else "OUT OF RANGE"
        print(f"    -> {metric} = {headline:.4f}  expected [{lo}, {hi}]  {verdict}")
        summary.append((name, metric, headline, lo, hi, in_range))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for name, metric, val, lo, hi, ok in summary:
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<20} {metric:<32} {val:.4f}  [{lo}, {hi}]")
    print("=" * 70)
    print("ALL BENCHMARKS IN RANGE" if all_ok else "SOME BENCHMARKS OUT OF RANGE")

    with open("notebooks/validation_output.json", "w") as fh:
        json.dump(
            [
                {"dataset": n, "metric": m, "value": round(v, 4),
                 "expected_low": lo, "expected_high": hi, "in_range": ok}
                for n, m, v, lo, hi, ok in summary
            ],
            fh,
            indent=2,
        )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
