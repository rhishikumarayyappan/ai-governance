# Phase 1 Week 3 — Benchmark Validation Results

**Date:** 2026-09-02
**Code version:** `68371ac` (BiasTestSuite unchanged from Phase 1 Week 2 — no `governance/` code written in Week 3)
**Script:** `notebooks/validate_phase1.py` (`poetry run python notebooks/validate_phase1.py`)
**Machine:** MacBook Air M4, Python 3.11.9

**Library versions:** scikit-learn 1.9.0 · numpy 2.4.6 · pandas 3.0.5 · scipy 1.17.1 · fairlearn 0.14.0

---

## Purpose

Confirm that the existing `BiasTestSuite` produces fairness numbers consistent
with the published literature when run against the three datasets named in the
Phase 1 exit criteria. This is a sanity check on the metric implementations, not
a model-quality exercise — the models are deliberately ordinary.

---

## Result summary

| Dataset | Protected attribute | Headline metric | Value | Expected range (BUILD_PLAN v2.0) | Verdict |
|---|---|---|---|---|---|
| UCI Adult Income | `sex` | `demographic_parity_difference` | **0.1745** | 0.15 – 0.25 | ✅ in range |
| ProPublica COMPAS | `race` (African-American vs Caucasian) | `equalized_odds_difference` | **0.1752** | 0.15 – 0.25 | ✅ in range |
| UCI German Credit | `age` (>25 vs ≤25) | `demographic_parity_difference` | **0.1149** | 0.05 – 0.20 | ✅ in range |

**All three headline metrics fall inside the published ranges.**

---

## Detail

### 1. UCI Adult Income — sex

- **Source:** `fairlearn.datasets.fetch_adult` (the 1994 US Census extract).
- **Setup:** rows with missing values dropped; target `class` = income >50K;
  protected columns excluded from features; 70/30 stratified split;
  `LogisticRegression(max_iter=2000)` on standardised numeric + one-hot categorical.
- **Literature anchor:** demographic parity gap by sex on Adult is routinely
  reported at ~0.17–0.19 (men predicted high-income far more often than women).

| Metric | Value | Status |
|---|---|---|
| demographic_parity_difference | 0.1745 | fail |
| equalized_odds_difference | 0.1054 | fail |
| equal_opportunity_difference | 0.1054 | fail |
| predictive_parity_difference | 0.0019 | pass |
| overall_accuracy_floor | 0.8440 | pass |

### 2. ProPublica COMPAS — race

- **Source:** `compas-scores-two-years.csv` from the ProPublica
  `compas-analysis` repository, fetched over HTTPS.
- **Setup:** ProPublica's own row filter (`days_b_screening_arrest` within ±30,
  `is_recid != -1`, `c_charge_degree != "O"`, `score_text != "N/A"`); restricted
  to African-American and Caucasian; target = `two_year_recid` (actual
  reoffending); 70/30 stratified split;
  `RandomForestClassifier(n_estimators=200)`.
- **Literature anchor:** Angwin et al. (2016) found a false-positive-rate gap of
  ~0.21 (44.8% vs 23.5%) and a false-negative-rate gap of ~0.20 between the two
  groups. `equalized_odds_difference` is the larger of the TPR/FPR gaps, so a
  value in the high-0.1s to low-0.2s is expected.

| Metric | Value | Status |
|---|---|---|
| demographic_parity_difference | 0.1741 | fail |
| equalized_odds_difference | 0.1752 | fail |
| equal_opportunity_difference | 0.1752 | fail |
| predictive_parity_difference | 0.1179 | fail |
| overall_accuracy_floor | 0.6263 | fail |

### 3. UCI German Credit — age

- **Source:** `sklearn.datasets.fetch_openml("credit-g", version=1)`.
- **Setup:** target `class` = "good" credit; protected attribute is `age`
  dichotomised at 25 (Kamiran & Calders 2009 convention); `age` itself excluded
  from features; 70/30 stratified split; `LogisticRegression(max_iter=2000)`.
- **Literature anchor:** demographic parity gap by the age>25 split on German
  Credit is generally reported at ~0.10–0.15; the model favours older applicants.

| Metric | Value | Status |
|---|---|---|
| demographic_parity_difference | 0.1149 | fail |
| equalized_odds_difference | 0.1103 | fail |
| equal_opportunity_difference | 0.1103 | fail |
| predictive_parity_difference | 0.1569 | fail |
| overall_accuracy_floor | 0.7467 | fail |

---

## Notes and known limitations

- **`status` values are not meaningful here.** With the default 0.10 threshold,
  almost every metric on these deliberately-biased research datasets reads
  "fail" / "warn". Week 3 validates the *numbers*, not the verdicts. Threshold
  justification and configurability land in Phase 2 (see `THRESHOLDS.md`, and
  Gap 1.2).
- **No confidence intervals yet.** These are bare point estimates. Phase 2
  (Gap 1.3) attaches a bootstrap 95% CI to every value; this validation should
  be re-run afterwards with CIs shown, per the Phase 2 exit criteria.
- **Datasets are frozen in time** (Census 1994, COMPAS 2013–2014, German Credit
  1990s) — Gap 9.5. They are the field-standard fairness benchmarks and are
  appropriate for validating metric *implementations*, but they are not
  representative of any live population. A modern supplementary dataset is a
  Phase 5 action item.
- **Reproducibility:** fixed `random_state=42` throughout. `fetch_adult` and
  `fetch_openml` cache under `~/scikit_learn_data`; the COMPAS CSV is fetched
  live each run. Re-running reproduces the values above exactly.
- **Metric 5 renamed 2026-09-02 (gap 9.12):** the tables above originally listed
  `individual_fairness_score`; it is now `overall_accuracy_floor`, which is what
  it always computed (`MetricFrame.overall` — the model's overall accuracy). Pure
  rename — re-running `validate_phase1.py` reproduces 0.8440 / 0.6263 / 0.7467
  **bit-identically**.

---

## Phase 1 exit criteria — status

- [x] Adult Income `demographic_parity_difference` in 0.15–0.25 → 0.1745
- [x] COMPAS `equalized_odds_difference` in 0.15–0.25 → 0.1752
- [x] German Credit `demographic_parity_difference` in 0.05–0.20 → 0.1149
- [x] `docs/VALIDATION_RESULTS.md` created
- [x] `pytest` → 20 passed, 0 failed

**Phase 1 is complete. Phase 2 (Statistical Rigour Layer) may begin.**
