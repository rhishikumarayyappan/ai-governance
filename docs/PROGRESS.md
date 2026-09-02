# AI Governance Platform — Progress Log

## Current Status
- **Phase 2 — IN PROGRESS.** Significance testing DONE (gap 1.4). Bootstrap
  confidence intervals DONE (gap 1.3).
- Phase 1 COMPLETE (Weeks 1–2 + Week 3 benchmark validation).
- Working to **BUILD_PLAN v2.0** (10 phases, 9 gap categories, `docs/GAP_CHECKLIST.md` is the authoritative tracker)
- Overall health: **Green**
- **34 tests passing, 0 failures** (~54s — bootstrap loops dominate)
- Last updated: 2026-09-02
- Next: **Bonferroni + Benjamini-Hochberg correction** (gap 1.1), then
  **reliability scoring** (gap 1.8, N-run SD three-tier flag; "insufficient_data"
  blocks a verdict), both added to `statistics.py`. Then Component 2.2
  (`THRESHOLDS.md`, gap 1.2), 2.3 (Simpson's paradox, gaps 1.5/9.9), 2.4
  (`tensions.py`, gap 1.9).

---

## Session Log

### Session 8 — 2026-09-02 — Phase 2: Bootstrap Confidence Intervals (COMPLETE)

**Scope:** bootstrap CIs only. Added to `governance/testing/statistics.py`;
existing `significance_test` functions untouched. No changes to `bias.py` /
`engine.py` / API. (Note: the prompt called this "Component 2.2"; the plan files
bootstrap CIs under Component 2.1's "Confidence Intervals" subsection. `THRESHOLDS.md`
is the real Component 2.2. Used the prompt's commit message verbatim as instructed.)

**Built:**
- `bootstrap_confidence_interval(metric_fn, y_true, y_pred, sensitive_features,
  n_iterations=1000, confidence_level=0.95, random_state=None) -> BootstrapResult`
  - Percentile bootstrap. **One** index draw with replacement per iteration,
    applied to all three arrays → each synthetic row keeps its real
    `(group, prediction, label)` triple (the joint relationship the metric
    measures). `sensitive_features` coerced to a positional array first so a
    pandas index can't misalign the draw.
  - Percentile bounds generalised: `lower = (1-c)/2·100`, `upper = 100-lower`.
  - Single seed threaded through one `np.random.default_rng` → same
    `random_state` gives byte-identical results.
- `BootstrapResult` — `point_estimate`, `ci_lower`, `ci_upper`,
  `confidence_level`, `n_iterations`, `bootstrap_distribution_summary`
  (`{min, max, std}` only — no mean/median, so `point_estimate` is unambiguously
  "the" number), `n_valid_iterations`, `n_skipped_single_group`,
  `reliability_warning`, `skip_breakdown`.
- **Failure handling:** single-group resamples AND any `metric_fn` exception AND
  non-finite values are **skipped, not fatal**, each counted by reason in
  `skip_breakdown` (`{"single_group": n}`, `{"ValueError": n}`, …). >5% skipped
  → `reliability_warning` set with the breakdown in prose. All iterations failed
  → `ci_lower/upper = nan` + a specific warning ("all N iterations failed — CI
  could not be computed, sample size or class balance likely insufficient"),
  never an exception. Consistent with "insufficient_data degrades, doesn't crash".
- **Five metric wrappers** to the `(y_true, y_pred, sensitive_features) -> float`
  shape, calling fairlearn/sklearn directly (never `bias.py`):
  `demographic_parity_wrapper` (abs, tolerates y_true=None),
  `equalized_odds_wrapper`, `equal_opportunity_wrapper` (both abs, require
  y_true), `predictive_parity_wrapper` (max-min group precision, mirrors
  `bias.py` exactly including `zero_division=0` — so an empty-positive group
  scores 0.0, does not raise), `individual_fairness_wrapper` (raw overall
  accuracy 0-1, **no abs** — matches `bias.py`'s inverted-threshold metric 5).
  `METRIC_WRAPPERS` dict keyed by the `bias.py` metric names.

**Design decision — 4th top-level field `skip_breakdown`, approved by
Rhishikumar.** Confirmed interface had 3 new fields; shipped 4. Reason: point 2
asked for exception types to be recorded so a developer can tell "single group"
(benign) from "metric error" (investigate). A `reliability_warning` string only
exists above the 5% threshold and must be parsed when present — so skips at e.g.
3% would be invisible as to cause. `skip_breakdown` is a structured dict, always
populated (`{}` when clean). Approved as "better engineering than a string that
only shows up sometimes", not scope creep.

**Verification:**
- Cross-validation (Step 5): n=5000, simulated true gap 0.20 → point estimate
  **0.2081**, 95% CI **[0.1790, 0.2341]**, width 0.055. The bootstrap std
  (0.0140) matches the analytic binomial difference-SE (~0.014) — this is the
  real evidence the *spread* of the resampling is correct, stronger than the
  point estimate matching the true gap. CI width ran ~0.055, a bit above the
  prompt's "0.02–0.04" guess — expected: that guess was ~1.5× tight for
  n≈2500/group.
- `tests/test_statistics.py`: 6 new tests (14 in file, 34 total). CI contains
  point estimate; narrow CI on large data (<0.05) + no false warning; small-data
  CI >3× wider; all 5 wrappers valid; fixed seed byte-identical incl. summary
  dict; single-group resamples skipped/tracked/flagged and reconciled.
- `pytest tests/ -v` → **34 passed, 0 failed** (~54s). Health endpoint 200.

**Commits:** `d198332` (statistics.py + test_statistics.py). Pushed at EOD.

**Exact next step:** multiple-comparison correction (gap 1.1) added to
`statistics.py` — Bonferroni (divide alpha by n_tests, conservative, default)
and Benjamini-Hochberg FDR (offered as an option); store both raw `threshold`
and `corrected_threshold`; the compliance mapper (Phase 5) will always use the
corrected one. `statsmodels.stats.multitest.multipletests` is the reference to
validate against (may add `statsmodels` as a dep, or hand-implement + validate).
Then reliability scoring (gap 1.8). Still do NOT touch `bias.py` / `engine.py` /
API until the explicit wiring sub-step later in Phase 2.

---

### Session 7 — 2026-09-02 — Phase 2 Component 2.1: Significance Testing (COMPLETE)

**Scope:** significance testing only. No changes to `bias.py`, `engine.py`, or
the API — verified. New file + new test file only.

**Built — `governance/testing/statistics.py`:**
- `significance_test(y_true, y_pred, sensitive_features, method="auto", *,
  n_permutations=1000, random_state=None, cross_check=True) -> SignificanceResult`
- `SignificanceResult` dataclass: `test_used`, `statistic`, `p_value`,
  `significant` (p < 0.05, raw — correction is a later component), `assumptions_met`,
  `sample_sizes` (per group), `detail` (dict).
- Four tests:
  1. **chi-squared** — `scipy.stats.chi2_contingency` with defaults (Yates
     correction on for 2×2). Matches a direct `chi2_contingency(crosstab)` call
     **bit-for-bit** (delta 0.00e+00 on the Step 5 cross-validation script).
  2. **Fisher's exact** — `scipy.stats.fisher_exact`, two-sided, 2×2 only.
     Matches scipy docs example to 1e-12. Raises `ValueError` pointing to
     `permutation` for >2 groups.
  3. **two-proportion z-test** — implemented manually (pooled SE, `norm.sf`).
     Matches hand calculation to 1e-12. `assumptions_met=False` when
     n·p̂ or n·(1−p̂) ≤ 5 for either group.
  4. **permutation test** — 1,000 label shuffles, statistic is fairlearn's
     `demographic_parity_difference` (imported direct from fairlearn, NOT from
     `bias.py` — siblings never import each other). p-value uses **add-one
     smoothing** `(1+k)/(1+n)` (Phipson & Smyth 2010), not raw `k/n`.
- `method="auto"`: >2 groups → permutation; 2 groups + any expected cell <5 →
  Fisher; else chi-squared.
- **Always-on permutation cross-check** (`cross_check=True` default): a
  permutation p-value is attached to `detail["permutation_p_value"]` on every
  call, whatever the primary test, plus `cross_check_divergence`. Opt-out is
  `cross_check=False` — deliberate and review-visible.

**Tests — `tests/test_statistics.py`: 28 passed, 0 failed** (20 prior + 8 new).
Six behaviour tests per the brief + two guard-rail tests (non-binary y_pred
rejected; 2-group-only method with 3 groups raises). Cross-validated against
scipy directly (Step 5). Health endpoint still 200.

**Three design decisions — all reviewed and approved by Rhishikumar, rationale
for the record:**
1. **28 tests not 26.** The two guard-rail tests verify confirmed design points
   2 and 3 (binary-only input; 2-group-only parametric tests). Testing a
   confirmed design point is doing the job properly, not scope creep. Kept as
   standalone tests — clearer than buried assertions.
2. **Add-one smoothing on the permutation p-value.** Raw `k/n` can return
   exactly `0.0`, which asserts a fairness gap "could never occur by chance" —
   statistically indefensible and dangerous to state as fact in a
   legal/regulatory context. `(1+k)/(1+n)` is bounded away from zero by design.
   Phipson & Smyth 2010 is the standard citation (in the code comment).
   **TODO for Component 2.2:** add one line to `THRESHOLDS.md` noting permutation
   p-values are bounded away from zero by design.
3. **`assumptions_met=False` on auto-mode Fisher fallback, `True` on explicit
   Fisher call.** The flag doesn't answer "is Fisher valid" (always yes) — it
   answers "did the caller get the test they implicitly expected, or did the
   system have to correct course." A compliance reviewer needs to see the red
   flag along the way, not just the final validity.

**Known cost (not a problem to solve):** the always-on cross-check adds ~3.7s
per `significance_test` call (1,000 × fairlearn's `demographic_parity_difference`,
which is slow). Test suite went 1.7s → 16.8s. Acceptable — this is a compliance
tool running test suites, not a live-request service. Vectorising the
permutation loop would reopen the "does it match fairlearn exactly" question we
deliberately closed by importing the metric directly. **Do not optimise unless
it becomes a real blocker.**

**Commits:** `e383182` (statistics.py + test_statistics.py). Pushed at EOD.

**Exact next step:** continue Phase 2 Component 2.1 — bootstrap confidence
intervals (1,000 iterations, 2.5/97.5 percentiles) on all 5 fairlearn metrics,
added to `statistics.py`. Then Bonferroni + Benjamini-Hochberg correction (store
raw `threshold` and `corrected_threshold`). Then reliability scoring (N-run SD,
three-tier flag; "insufficient_data" blocks a verdict). `statsmodels` may be
added here (`multipletests`) or corrections done by hand against a scipy
reference. Still do NOT touch `bias.py` / `engine.py` / API until the wiring
sub-step, which is explicitly called out later in Phase 2.

---

### Session 6 — 2026-09-02 — Phase 1 Week 3: Benchmark Validation (COMPLETE)

**Context:** BUILD_PLAN was replaced with v2.0 this session — phase count 6 → 10,
LLM testing elevated to Phase 3, statistical rigour is now Phase 2, plus new
`docs/GAP_CHECKLIST.md` tracking 94 gaps across 9 categories (87 closed in plan,
7 deferred with risk statements). Phase 1 Week 3 validation is unchanged from
v1.0 and was the assigned task.

**What was done (no `governance/` code changed — validation only):**
- `notebooks/validate_phase1.py` — runs the existing `BiasTestSuite` against
  three public datasets, checks the headline metric against BUILD_PLAN v2.0
  ranges, writes `notebooks/validation_output.json`.
  - UCI Adult Income (`fairlearn.datasets.fetch_adult`), sex → LogisticRegression
  - ProPublica COMPAS (live CSV from propublica/compas-analysis), race, ProPublica
    row filter, African-American vs Caucasian → RandomForest
  - UCI German Credit (`fetch_openml("credit-g")`), age dichotomised at 25 →
    LogisticRegression
- `docs/VALIDATION_RESULTS.md` — full write-up with per-dataset metric tables,
  literature anchors, and known limitations (no CIs yet, thresholds not yet
  justified, datasets frozen in time — Gap 9.5).

**Results — all three headline metrics in published range:**
| Dataset | Metric | Value | Expected |
|---|---|---|---|
| Adult Income | demographic_parity_difference | 0.1745 | 0.15–0.25 |
| COMPAS | equalized_odds_difference | 0.1752 | 0.15–0.25 |
| German Credit | demographic_parity_difference | 0.1149 | 0.05–0.20 |

`pytest tests/ -q` → **20 passed, 0 failed** (unchanged).

**Deferred Register reviewed** (mandatory at phase completion): no trigger
conditions met.

**Exact next step:** Begin **Phase 2 — Statistical Rigour Layer**. Open a fresh
session, read `docs/BUILD_PLAN.md` (Phase 2 section), `docs/GAP_CHECKLIST.md`
(Category 1), and `docs/PROGRESS.md`. Build `governance/testing/statistics.py`
first (Component 2.1: chi-squared / Fisher / z-test / permutation + bootstrap CI
+ Bonferroni/BH + reliability). Then `THRESHOLDS.md`, then Simpson's paradox and
`governance/compliance/tensions.py`. Phase 2 adds `scipy.stats` + `statsmodels`.

---

### Session 1 — 2026-09-01 — Phase 0 Foundation (COMPLETE)
- Environment: Homebrew (pre-existing), pyenv 2.8.4 installed + wired into `~/.zshrc`,
  Python 3.11.9 via pyenv (global), Poetry 2.1.4 with in-project venv.
- Project scaffold per BUILD_PLAN.md. Phase 0 deps only (FastAPI, SQLAlchemy,
  Pydantic, uvicorn; dev: pytest, httpx).
- `governance/config.py`, `governance/db/database.py` (SQLite, WAL mode, FK on,
  `SessionLocal`, `get_db`, `init_db`), `governance/db/models.py` (4 ORM models).
- `governance/registry/` (schemas/service/router), `governance/main.py`
  (`/health` + registry router, `init_db()` at import).
- Git init, branch `main`, GitHub remote `origin`, first commits pushed.

### Session 2 — 2026-09-01 — Phase 1 Week 1: Model Adapter Layer
- Added scikit-learn, pandas, numpy, scipy (upper-bounded for Python 3.11).
- `governance/testing/adapters.py`: `ModelAdapter` interface, `SklearnAdapter`,
  `PickleAdapter` (subclasses Sklearn → identical results), `APIAdapter`
  (stdlib `urllib`, no new dep), `load_adapter()` factory.
- `predict_proba()` returns 2D array or `None` cleanly, never raises.
- `tests/test_adapters.py` — 5 tests. Committed `4d94239`.

### Session 3 — 2026-09-01 — Phase 1 Week 2 Component 1: BiasTestSuite
- Added fairlearn 0.14.0.
- `governance/testing/bias.py`: `BiasTestResult` dataclass (4dp rounding,
  strict status whitelist, `passed` property), `BiasTestSuite` with
  `DEFAULT_THRESHOLDS` class attribute, `_get_status()` centralised threshold
  logic, `run()` returns exactly 5 results in fixed order.
- 5 metrics: demographic parity / equalized odds / equal opportunity (fairlearn),
  predictive parity (manual sklearn `precision_score`), individual fairness
  (fairlearn `MetricFrame` accuracy, **inverted threshold** — fail below 0.80).
- Metric 1 `detail` carries per-group positive rates as a flat float dict.
- `tests/test_bias.py` — 5 tests. Committed `f5dc37a`.

### Session 4 — 2026-09-01 — Phase 1 Week 2 Component 2: Engine Orchestrator
- `governance/testing/engine.py`: `run_bias_tests(system_id, model_source,
  X_test, y_test, protected_attributes, config=None) -> str` and
  `get_run_results(run_id) -> list[dict]`.
- Staged: validate system_id (ValueError before any DB write) → open TestRun
  "running" → load model → per-attribute BiasTestSuite → save TestResult rows →
  close "complete". Model-load failure and top-level safety net both mark the
  run "failed" (fresh session, idempotent) — a run is never stuck in "running".
- Added `get_session()` context manager to `governance/db/database.py`.
- `tests/conftest.py`: function-scoped `test_db` fixture — per-test temp SQLite,
  own engine, fresh tables, `SessionLocal` monkeypatched. Real DB never touched.
- `tests/test_engine.py` — 4 tests. Failure paths verified manually. Committed `4306bec`.

### Session 5 — 2026-09-01 — Phase 1 Week 2 Component 3: API endpoints (TODAY)

**What was built today**
- `governance/testing/schemas.py` — 3 Pydantic response models
  (`TestRunResponse`, `TestRunResultsResponse`, `TestResultItem`), all
  `from_attributes=True`.
- `governance/testing/router.py` — 3 API endpoints (prefix `/api/v1`, tag
  `testing`).
- Router mounted in `governance/main.py`.
- `tests/test_api_testing.py` — 6 tests (FastAPI TestClient + temp DB).
- New dependency: `python-multipart` 0.0.32 (required by FastAPI for
  form/file uploads).

**What works right now (cumulative)**
- Full project structure and SQLite database (WAL mode, 4 tables auto-created).
- `GET /health` endpoint.
- `GET` / `POST /api/v1/systems` — register and list AI systems.
- Model adapter layer — `SklearnAdapter`, `PickleAdapter`, `APIAdapter`,
  `load_adapter()`.
- `BiasTestSuite` — 5 fairness metrics with threshold logic, inverted
  threshold for individual fairness, per-group `detail`.
- Engine orchestrator — creates `TestRun`, runs tests, saves `TestResult`
  rows, handles all error cases (never leaves a run "running").
- `POST /api/v1/test-runs` — accepts model upload (.pkl) and CSV, runs the
  full bias test, returns a completed `TestRunResponse` (201).
- `GET /api/v1/test-runs/{id}` — run status.
- `GET /api/v1/test-runs/{id}/results` — 5 results.
- Full test isolation — `ai_governance.db` is never touched by the test suite.

**What does NOT work yet**
- Week 3 validation against published benchmarks (Adult Income, COMPAS, German
  Credit).
- Compliance mapper (Phase 2).
- Streamlit dashboard (Phase 3).
- PDF reports and SHAP (Phase 4).
- SDK and demo (Phase 5).

**Decisions made today**
- Synchronous handlers (plain `def`) — consistent with the no-async rule;
  `run_bias_tests` is fully synchronous, so `await` on uploads adds nothing.
  Uploads read via `UploadFile.file.read()`.
- Comma-separated string for `protected_attributes` — more reliable than
  repeated `Form` fields across HTTP clients. Split on comma in the handler.
- `ValueError` from the engine → HTTP 404; all other exceptions → HTTP 500
  with `str(e)` as the `detail`, so every error response is readable (FastAPI's
  default 500 has no body detail). The engine has already written "failed" to
  the `TestRun` before re-raising, so there is no double-write risk.
- Lazy `app` import inside the test `client` fixture — ensures `main.py`'s
  module-level `init_db()` targets the temp database, not the real one.
- 422 status passed as the literal `422` in the router (`HTTP_422_
  UNPROCESSABLE_ENTITY` is deprecated / renamed in current Starlette).

**Problems encountered and solved (Phase 1 to date)**
1. Newest numpy / scipy / pandas releases dropped Python 3.11. Fixed by pinning
   upper bounds (`numpy<2.5`, `scipy<1.18`, `pandas<3.1`); Poetry then selects
   the newest 3.11-compatible versions (numpy 2.4.6, scipy 1.17.1, pandas 3.0.5).
2. `attribute_name` could not live in metric 1's `detail` (a test requires every
   value there to be a float 0–1). Put it in the `detail` of metrics 2–5 instead.
3. Smoke test referenced `get_session()` which didn't exist — added it to
   `database.py` as the standard non-API session helper.
4. `RiskTier.HIGH` vs the actual lowercase enum members — used `RiskTier.high`.
5. `X_test.drop(columns=protected_attributes)` raised `KeyError` on unknown
   column names — added `errors="ignore"` (the per-attribute existence check
   still does the warn-and-skip).
6. Standalone scripts hit "no such table" because only the FastAPI app calls
   `init_db()` — standalone entry points must call it explicitly.
7. pytest tried to collect the `TestRun` / `TestResult` ORM classes as test
   classes — import the `models` module, not the names, in test files.
8. FastAPI form/file uploads need `python-multipart` — added it.
9. API test isolation: `main.py` runs `init_db()` at import, which would hit the
   real DB — solved with the lazy `app` import in the `client` fixture.

**Test results**
`pytest tests/ -v` → **20 passed, 0 failed** (1 non-blocking StarletteDeprecation
warning about httpx/TestClient, present since Phase 0).

**Exact next step**
Start **Phase 1 Week 3 — validation against three published datasets**. Open a
fresh session, read `docs/BUILD_PLAN.md` and `docs/PROGRESS.md`, then run the
Week 3 validation prompt. **Do not write new code in Week 3** — only run the
existing engine against real data (UCI Adult Income, COMPAS/ProPublica, UCI
German Credit) and verify the numbers match published benchmarks within 5%.

---

## Phase Completion Checklist

### Phase 0 — Foundation — COMPLETE
- [x] pyenv + Python 3.11 installed
- [x] Poetry installed
- [x] Project folder and structure created
- [x] All dependencies installed via Poetry *(phased plan)*
- [x] SQLite database with 4 tables created
- [x] GET /health endpoint works
- [x] GET /api/v1/systems endpoint works
- [x] POST /api/v1/systems endpoint works
- [x] Git repository created with first commit
- [x] Pushed to GitHub (github.com/rhishikumarayyappan/ai-governance)

### Phase 1 — Testing Engine

BUILD_PLAN v2.0 Phase 1 exit criteria — **ALL MET (2026-09-02)**:
- [x] Adult Income demographic_parity_difference in 0.15–0.25 → 0.1745
- [x] COMPAS equalized_odds_difference in 0.15–0.25 → 0.1752
- [x] German Credit demographic_parity_difference in 0.05–0.20 → 0.1149
- [x] docs/VALIDATION_RESULTS.md created
- [x] pytest → 20 passed, 0 failed
- [x] All 5 metrics have pytest tests with hardcoded expected values
      (`tests/test_bias.py`)
- [x] `governance/testing/engine.py` (Engine Orchestrator) built and working
- [x] `tests/test_adapters.py` — all 5 tests pass
- [x] `tests/test_bias.py` — all 5 tests pass
- [x] Results save correctly to SQLite
- [x] POST /api/v1/test-runs creates a TestRun in SQLite and triggers a run
- [x] GET /api/v1/test-runs/{id}/results returns the saved results

Progress: **9 / 9 done — Phase 1 COMPLETE.**

Component status:
- [x] Component 1 — Model adapter layer (Sklearn, Pickle, API)
- [x] Component 2 — BiasTestSuite with 5 metrics + Engine orchestrator
- [x] Component 3 — API endpoints (POST /test-runs, GET status, GET results)
- [x] Week 3 — Validated against UCI Adult Income (0.1745, in range)
- [x] Week 3 — Validated against COMPAS (0.1752, in range) + German Credit (0.1149, in range)

### Phase 2 — Statistical Rigour Layer (v2.0) — IN PROGRESS

Component 2.1 — Statistical Testing Module (`governance/testing/statistics.py`):
- [x] Significance testing — chi-squared, Fisher's exact, z-test, permutation
- [x] "auto" test selection by group count + expected cell size
- [x] `assumptions_met` flag; always-on permutation cross-check
- [x] Validated against scipy reference outputs (bit-identical on chi-squared)
- [x] Bootstrap confidence intervals (1,000 iterations) — generic
      `bootstrap_confidence_interval` + 5 metric wrappers + `METRIC_WRAPPERS`
- [x] `tests/test_statistics.py` — 14 tests (8 significance + 6 bootstrap)
- [ ] Bonferroni + Benjamini-Hochberg correction; `corrected_threshold`
- [ ] Reliability scoring (N-run SD, three-tier flag); "insufficient_data" blocks verdict

Component 2.2 — `THRESHOLDS.md` (not started). **Carry-over TODO:** note that
permutation p-values are bounded away from zero by design (add-one smoothing).
Component 2.3 — Simpson's paradox detection (not started).
Component 2.4 — `governance/compliance/tensions.py` (not started).

Gap tracker: **1.3, 1.4 ✅ closed.** 1.1, 1.2, 1.5, 1.8, 1.9, 9.9 still open — see
`docs/GAP_CHECKLIST.md`.

---

> **NOTE (2026-09-02):** the phase list below is the superseded v1.0 structure.
> Under BUILD_PLAN v2.0, Phase 2 is the Statistical Rigour Layer and the
> compliance mapper moves to Phase 5. `docs/GAP_CHECKLIST.md` is now the
> authoritative tracker. The rows below are retained only for history.

### Phase 2 (v1.0, superseded) — Compliance Mapper
- [ ] eu_ai_act.json rules file complete (Articles 9,10,13,14,15)
- [ ] gdpr.json rules file complete (Articles 22, 25)
- [ ] ComplianceMapper engine built
- [ ] Scorer and aggregator built
- [ ] Compliance scores save to SQLite

### Phase 3 — Dashboard
- [ ] Page 1: Registry working
- [ ] Page 2: Run Tests working
- [ ] Page 3: Results/Dashboard working
- [ ] Page 4: Reports working
- [ ] Non-technical person test passed

### Phase 4 — PDF + Explainability
- [ ] ReportLab PDF generation working
- [ ] PDF looks professional
- [ ] SHAP integration working (1000 row cap)
- [ ] Download works in Streamlit

### Phase 5 — SDK + Demo
- [ ] LocalRunner SDK built
- [ ] demo.py runs in under 5 minutes
- [ ] Works fully offline
- [ ] Demo shown to first prospect
