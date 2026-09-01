# AI Governance Platform — Progress Log

## Current Status
- **Phase 1 Week 2 — COMPLETE**
- Overall health: **Green**
- **20 tests passing, 0 failures**
- Last updated: 2026-09-01
- Next: Phase 1 Week 3 — validation against published benchmarks (no new code)

---

## Session Log

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

BUILD_PLAN.md Phase 1 exit criteria:
- [ ] BiasTestSuite.run() produces numbers within 5% of published Adult Income
      benchmarks — **PENDING (Week 3)**
- [ ] COMPAS false positive rate gap matches ProPublica within 5%
      (African-American vs Caucasian) — **PENDING (Week 3)**
- [x] All 5 metrics have pytest tests with hardcoded expected values
      (`tests/test_bias.py`)
- [x] `governance/testing/engine.py` (Engine Orchestrator) built and working
- [x] `tests/test_adapters.py` — all 5 tests pass
- [x] `tests/test_bias.py` — all 5 tests pass
- [x] Results save correctly to SQLite
- [x] POST /api/v1/test-runs creates a TestRun in SQLite and triggers a run
- [x] GET /api/v1/test-runs/{id}/results returns the saved results

Progress: **7 / 9 done, 2 pending (both Week 3 benchmark validation).**

Component status:
- [x] Component 1 — Model adapter layer (Sklearn, Pickle, API)
- [x] Component 2 — BiasTestSuite with 5 metrics + Engine orchestrator
- [x] Component 3 — API endpoints (POST /test-runs, GET status, GET results)
- [ ] Week 3 — Validated against UCI Adult Income (within 5% of published)
- [ ] Week 3 — Validated against COMPAS dataset (within 5% of ProPublica)

### Phase 2 — Compliance Mapper
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
