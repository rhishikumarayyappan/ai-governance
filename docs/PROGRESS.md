# AI Governance Platform — Progress Log

## Current Status
- Phase: 0 — Foundation — **COMPLETE** (all exit criteria met and verified)
- Started: September 2026
- Last updated: 2026-09-01

---

## Session Log

### Session 1 — 2026-09-01

**What was done**

Environment:
- Homebrew: already installed (5.1.12, Apple Silicon).
- pyenv: installed via Homebrew (2.8.4). Added init block to `~/.zshrc`
  (`PYENV_ROOT`, PATH, `pyenv init - zsh`). Verified it persists in a fresh
  interactive shell.
- Python: 3.11.9 installed via pyenv and set as the global default
  (`python --version` → 3.11.9, from the pyenv shim).
- Poetry: already installed (2.1.4, official installer at `~/.local/bin/poetry`).
  Set `virtualenvs.in-project = true` so the env lives at `./.venv`.

Project:
- Poetry project initialised in the existing `~/ai-governance` folder. Python
  pinned to `>=3.11,<3.12`. Venv created at `./.venv` on Python 3.11.9.
- Full folder/file structure created exactly per BUILD_PLAN.md (governance
  package with db/registry/testing/compliance/reporting, dashboard/pages, sdk,
  tests/fixtures, notebooks). Most files are empty stubs for later phases.
- Dependencies installed (Phase 0 only — see decision below): fastapi 0.141.1,
  uvicorn 0.52.4, sqlalchemy 2.0.52, pydantic 2.13.5, pydantic-settings 2.15.0,
  plus dev: pytest 8.4.2, httpx 0.28.1. `poetry install` clean, no conflicts.
  `poetry.lock` committed.
- `governance/config.py`: pydantic-settings config (env prefix `AIGOV_`,
  optional `.env`). Default DB path `~/ai-governance/ai_governance.db`.
- `governance/db/database.py`: SQLite engine, `SessionLocal` factory, `get_db()`
  FastAPI dependency, `init_db()`. WAL mode + `foreign_keys=ON` set on every
  connection via a `connect` event listener. `check_same_thread=False` (safe:
  short-lived per-request sessions).
- `governance/db/models.py`: four ORM models — `AISystem`, `TestRun`,
  `TestResult`, `ComplianceScore` — with all fields, JSON columns, string UUID
  PKs, FKs with `ondelete=CASCADE`, and relationships wired both directions.
  `risk_tier` is a real enum (`RiskTier`); other category fields are strings per
  the plan.
- `governance/registry/` — `schemas.py` (`AISystemCreate`, `AISystemRead`),
  `service.py` (`list_systems`, `create_system`), `router.py` (APIRouter,
  prefix `/api/v1`).
- `governance/main.py`: FastAPI app, calls `init_db()` at startup (synchronous),
  CORS middleware (localhost origins for the future Streamlit dashboard),
  `GET /health`, includes the registry router.
- `.gitignore`, `README.md`, `.env.example` written.
- Git initialised, branch `main`, first commit `657f70a`.

**What works (verified)**
- `GET /health` → `{"status": "ok", "version": "0.1.0"}` (200).
- `GET /api/v1/systems` → `[]` when empty (200).
- `POST /api/v1/systems` → 201, persists the record with a generated UUID and
  `created_at`; created systems then appear in `GET /api/v1/systems`.
- Tested both via FastAPI TestClient and against a live `uvicorn` server (curl).
- SQLite confirmed in WAL mode (`PRAGMA journal_mode` → `wal`); data persists to
  `ai_governance.db` and reads back correctly from a separate connection.
- App tables auto-create on startup: `ai_systems`, `test_runs`, `test_results`,
  `compliance_scores`.

**What does NOT work yet / not built**
- No automated pytest tests yet (test files are empty stubs). Manual + TestClient
  verification only this session.
- Everything Phase 1+: testing engine, adapters, bias metrics, compliance
  mapper, dashboard, reports, SDK — all stub files, no logic.
- No GitHub remote configured yet (local repo only). Build plan calls for
  pushing to GitHub — do this next session or when a remote is available.
- Streamlit / fairlearn / shap / lime / scikit-learn / reportlab / garak / ollama
  not installed yet (deferred — see decision).

**Decisions made this session**
1. **Phased dependency install** (confirmed with Rhishikumar). Only Phase 0 libs
   installed now. `garak` and `alibi` are dependency-heavy and conflict-prone;
   installing the full stack risked eating the 5-day Phase 0 window (Risk #3).
   Each heavier group gets added at the start of the phase that first uses it.
2. **`__init__.py` in every subpackage** under `governance/` and in `tests/`
   (build plan only shows two explicitly, but Python needs them for imports).
3. **Endpoints live in `governance/registry/`**, not inline in `main.py` — this
   is exactly the module the build plan already defines, so no over-engineering,
   and Phase 1 test-run endpoints will follow the same pattern.
4. **Startup is synchronous** — `init_db()` is called at module load, no async
   lifespan handler. Route handlers are plain `def` (FastAPI runs them in a
   threadpool). Keeps to the "no async" rule.
5. **Branch named `main`** (renamed from git's default `master`).
6. **`risk_tier` is the only enum**; `model_type`, `sector`, `status`,
   `regulation`, `module` are strings with allowed values documented inline —
   matches how the build plan marks them.
7. String UUID primary keys generated app-side (`str(uuid.uuid4())`).

**Known minor issues (non-blocking)**
- `fastapi.testclient` emits a StarletteDeprecationWarning about httpx. Tests
  still pass. Revisit when building the Phase 1 test suite.

---

## Next Step — Phase 1, Week 1 (Model Adapter Layer)

Per BUILD_PLAN.md "Phase 1 — Testing Engine", Days 6–10:

1. Add Phase 1 dependencies: `scikit-learn`, `pandas`, `numpy`, `scipy`,
   `fairlearn`. (Hold `alibi` until actually needed.)
2. Build the model adapter layer in `governance/testing/adapters.py`:
   - `SklearnAdapter` — in-memory sklearn model
   - `PickleAdapter` — serialised model file on disk
   - `APIAdapter` — REST endpoint returning predictions
   - Common interface: `.predict(X)` / `.predict_proba(X)`.
3. Write `tests/test_bias.py` fixtures that load the UCI Adult Income dataset
   (`sklearn.datasets.fetch_openml`).
4. Also worth doing early: set up a GitHub remote and push (build plan Core
   Principle — "save to GitHub after every meaningful change").

Session start next time:
"Read docs/BUILD_PLAN.md and docs/PROGRESS.md then continue where we left off."

---

## Phase Completion Checklist

### Phase 0 — Foundation
- [x] pyenv + Python 3.11 installed
- [x] Poetry installed
- [x] Project folder and structure created
- [x] All dependencies installed via Poetry *(Phase 0 subset — phased plan)*
- [x] SQLite database with 4 tables created
- [x] GET /health endpoint works
- [x] GET /api/v1/systems endpoint works
- [x] POST /api/v1/systems endpoint works
- [x] Git repository created with first commit
- [ ] Pushed to GitHub *(no remote yet)*

### Phase 1 — Testing Engine
- [ ] Model adapter layer built (Sklearn, Pickle, API)
- [ ] BiasTestSuite with 5 metrics built
- [ ] Validated against UCI Adult Income (within 5% of published)
- [ ] Validated against COMPAS dataset (within 5% of ProPublica)
- [ ] All metrics have pytest tests
- [ ] Results save to SQLite

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
