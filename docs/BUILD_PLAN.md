# AI Governance Platform — Master Build Plan

## What We Are Building
An AI regulatory compliance testing platform. Enterprises connect their AI models to it, and it runs technical tests to detect bias, check explainability, and map results to EU AI Act and GDPR obligations. It produces compliance dashboards, audit-ready reports, and remediation guidance.

---

## Hardware & Budget
- MacBook Air M4 (local development)
- Google Colab Free Tier (experiments only)
- Zero budget — 100% free and open source stack

---

## Core Principles (Never Break These)
1. Monolith first — single FastAPI app, not microservices
2. SQLite not PostgreSQL — no database server needed for prototype
3. Streamlit not React — dashboard built fast, migrated later
4. Testing engine is the only thing that matters — 60% of effort goes here
5. Validate against real published datasets before moving to next phase

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (via pyenv) |
| Dependency management | Poetry |
| API framework | FastAPI |
| Database | SQLite via SQLAlchemy |
| Dashboard | Streamlit |
| Fairness testing | fairlearn, alibi |
| Explainability | shap, lime |
| ML | scikit-learn, pandas, numpy, scipy |
| LLM testing | Ollama (local) + Garak |
| PDF reports | ReportLab |
| Version control | Git + GitHub |
| Testing | pytest |

---

## Project Structure

```
ai-governance/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── docs/
│   ├── BUILD_PLAN.md          ← this file
│   └── PROGRESS.md            ← updated after every session
│
├── governance/                ← main Python package
│   ├── __init__.py
│   ├── main.py                ← FastAPI entry point
│   ├── config.py
│   ├── db/
│   │   ├── database.py        ← SQLite connection
│   │   └── models.py          ← ORM table definitions
│   ├── registry/
│   │   ├── router.py
│   │   ├── service.py
│   │   └── schemas.py
│   ├── testing/
│   │   ├── engine.py
│   │   ├── adapters.py        ← universal model adapter
│   │   ├── bias.py            ← bias & fairness tests
│   │   ├── explainability.py
│   │   └── results.py
│   ├── compliance/
│   │   ├── mapper.py          ← maps results to regulations
│   │   ├── scorer.py
│   │   └── rules/
│   │       ├── eu_ai_act.json
│   │       └── gdpr.json
│   └── reporting/
│       └── generator.py
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_Registry.py
│       ├── 2_Run_Tests.py
│       ├── 3_Results.py
│       └── 4_Reports.py
│
├── sdk/
│   └── aigovernance/
│       ├── __init__.py
│       ├── client.py
│       └── runner.py
│
├── tests/
│   ├── conftest.py
│   ├── test_bias.py
│   ├── test_compliance.py
│   └── fixtures/
│
└── notebooks/
```

---

## Database Tables (SQLite via SQLAlchemy)

### Table 1: ai_systems
Stores every registered AI system.
- id (string, primary key, UUID)
- name (string, required)
- description (string)
- model_type (string) — classification / regression / llm / ranking
- risk_tier (enum) — unacceptable / high / limited / minimal
- sector (string) — finance / healthcare / hr / insurance / other
- owner (string, email)
- created_at (datetime)

### Table 2: test_runs
One row per test execution.
- id (string, primary key, UUID)
- system_id (foreign key → ai_systems)
- status (string) — pending / running / complete / failed
- config (JSON) — test configuration used
- started_at (datetime)
- completed_at (datetime)

### Table 3: test_results
One row per individual metric result.
- id (string, primary key, UUID)
- run_id (foreign key → test_runs)
- module (string) — bias / explainability / robustness
- metric_name (string) — demographic_parity_difference / equalized_odds etc
- metric_value (float)
- threshold (float)
- status (string) — pass / warn / fail
- detail (JSON) — per-group breakdown, feature importances

### Table 4: compliance_scores
One row per regulatory obligation assessed.
- id (string, primary key, UUID)
- run_id (foreign key → test_runs)
- regulation (string) — eu_ai_act / gdpr
- article (string) — Article 9 / Article 10 etc
- status (string) — compliant / partial / non_compliant
- evidence (JSON) — which test results drove this score

---

## API Endpoints

### Phase 0 (Foundation)
- GET /health → {"status": "ok"}
- GET /api/v1/systems → list of registered systems
- POST /api/v1/systems → register a new system

### Phase 1 (Testing Engine)
- POST /api/v1/test-runs → trigger a new test run
- GET /api/v1/test-runs/{id} → get test run status
- GET /api/v1/test-runs/{id}/results → get test results

### Phase 2 (Compliance)
- GET /api/v1/test-runs/{id}/compliance → get compliance scores
- GET /api/v1/systems/{id}/compliance-history → history of compliance

### Phase 3 (Reports)
- POST /api/v1/reports → generate a report
- GET /api/v1/reports/{id} → download a report

---

## Phase 0 — Foundation (Days 1–5)
**Goal:** Running FastAPI app with SQLite, three working endpoints, everything committed to GitHub.

### Steps
1. Install pyenv → Python 3.11
2. Install Poetry
3. Create project folder and structure
4. Set up pyproject.toml with all dependencies
5. Create SQLAlchemy models (4 tables above)
6. Create FastAPI app with 3 endpoints
7. Confirm all endpoints work
8. Git init and first commit

### Exit Criteria — Do NOT move to Phase 1 until:
- [ ] GET /health returns {"status": "ok"}
- [ ] GET /api/v1/systems returns []
- [ ] POST /api/v1/systems creates a record in SQLite
- [ ] Git repository created with first commit

---

## Phase 1 — Testing Engine: Bias & Fairness (Days 6–25)
**Goal:** Correct bias tests validated against published research benchmarks.

### What We Are Building
A module that accepts any AI model, runs it against test data, and measures 5 fairness metrics.

### The 5 Metrics
1. Demographic Parity Difference — gap in positive prediction rate between groups (fail if > 10%)
2. Equalized Odds Difference — gap in true positive AND false positive rates (fail if > 10%)
3. Equal Opportunity Difference — gap in true positive rate (fail if > 10%)
4. Predictive Parity Difference — gap in precision between groups (fail if > 10%)
5. Individual Fairness Score — similar people get similar predictions (fail if < 0.80)

### Model Adapter Layer
Universal translator so testing engine works with any model type:
- SklearnAdapter — in-memory sklearn model
- PickleAdapter — serialised model file on disk
- APIAdapter — REST endpoint that returns predictions

### Validation Datasets (MUST use these — they have published benchmarks)
1. UCI Adult Income — gender/race bias, demographic parity diff ~0.19 for naive model
2. COMPAS Recidivism (ProPublica) — racial bias, false positive rate gap ~0.20 Black vs White
3. German Credit (UCI) — age/gender bias, demographic parity diff ~0.12

### Week Structure
- Week 1 (Days 6-10): Build model adapter layer
- Week 2 (Days 11-17): Build BiasTestSuite with all 5 metrics using fairlearn
- Week 3 (Days 18-25): Validate numbers against published benchmarks — DO NOT SKIP THIS

### Exit Criteria — Do NOT move to Phase 2 until:
- [ ] BiasTestSuite.run() produces numbers within 5% of published Adult Income benchmarks
- [ ] COMPAS false positive rate gap matches ProPublica within 5%
- [ ] All 5 metrics have pytest tests with hardcoded expected values
- [ ] Results save correctly to SQLite
- [ ] POST /api/v1/test-runs triggers a test and saves results

---

## Phase 2 — Compliance Mapper (Days 26–42)
**Goal:** Map test results to EU AI Act articles and GDPR obligations using JSON rules files.

### What We Are Building
An engine that reads test results (numbers) and maps them to specific legal obligations (words).
Rules stored as JSON files — not hardcoded — so regulations can be updated without code changes.

### Regulations to Implement
EU AI Act articles: 9 (risk management), 10 (data governance), 13 (transparency), 14 (human oversight), 15 (accuracy/robustness)
GDPR: Article 22 (automated decisions), Article 25 (privacy by design)

### Compliance Status Logic
- compliant: all relevant metrics pass threshold
- partial: some metrics warn but none fail, or some tests not run
- non_compliant: any metric fails threshold
- not_applicable: risk tier not covered by this obligation

### Compliance Score Formula
Score = (compliant × 1.0 + partial × 0.5 + non_compliant × 0.0) / total applicable obligations

### Exit Criteria — Do NOT move to Phase 3 until:
- [ ] Model with demographic_parity_difference = 0.20 → Non-Compliant EU AI Act Article 9
- [ ] Model with demographic_parity_difference = 0.05 → Compliant EU AI Act Article 9
- [ ] Adding a new regulation = creating a new JSON file, zero Python changes
- [ ] Scores persist to SQLite ComplianceScore table

---

## Phase 3 — Streamlit Dashboard (Days 43–58)
**Goal:** Working UI a non-technical CRO can use. Streamlit reads SQLite directly (no API calls).

### 4 Pages
1. Registry — view all registered systems, form to register new ones
2. Run Tests — select system, upload model (.pkl) and test data (.csv), run tests
3. Results — RAG compliance dashboard per regulation, expandable obligation detail
4. Reports — download PDF report for any test run

### Important Constraints
- Streamlit reads SQLite directly in prototype (not via API — skip that layer)
- Use @st.cache_data for all database reads
- Cap SHAP computation at 1,000 rows with a visible note in UI
- Do NOT fix Streamlit performance issues for prototype — document and move on

### Exit Criteria
- [ ] Non-technical person navigates all 4 pages without help
- [ ] Test results persist after closing and reopening browser
- [ ] RAG dashboard shows correct colours for each compliance status

---

## Phase 4 — PDF Reports & Explainability (Days 59–72)
**Goal:** Professional board-ready PDF + SHAP feature importance.

### PDF Report Contents
- System name and test date
- Overall compliance status per regulation
- Bias metrics table (metric name, value, threshold, pass/fail)
- Per-obligation compliance breakdown
- Remediation action list (ordered by priority)

### SHAP Implementation
- Use shap.Explainer — works with any sklearn-compatible model
- Sample 1,000 rows maximum — hard limit in code
- Output: feature importance dict + top feature name
- Feed into EU AI Act Article 13 compliance score

### Exit Criteria
- [ ] PDF generates and is readable by a non-technical person
- [ ] SHAP runs in under 30 seconds on 10,000-row dataset
- [ ] Download button works in Streamlit dashboard

---

## Phase 5 — SDK & Demo (Days 73–90)
**Goal:** pip-installable SDK + 5-minute demo using Adult Income dataset.

### LocalRunner SDK
Single class that runs everything locally with no server:
```python
from aigovernance import LocalRunner
runner = LocalRunner(system_name="my-model", risk_tier="high")
result = runner.run(model=model, X_test=X_test, y_test=y_test,
                    protected_attributes=["gender"])
result.print_summary()
result.export_report("report.pdf")
```

### Demo Script
Uses UCI Adult Income dataset (fetch_openml — free, automatic download).
Trains a RandomForestClassifier without fairness constraints.
Shows real bias detected, mapped to real regulations, PDF generated.
Total runtime: under 5 minutes.

### Important Note
Do NOT publish to PyPI for first demo. Use: pip install -e .
This allows instant fixes without publishing delay.

### Exit Criteria
- [ ] demo.py runs end-to-end in under 5 minutes on fresh machine
- [ ] Shows EU AI Act compliance status for each article
- [ ] PDF report generates and looks professional
- [ ] Fully works offline (no internet required)

---

## Top 10 Risks — Read Before Every Phase

1. Bias metric implementation wrong → validate against published benchmarks before Phase 2
2. Compliance mapping legally wrong → read primary EU AI Act text for every article referenced
3. Phase 0 takes 3 weeks → hard 5-day deadline, working health endpoint beats perfect structure
4. SQLite concurrency issues → enable WAL mode, acceptable for prototype
5. SHAP timeouts → always sample 1,000 rows, hard cap in code
6. Streamlit session_state bugs → clear state at start of every test run
7. Demo breaks with real customer model → test with 5 different sklearn models before demo
8. Colab loses work → save to GitHub after every meaningful change
9. PDF looks amateur → spend 1 full day on styling, show to non-technical person first
10. No real test data → use Adult Income and COMPAS from Day 1

---

## Progress Tracking
See PROGRESS.md for current status.
Update PROGRESS.md at the end of every working session.
