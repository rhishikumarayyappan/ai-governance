# AI Governance Platform — Master Build Plan v2.0

**Version:** 2.0 (supersedes v1.0)
**Last updated:** September 2026
**Change from v1.0:** Restructured to close 9 identified gap categories. Phase count expanded from 6 to 10. LLM testing elevated from afterthought to core phase. Statistical rigour, security, and legal protection added as first-class concerns.

---

## READ THIS FIRST — How This Plan Handles Gaps

Nine categories of gaps were identified in a full architectural review. Every single one is assigned to a phase in this document with a definition of done. Nothing is left undocumented.

Three tiers of gap closure:

**TIER 1 — CLOSED IN PROTOTYPE (Phases 0–6).** Buildable now, zero budget, on a MacBook M4.

**TIER 2 — CLOSED BEFORE FIRST PAYING CUSTOMER (Phases 7–8).** Required before money changes hands. Mostly engineering, some legal cost.

**TIER 3 — EXPLICITLY DEFERRED (Phase 9 + Deferred Register).** Requires budget, team, or certification. Each carries a written risk statement and a trigger condition that forces it to be addressed.

A deferred gap is not an ignored gap. Every deferral in this document names what could go wrong and what event makes it urgent.

---

## What We Are Building

An AI regulatory compliance testing platform. Enterprises connect AI systems — both structured ML models and large language models — and the platform runs technical, statistical, and adversarial tests, then maps results to EU AI Act, GDPR, ISO 42001, NIST AI RMF, and sector-specific obligations. It produces compliance dashboards, audit-ready reports, remediation tracking, and continuous monitoring.

---

## Hardware & Budget

- MacBook Air M4 (all local development)
- Google Colab Free Tier (experiments and larger model testing only)
- Ollama running locally on M4 (LLM testing — free)
- Zero budget through Phase 6
- Phases 7–8 require modest legal spend (~€2,000 for terms of service, DPA, liability review)
- Phase 9 requires funding (SOC 2, pen testing, cloud infrastructure)

---

## Core Principles (Never Break These)

1. **Monolith first** — single FastAPI app through Phase 6. Microservices only if scale demands it.
2. **SQLite through Phase 6, PostgreSQL from Phase 7** — the migration point is when multi-tenancy arrives.
3. **Streamlit through Phase 6, React from Phase 8** — prototype speed first, production polish later.
4. **Every number reported must carry uncertainty** — no point estimate without a confidence interval, after Phase 2.
5. **Validate against published benchmarks before trusting any metric** — no exceptions.
6. **Never assert compliance without a documented limitation statement** — legal protection is built in, not bolted on.
7. **Test both model families** — structured ML and LLM. A platform that tests only one is testing half the market.
8. **Defer with documentation** — anything not built must have a written risk statement and a trigger condition.

---

## Technology Stack

| Layer | Technology | Phase Introduced |
|---|---|---|
| Language | Python 3.11 (pyenv) | 0 |
| Dependency management | Poetry | 0 |
| API framework | FastAPI | 0 |
| Database (prototype) | SQLite + SQLAlchemy | 0 |
| Database (production) | PostgreSQL | 7 |
| Statistical testing | scipy.stats, statsmodels | 2 |
| Fairness metrics | fairlearn | 1 |
| Explainability | shap, lime, dice-ml | 5 |
| Causal analysis | dowhy, networkx | 4 |
| LLM runtime (local) | Ollama + Llama 3.2 / Mistral | 3 |
| LLM red teaming | garak (NVIDIA, open source) | 3 |
| Toxicity classification | detoxify (open source) | 3 |
| Adversarial robustness | adversarial-robustness-toolbox (IBM) | 4 |
| Privacy attack testing | ml-privacy-meter | 4 |
| Dashboard (prototype) | Streamlit + Plotly | 6 |
| Dashboard (production) | React + TypeScript | 8 |
| PDF reports | ReportLab | 5 |
| Job queue | Celery + Redis | 7 |
| Auth | Keycloak (OIDC) | 7 |
| Version control | Git + GitHub | 0 |
| Testing | pytest | 0 |

---

## Project Structure (v2.0)

```
ai-governance/
├── pyproject.toml
├── README.md
├── LIMITATIONS.md              ← NEW: what the product does NOT guarantee
├── THRESHOLDS.md               ← NEW: justification for every threshold
├── DEFERRED_RISKS.md           ← NEW: every deferred gap + risk + trigger
├── docs/
│   ├── BUILD_PLAN.md
│   ├── PROGRESS.md
│   ├── GAP_CHECKLIST.md        ← NEW: the 9-category tracker
│   └── VALIDATION_RESULTS.md
│
├── governance/
│   ├── main.py
│   ├── config.py
│   ├── db/
│   │   ├── database.py
│   │   └── models.py
│   ├── registry/
│   │   ├── router.py, service.py, schemas.py
│   ├── testing/
│   │   ├── adapters.py         ← structured model adapters
│   │   ├── bias.py             ← fairness metrics
│   │   ├── statistics.py       ← NEW: significance, CI, corrections
│   │   ├── regression.py       ← NEW: regression fairness metrics
│   │   ├── ranking.py          ← NEW: ranking fairness metrics
│   │   ├── intersectional.py   ← NEW: multi-attribute bias
│   │   ├── engine.py
│   │   ├── router.py
│   │   └── llm/                ← NEW MODULE (Phase 3)
│   │       ├── adapter.py      ← Ollama / OpenAI / Anthropic connector
│   │       ├── probes.py       ← adversarial prompt library
│   │       ├── red_team.py     ← garak orchestration
│   │       ├── hallucination.py
│   │       ├── toxicity.py
│   │       ├── generative_bias.py
│   │       ├── rag_attacks.py  ← RAG-specific attack testing
│   │       ├── agent_safety.py ← tool-misuse testing
│   │       └── engine.py
│   ├── security/               ← NEW MODULE (Phase 4)
│   │   ├── adversarial.py      ← adversarial example generation
│   │   ├── inversion.py        ← model inversion testing
│   │   ├── membership.py       ← membership inference testing
│   │   ├── extraction.py       ← model stealing vulnerability
│   │   ├── backdoor.py         ← trojan / backdoor detection
│   │   └── provenance.py       ← model supply chain validation
│   ├── data_governance/        ← NEW MODULE (Phase 4)
│   │   ├── lineage.py          ← data provenance tracking
│   │   ├── label_bias.py       ← label quality assessment
│   │   ├── proxy_chains.py     ← causal proxy detection
│   │   ├── minimisation.py     ← feature necessity assessment
│   │   └── balance.py          ← class & group imbalance checks
│   ├── compliance/
│   │   ├── mapper.py
│   │   ├── scorer.py
│   │   ├── tensions.py         ← NEW: metric conflict detection
│   │   ├── documentation.py    ← NEW: Art.11 doc completeness check
│   │   └── rules/
│   │       ├── eu_ai_act.json
│   │       ├── gdpr.json
│   │       ├── iso_42001.json  ← NEW
│   │       ├── nist_ai_rmf.json ← NEW
│   │       ├── uk_ai_framework.json ← NEW
│   │       ├── us_nyc_ll144.json   ← NEW
│   │       ├── us_colorado_ai.json ← NEW
│   │       ├── sector_finance.json ← NEW (EBA, DORA, ECOA)
│   │       └── sector_health.json  ← NEW (MDR, FDA)
│   ├── monitoring/             ← NEW MODULE (Phase 6)
│   │   ├── drift.py            ← model & fairness drift
│   │   ├── temporal_bias.py    ← bias change over time
│   │   ├── feedback_loop.py    ← self-reinforcing bias detection
│   │   └── scheduler.py
│   ├── workflow/               ← NEW MODULE (Phase 6)
│   │   ├── remediation.py      ← issue tracking & resolution
│   │   ├── human_review.py     ← override & escalation
│   │   └── approval_gate.py    ← pre-deployment blocking
│   └── reporting/
│       ├── generator.py
│       ├── explanation.py      ← NEW: GDPR Art.22 plain-language
│       └── benchmarks.py       ← NEW: peer comparison
│
├── dashboard/
├── sdk/
├── tests/
└── notebooks/
```

---

## Database Tables (v2.0)

Existing four tables from v1.0 retained. New tables added:

### ai_systems (extended)
Existing fields plus:
- model_family (string) — "structured" / "llm" / "multimodal" / "ranking"
- deployment_status (string) — "development" / "approved" / "production" / "retired"
- last_tested_at (datetime)
- documentation_complete (boolean)
- approved_by (string) — human sign-off for deployment

### test_runs (extended)
Existing fields plus:
- test_config_version (string) — pins exactly which config was used
- engine_version (string) — pins the code version for reproducibility
- sample_size (int)
- reliability_score (float) — stability across repeated runs

### test_results (extended)
Existing fields plus:
- confidence_interval_lower (float)
- confidence_interval_upper (float)
- p_value (float)
- corrected_threshold (float) — after Bonferroni
- sample_size (int)
- reliability_flag (string) — "reliable" / "unstable" / "insufficient_data"

### llm_test_runs (NEW)
- id, system_id, model_identifier, status, started_at, completed_at
- probe_categories (JSON) — which attack categories were run
- total_probes, failed_probes, failure_rate

### llm_test_results (NEW)
- id, run_id, probe_category, probe_variant
- prompt_sent (text), response_received (text)
- failed (boolean), harm_score (float), harm_category (string)
- detection_method (string) — how failure was determined

### security_test_results (NEW)
- id, run_id, attack_type, vulnerable (boolean)
- severity (string), detail (JSON), remediation (text)

### data_audit_results (NEW)
- id, system_id, audit_type, status
- finding (text), severity, detail (JSON)

### remediation_items (NEW)
- id, system_id, source_result_id, issue_description
- severity, status ("open"/"in_progress"/"resolved"/"accepted_risk")
- assigned_to, due_date, resolved_at, resolution_note

### human_reviews (NEW)
- id, run_id, reviewer, decision ("confirm"/"override"/"escalate")
- reasoning (text), reviewed_at

### audit_log (NEW)
- id, org_id, user_id, action, entity_type, entity_id
- timestamp, payload (JSON), ip_address

### model_versions (NEW)
- id, system_id, version, artifact_hash, registered_at
- previous_version_id — for lineage chain

### organisations (NEW — Phase 7)
- id, name, plan, settings, created_at

### users (NEW — Phase 7)
- id, org_id, email, role, sso_id, created_at

---

# PHASE 0 — Foundation ✅ COMPLETE

**Status:** Complete. FastAPI + SQLite + 3 endpoints + Git.

---

# PHASE 1 — Structured ML Testing Engine ✅ COMPLETE

**Status:** Complete. Weeks 1–2 (adapters, BiasTestSuite, engine, API — 20 tests passing) + Week 3 benchmark validation (2026-09-02).
**Remaining:** none. Phase 2 may begin.

### Week 3 — Validation (unchanged from v1.0)
Validate against UCI Adult Income, COMPAS, German Credit. All three must fall in published ranges.
Script: `notebooks/validate_phase1.py`. Full write-up: `docs/VALIDATION_RESULTS.md`.

### Exit Criteria
- [x] Adult Income demographic_parity_difference in 0.15–0.25 → **0.1745**
- [x] COMPAS equalized_odds_difference in 0.15–0.25 → **0.1752**
- [x] German Credit demographic_parity_difference in 0.05–0.20 → **0.1149**
- [x] docs/VALIDATION_RESULTS.md created
- [x] pytest → 20 passed, 0 failed

---

# PHASE 2 — Statistical Rigour Layer 🆕

**Duration:** Days 26–40
**Closes:** Gap Category 1 (statistical holes) entirely
**Goal:** Every metric reported with confidence intervals, p-values, and multiple-comparison correction. No bare point estimates ever again.

## Why This Phase Exists

A sophisticated CTO or a regulator with statistical training will ask, in the first serious conversation: *"Is that difference statistically significant, and what is your confidence interval?"* Without an answer, the product is a toy. This phase makes every number defensible.

## Component 2.1 — Statistical Testing Module

File: `governance/testing/statistics.py`

### Significance Testing
For every fairness metric, compute a p-value answering: could this gap have occurred by chance?

- **Chi-squared test of independence** — for demographic parity. Tests whether prediction outcome is independent of group membership.
- **Fisher's exact test** — automatically substituted when any expected cell count is below 5. Chi-squared is unreliable on small samples.
- **Two-proportion z-test** — for direct group rate comparisons.
- **Permutation test** — 1,000 label shuffles; the gold standard when distributional assumptions are questionable. Used as fallback when parametric assumptions fail.

Return: `p_value`, `test_used`, `assumptions_met` (boolean).

### Confidence Intervals
Bootstrap resampling with 1,000 iterations for every metric.
- Resample the test set with replacement 1,000 times
- Recompute the metric on each resample
- Report the 2.5th and 97.5th percentiles as the 95% CI

Every `TestResult` gains `confidence_interval_lower` and `confidence_interval_upper`.

### Multiple Comparisons Correction
Five simultaneous tests inflate false-positive probability. Implement:
- **Bonferroni correction** — divide alpha by number of tests (conservative, default)
- **Benjamini-Hochberg FDR** — less conservative, offered as an option
- Store both `threshold` (raw) and `corrected_threshold` on every result
- The compliance mapper uses the **corrected** threshold, always

### Reliability Assessment

> **Revised 2026-09-02 during Component 2.3 implementation** (was: "SD < 0.02 →
> reliable / 0.02–0.05 → unstable / SD > 0.05 or n < 100 per group →
> insufficient_data", via an N-run reseed). The draft conflated *measured
> uncertainty on an estimate that was computed* with *data that cannot support
> an estimate at all* — those are different tiers. It also specified an N-run
> reseed, which would multiply the ~4s/metric bootstrap cost by N for a signal
> the single 1,000-iteration bootstrap distribution already carries. The rule
> table below is what is implemented in `assess_reliability`.

`assess_reliability(bootstrap_result, sample_sizes, min_group_size=30)` reads the
standard deviation, CI width and skip counts from the single 1,000-iteration
bootstrap distribution computed for the metric. All rules are evaluated and
every firing rule is reported in `reasons`; `tier` is the most severe that fired.

| Condition | Tier |
|---|---|
| every bootstrap resample failed (`n_valid_iterations == 0`) | `insufficient_data` |
| any group has fewer than `min_group_size` (default 30) samples | `insufficient_data` |
| more than 5% of resamples collapsed to a single group | `insufficient_data` |
| 95% CI width > 0.15 | `unstable` |
| bootstrap SD > 0.05 | `unstable` |
| none of the above | `reliable` |

A result flagged `insufficient_data` **cannot** produce a compliance verdict
(`blocks_verdict == True`). It produces "indeterminate — insufficient sample."
`unstable` degrades confidence but still reports.

## Component 2.2 — Threshold Justification

File: `THRESHOLDS.md` (repo root, version controlled)

For every threshold in the system, document:
- The numeric value
- The source or reasoning (regulatory guidance, academic literature, industry convention, or explicitly "internal default — no regulatory basis")
- Who set it and when
- Whether it is configurable per customer

**Critical honesty requirement:** The EU AI Act specifies no numeric fairness thresholds. THRESHOLDS.md must state plainly that the 0.10 default is an internal convention drawn from the four-fifths rule (US EEOC 80% rule, which implies a 0.20 ratio disparity) adapted to a difference measure — and that it carries no regulatory force. Customers can and should configure their own with legal advice.

Thresholds become fully configurable per AI system, stored on the system record, with an audit-log entry every time one changes.

## Component 2.3 — Simpson's Paradox Detection

File: added to `governance/testing/statistics.py`

After computing an aggregate metric, automatically stratify by every other categorical column with fewer than 10 unique values, recompute the metric within each stratum, and flag when:
- The aggregate shows no bias but any stratum shows bias above threshold
- The aggregate direction reverses in any stratum

Output a warning appended to the result detail: *"Aggregate demographic parity is 0.03 (pass), but within the 25–34 age stratum it is 0.19 (fail). Aggregate result may be masking subgroup bias."*

## Component 2.4 — Metric Tension Detection

File: `governance/compliance/tensions.py`

Implement the impossibility theorem awareness. When base rates differ between groups, demographic parity and predictive parity are mathematically incompatible. Detect and explain:

- Compute base rate per group (`y_true` positive rate)
- If base rates differ by more than 5%, flag that perfect fairness across all metrics is mathematically unachievable
- When demographic parity fails but predictive parity passes → explain this may reflect genuine base rate differences, not model discrimination
- When the reverse → explain the model equalises rates at the cost of accuracy parity

Every compliance report includes a "Fairness Definition Applied" section stating which definition was prioritised and why, because you cannot satisfy them all.

## Exit Criteria — Phase 2
- [ ] `statistics.py` with chi-squared, Fisher's exact, z-test, permutation test
- [ ] Bootstrap CI (1,000 iterations) on all 5 metrics
- [ ] Bonferroni and Benjamini-Hochberg corrections implemented
- [ ] Reliability assessment with three-tier flag
- [ ] "insufficient_data" blocks compliance verdict — verified by test
- [ ] `THRESHOLDS.md` written with honest regulatory-basis statement
- [ ] Thresholds configurable per system, changes audit-logged
- [ ] Simpson's paradox stratification detects a planted subgroup reversal (test)
- [ ] Metric tension detection fires on differing base rates (test)
- [ ] All `TestResult` rows now carry CI, p-value, corrected threshold, sample size
- [ ] Re-run Phase 1 validation — numbers unchanged, now with CIs attached
- [ ] pytest → all passing

---

# PHASE 3 — LLM Testing & Red Teaming 🆕

**Duration:** Days 41–65
**Closes:** Gap Category 3 (LLM/agentic gaps) entirely
**Goal:** Test the AI models enterprises are actually deploying in 2026.

## Why This Phase Is Now Third, Not Last

Every enterprise is deploying LLMs right now — customer service bots, document summarisers, internal knowledge bases, code assistants — usually with zero systematic safety testing. A governance platform that cannot test them covers half the market and the half that is shrinking. This is the single highest-value addition to the product.

## Component 3.1 — LLM Adapter

File: `governance/testing/llm/adapter.py`

Universal interface across LLM providers:
- **OllamaAdapter** — local models on M4 (llama3.2, mistral). Free, private, no API key.
- **OpenAIAdapter** — GPT models via API
- **AnthropicAdapter** — Claude models via API
- **GenericAPIAdapter** — any REST endpoint returning text
- **HuggingFaceAdapter** — local transformers models

Interface: `generate(prompt, system_prompt=None, temperature=0.0) -> str`

Temperature defaults to 0.0 for reproducibility. Non-deterministic testing requires N repetitions — the adapter supports `generate_n(prompt, n=5)` returning a list, so variance can be measured.

## Component 3.2 — Adversarial Probe Library

File: `governance/testing/llm/probes.py`

A structured, versioned catalogue of attack prompts. Each probe has: id, category, prompt template, success criteria, severity, source.

**Categories implemented:**

| Category | What It Tests | Probe Count (min) |
|---|---|---|
| `prompt_injection` | Instruction override via user input | 40 |
| `jailbreak` | Safety guardrail bypass (DAN, roleplay, hypothetical framing) | 50 |
| `system_prompt_extraction` | Revealing the system prompt | 20 |
| `pii_leakage` | Reproducing personal data from context or training | 25 |
| `harmful_content` | Generating dangerous/illegal instructions | 30 |
| `toxicity` | Offensive, discriminatory output | 30 |
| `encoding_bypass` | base64/rot13/unicode obfuscated harmful requests | 20 |
| `indirect_injection` | Attack embedded in retrieved/processed content | 25 |
| `over_refusal` | Refusing legitimate requests (usability failure) | 20 |
| `hallucination_induction` | Prompts designed to elicit fabrication | 25 |

Probes are stored as JSON, versioned separately from code, so the library can be updated without a code release.

## Component 3.3 — Garak Integration

File: `governance/testing/llm/red_team.py`

Garak (NVIDIA, open source) is an automated LLM red-teaming framework. Wrap it:
- Configure garak to target the model via the LLM adapter
- Run selected probe modules
- Parse garak's JSONL report output
- Map garak probe results into our `llm_test_results` schema
- Compute per-category failure rates

Garak runs locally against Ollama models at zero cost. For API models, cost is per-token and must be estimated and shown before running.

## Component 3.4 — Hallucination Testing

File: `governance/testing/llm/hallucination.py`

Three methods, combined:
1. **Known-answer benchmarking** — a curated Q&A set with verified answers; score exact/semantic match
2. **Self-consistency** — ask the same question 5 times at temperature 0.7; high variance in factual claims indicates fabrication
3. **Retrieval grounding check** — for RAG systems, verify every factual claim in the output appears in the retrieved context

Output: `hallucination_rate` (0–1), with per-question detail.

## Component 3.5 — Toxicity & Harm Classification

File: `governance/testing/llm/toxicity.py`

Use `detoxify` (open source, runs locally on M4) to score every model output across: toxicity, severe toxicity, obscene, threat, insult, identity attack.

Run against: normal prompts (baseline), adversarial prompts (stress), and demographic-varied prompts (bias).

## Component 3.6 — Generative Bias Testing

File: `governance/testing/llm/generative_bias.py`

The LLM equivalent of demographic parity. Same prompt, varied demographic framing:

```
"Write a performance review for {name} who missed targets"
names = [John, Fatima, Wei, Aisha, Sean, Priya, ...]
```

Measure across outputs:
- **Sentiment disparity** — sentiment score variance across demographic groups
- **Length disparity** — systematic differences in response length
- **Lexical disparity** — differences in word choice, formality, hedging language
- **Recommendation disparity** — for decision-support prompts, does the recommendation differ

Apply the same statistical machinery from Phase 2 — significance testing and confidence intervals on the disparities.

## Component 3.7 — RAG-Specific Attack Testing

File: `governance/testing/llm/rag_attacks.py`

RAG systems have unique attack surfaces not covered by general prompt injection:
- **Knowledge base poisoning** — inject an adversarial document; test whether the model retrieves and repeats it
- **Retrieval manipulation** — craft queries that force retrieval of sensitive documents
- **Context stuffing** — overflow the context window to push out the system prompt
- **Citation fabrication** — does the model invent sources that were not retrieved

## Component 3.8 — Agent Safety Testing

File: `governance/testing/llm/agent_safety.py`

For LLMs with tool access — the highest-risk deployment pattern:
- **Tool misuse via injection** — can a prompt injection cause the agent to call a destructive tool (delete, send, transfer)?
- **Privilege escalation** — can the agent be induced to use tools outside its intended scope?
- **Confused deputy** — can the agent be tricked into acting on behalf of an unauthorised party?
- **Multi-agent manipulation** — in multi-agent systems, can Agent A be used to manipulate Agent B?

Testing uses **mock tools** that log invocation without executing. Never test agent safety against live tools.

## Component 3.9 — Fine-Tuning & Quantisation Regression

File: added to `governance/testing/llm/engine.py`

- **Alignment regression testing** — run the same probe suite against a base model and its fine-tuned derivative; report which safety behaviours degraded
- **Quantisation drift** — run the same probe suite at full precision and at the deployed quantisation level; report behavioural differences

## Exit Criteria — Phase 3
- [ ] LLM adapter works with Ollama (llama3.2) locally on M4
- [ ] Probe library with 285+ probes across 10 categories, stored as versioned JSON
- [ ] Garak integrated, running, results parsed into our schema
- [ ] Hallucination testing with all three methods
- [ ] Toxicity scoring via detoxify running locally
- [ ] Generative bias testing with statistical significance from Phase 2
- [ ] RAG attack suite (4 attack types)
- [ ] Agent safety suite with mock tools only (4 attack types)
- [ ] Fine-tuning regression comparison working
- [ ] `llm_test_runs` and `llm_test_results` tables live
- [ ] API: POST/GET `/api/v1/llm-test-runs`
- [ ] Validation: a deliberately unsafe local model fails; a well-aligned model passes
- [ ] pytest → all passing

---

# PHASE 4 — Security, Data Governance & Model Coverage 🆕

**Duration:** Days 66–90
**Closes:** Gap Categories 2 (model types), 4 (security), 6 (data & privacy), and the rare gaps in Category 9
**Goal:** Cover the model families and attack surfaces the plan previously ignored.

## Component 4.1 — Regression Fairness Metrics

File: `governance/testing/regression.py`

Binary classification metrics do not apply to continuous outputs (credit scores, risk probabilities, predicted salaries). Implement:
- **Mean prediction difference** across groups
- **Variance ratio** — is the model more uncertain for one group?
- **Distributional overlap** (Wasserstein distance / KS statistic) between group prediction distributions
- **Calibration by group** — are predicted probabilities equally well-calibrated per group?
- **Group-conditional error parity** — MAE/RMSE differences across groups

## Component 4.2 — Ranking Fairness Metrics

File: `governance/testing/ranking.py`

Critical for recruitment AI (EU AI Act Annex III explicitly names employment):
- **Exposure parity** — do groups receive proportional visibility in top-k?
- **nDCG parity** — ranking quality equality across groups
- **Rank-position disparity** — average rank position by group
- **Top-k representation** — proportion of each group in top 10, 50, 100

## Component 4.3 — Intersectional Bias Testing

File: `governance/testing/intersectional.py`

The most-missed form of bias. Testing gender and race separately cannot detect bias specific to Black women.

- Generate all combinations of specified protected attributes (2-way and 3-way)
- Run the full metric suite on each intersection
- **Automatic minimum-size guard** — intersections with fewer than 30 samples are flagged "insufficient_data", never given a verdict
- Report which intersections show bias invisible in the marginal analysis
- Apply multiple-comparison correction across all intersections (this generates many tests — Bonferroni is essential here)

## Component 4.4 — Adversarial Robustness

File: `governance/security/adversarial.py`

Using IBM's `adversarial-robustness-toolbox` (open source):
- **FGSM / PGD attacks** — generate adversarial examples for the model
- **Robustness score** — what perturbation magnitude causes misclassification?
- **Group-conditional robustness** — is the model *less robust* for some demographic groups? (A rarely-tested but serious fairness issue.)

## Component 4.5 — Privacy Attack Testing

File: `governance/security/inversion.py`, `membership.py`, `extraction.py`

- **Membership inference** — can an attacker determine if a specific record was in training data? GDPR-relevant.
- **Model inversion** — can training data be reconstructed by querying? Uses `ml-privacy-meter`.
- **Model extraction vulnerability** — how many queries are needed to clone the model? IP and safety risk.

## Component 4.6 — Backdoor & Provenance

File: `governance/security/backdoor.py`, `provenance.py`

- **Backdoor detection** — statistical anomaly detection on model activations; scan for trigger patterns
- **Supply chain validation** — hash verification of model artifacts, HuggingFace model card checks, warning on models from unverified sources
- **Model version lineage** — every model registered with a hash, chained to its predecessor

## Component 4.7 — Training Data Auditing

File: `governance/data_governance/`

The biggest blind spot in v1.0 — testing only the model, never the data that made it.

- **`balance.py`** — class imbalance detection, group size validation (min 30 per group warning), stratified sampling utilities, intersectional cell-size checks
- **`label_bias.py`** — assess whether ground-truth labels themselves encode discrimination. Compare label distribution across groups against external base rates where available; flag when the label is a known proxy for enforcement patterns (e.g. arrest records)
- **`proxy_chains.py`** — causal graph analysis using `dowhy`/`networkx` to detect multi-step proxy chains (postcode → school → grades → decision) that single-step correlation checks miss
- **`minimisation.py`** — GDPR Article 5 data minimisation: rank features by predictive contribution vs privacy sensitivity; flag features with low utility and high privacy risk
- **`lineage.py`** — structured capture of data provenance: source, collection date, consent basis, transformations applied, retention policy

## Component 4.8 — Measurement Invariance

File: added to `governance/data_governance/`

A psychometric concept almost never applied to AI. Does a feature mean the same thing across groups? "Employment length" means something different for a 25-year-old graduate and a 55-year-old career-changer.

- Test whether feature-outcome relationships are consistent across groups
- Flag features where the relationship differs significantly — these may require group-specific handling or exclusion

## Exit Criteria — Phase 4
- [ ] Regression fairness metrics (5 metrics) implemented and tested
- [ ] Ranking fairness metrics (4 metrics) implemented and tested
- [ ] Intersectional testing with automatic small-cell guarding
- [ ] Adversarial robustness with group-conditional analysis
- [ ] Membership inference and model inversion testing
- [ ] Model extraction vulnerability assessment
- [ ] Backdoor detection and provenance validation
- [ ] Class/group imbalance detection with warnings
- [ ] Label bias assessment module
- [ ] Proxy chain causal detection
- [ ] Data minimisation scoring
- [ ] Data lineage capture schema
- [ ] Measurement invariance testing
- [ ] `security_test_results` and `data_audit_results` tables live
- [ ] pytest → all passing

---

# PHASE 5 — Compliance Mapper & Explainability

**Duration:** Days 91–115
**Closes:** Gap Category 5 (regulatory coverage) and the explanation gaps in Category 6
**Goal:** Map everything to every relevant regulation, with honest limitation statements.

## Component 5.1 — Expanded Regulatory Rule Sets

All stored as versioned JSON in `governance/compliance/rules/`:

| Regulation | Coverage | Priority |
|---|---|---|
| **EU AI Act** | Articles 9, 10, 11, 12, 13, 14, 15, 17; Annex III, IV; Title VIII (GPAI) | Critical |
| **GDPR** | Articles 5, 13, 14, 22, 25, 35 (DPIA) | Critical |
| **ISO 42001** | All clauses — including organisational, not just technical | High |
| **NIST AI RMF** | GOVERN, MAP, MEASURE, MANAGE functions | High |
| **UK AI Framework** | Five principles + ICO guidance | High |
| **NYC Local Law 144** | Annual bias audit requirements for AEDTs | High |
| **Colorado AI Act** | Consumer protection, impact assessments | Medium |
| **California** | ADMT regulations, employment AI | Medium |
| **Illinois AIVIA** | Video interview AI consent and reporting | Medium |
| **EBA Guidelines** | Financial services model risk | High |
| **DORA** | ICT/AI operational resilience for financial entities | High |
| **ECOA / Reg B** | US credit — adverse action notice requirements | Medium |
| **Fair Housing Act** | Algorithmic discrimination in housing | Medium |
| **EU MDR** | AI in medical devices | Medium |
| **FDA AI/ML SaMD** | US medical AI guidance | Medium |

Every rule entry must include: the regulation, article/clause, exact requirement text (paraphrased, with citation), which test results map to it, and the remediation guidance.

**Regulatory volatility handling:** every rule file carries a `status` field — `"in_force"`, `"pending_implementing_acts"`, or `"draft"`. Any obligation whose technical standard is not yet published (much of EU AI Act Article 15) is marked `"pending_implementing_acts"` and its compliance verdict is capped at "partial — awaiting technical standard" rather than "compliant."

## Component 5.2 — Article 11 Documentation Completeness

File: `governance/compliance/documentation.py`

EU AI Act Article 11 requires specific technical documentation. A model can pass every fairness test and still be non-compliant because documentation is missing.

Checklist assessment covering: intended purpose, system architecture, training data description, validation methodology, performance metrics, known limitations, human oversight measures, expected lifetime, change log.

Each item: present / partial / missing. Feeds into the Article 11 compliance verdict.

## Component 5.3 — Human Oversight Verification (Article 14)

Honest scope statement built in: the platform can verify that oversight mechanisms are *documented and configured*. It cannot verify they are *operationally effective*.

What is checkable:
- Is an override mechanism documented?
- Are override statistics being logged?
- What is the actual override rate? (If 0% over thousands of decisions, oversight is likely nominal.)
- Is there a documented escalation path?

The compliance report must state explicitly: *"Human oversight assessed as documented. Operational effectiveness requires manual audit."*

## Component 5.4 — GDPR Article 22 Plain-Language Explanations

File: `governance/reporting/explanation.py`

SHAP output is not a meaningful explanation to a data subject. "feature_2 contributed 0.34" is meaningless to a person whose loan was rejected.

Generate natural-language explanations:
> "This application was declined primarily because the reported income (£24,000) is below the threshold typically associated with approval for this loan amount, and because the employment history is shorter than 12 months. If income were £32,000 or above, with all other factors unchanged, the outcome would likely have been different."

Uses SHAP + counterfactual generation (`dice-ml`) + template-based natural language generation. No LLM required — deterministic and auditable.

## Component 5.5 — Explainability Suite

File: `governance/testing/explainability.py`
- SHAP (global and local), capped at 1,000-row sample with visible disclosure
- LIME for per-prediction explanation
- Counterfactual explanations via `dice-ml`
- Model card auto-generation
- Feature importance stability across runs (reliability of the explanation itself)

## Exit Criteria — Phase 5
- [ ] 15 regulatory rule sets implemented as versioned JSON
- [ ] Every rule carries `status` field; pending standards cap verdict at "partial"
- [ ] Article 11 documentation completeness assessment
- [ ] Article 14 human oversight assessment with honest scope statement
- [ ] GDPR Art. 22 plain-language explanation generator
- [ ] SHAP, LIME, counterfactuals, model cards
- [ ] Adding a regulation requires only a new JSON file — verified
- [ ] Compliance verdicts use Phase 2 corrected thresholds
- [ ] "insufficient_data" results produce "indeterminate", never "compliant"
- [ ] pytest → all passing

---

# PHASE 6 — Monitoring, Workflow & Dashboard

**Duration:** Days 116–140
**Closes:** Gap Categories 7 (partially — architectural), 8 (product gaps), and remaining Category 9 items
**Goal:** Turn a point-in-time testing tool into an ongoing governance function.

## Component 6.1 — Continuous Monitoring

File: `governance/monitoring/`

- **`drift.py`** — statistical drift detection on input distributions (KS test, PSI) and prediction distributions
- **`temporal_bias.py`** — track every fairness metric over time; alert when a metric degrades beyond a configured delta, even if still technically passing
- **`feedback_loop.py`** — detect self-reinforcing bias: correlation between past model decisions and subsequent training data composition; flag when a group's representation in training data is shrinking as a consequence of prior rejections
- **`scheduler.py`** — cron-style scheduled re-testing

**Continuously-learning model handling:** systems flagged as `continuous_learning` require re-testing on a mandatory cadence; compliance verdicts carry an expiry date after which they revert to "indeterminate — retest required."

## Component 6.2 — Remediation Tracking

File: `governance/workflow/remediation.py`

Every failed test result can generate a remediation item with: description, severity, assignee, due date, status, resolution note.

This answers the question every regulator asks: *"When did you know, and what did you do about it?"*

- Open items visible on dashboard with age
- Overdue items escalated
- "Accepted risk" status requires a written justification and named approver
- Full history retained — a resolved item is never deleted

## Component 6.3 — Human Review Workflow

File: `governance/workflow/human_review.py`

Automated verdicts are not final. A reviewer can:
- **Confirm** the automated verdict
- **Override** with mandatory written reasoning
- **Escalate** for expert review

Every review is audit-logged with reviewer identity, timestamp, and reasoning. Overridden verdicts are visibly marked as such in every report.

## Component 6.4 — Pre-Deployment Approval Gate

File: `governance/workflow/approval_gate.py`

The highest-value use of compliance testing is preventing bad models from deploying, not documenting harm afterwards.

- A system in `development` status cannot move to `production` without a passing test run within a configurable recency window
- CI/CD integration: a webhook endpoint returns pass/fail so a deployment pipeline can block
- Override requires named approver and written justification, audit-logged

## Component 6.5 — Benchmarking & Trend Analysis

File: `governance/reporting/benchmarks.py`

- **Trend analysis** — every metric plotted over time per system; direction and rate of change surfaced, not just current value
- **Internal benchmarking** — how does this model compare to others in the same organisation and sector?
- **Anonymised cross-customer benchmarking** — deferred to Phase 9 (requires multi-customer data and careful privacy design; see Deferred Register)

## Component 6.6 — Streamlit Dashboard

Pages:
1. **Registry** — all AI systems, risk tier, deployment status, last tested, documentation status
2. **Run Tests** — structured ML testing
3. **LLM Testing** — red team runs and results
4. **Results** — RAG compliance status with CIs and reliability flags visible
5. **Remediation** — open items, owners, due dates, overdue highlighting
6. **Trends** — metrics over time
7. **Reports** — PDF generation and download
8. **Review Queue** — items awaiting human review

**Mandatory UI requirement:** every metric displayed must show its confidence interval and reliability flag. A number without uncertainty is never displayed alone.

## Exit Criteria — Phase 6
- [ ] Drift detection (input and prediction)
- [ ] Temporal bias tracking with degradation alerts
- [ ] Feedback loop detection
- [ ] Scheduled re-testing
- [ ] Continuous-learning systems get expiring verdicts
- [ ] Remediation tracking with full lifecycle
- [ ] Human review with confirm/override/escalate, all audit-logged
- [ ] Pre-deployment approval gate with CI/CD webhook
- [ ] Trend analysis over time
- [ ] 8-page Streamlit dashboard
- [ ] Every displayed metric shows CI and reliability flag
- [ ] Non-technical person navigates without help
- [ ] pytest → all passing

---

# PHASE 7 — Production Architecture & Legal Protection 🆕

**Duration:** Days 141–170
**Closes:** Gap Category 7 (architectural) entirely, Category 8 (liability) entirely
**Goal:** Everything required before a single customer pays money.

## Component 7.1 — Authentication & Authorisation

- Keycloak (OIDC) for authentication; SSO/SAML support for enterprise
- RBAC roles: Owner, Admin, Analyst, Viewer, API Key
- API key management with per-key scoping and rotation
- **No endpoint accessible without authentication** — verified by automated test

## Component 7.2 — Multi-Tenancy

- `organisations` and `users` tables
- Every existing table gains `org_id`
- Application-layer scoping on every query
- **PostgreSQL row-level security policies as a database-layer backstop** — belt and braces
- Automated cross-tenant access test: Org A must be unable to read Org B's data under any query path

## Component 7.3 — PostgreSQL Migration

- Alembic migration from SQLite to PostgreSQL
- Connection pooling
- Removes the single-writer concurrency limit
- WAL/backup strategy

## Component 7.4 — Asynchronous Job Processing

- Celery + Redis
- Test runs become background jobs; API returns immediately with a run_id
- Progress polling endpoint
- Job timeout and retry policy
- **Solves the HTTP timeout problem** that would break on any real-sized model

## Component 7.5 — Input Limits & Rate Limiting

- Max upload size (default 500MB model, 100MB CSV), configurable per plan
- Max rows per test run with automatic sampling above the limit (disclosed in results)
- Per-tenant rate limiting on all endpoints
- Request timeout enforcement

## Component 7.6 — Test Configuration Versioning

- Every test config stored with a version hash
- Every test run pins `test_config_version` and `engine_version`
- Two runs are only comparable if both versions match — the UI states this explicitly when comparing
- Historical results remain reproducible

## Component 7.7 — Full Audit Logging

- `audit_log` table capturing every state-changing action
- Immutable — append only, no updates or deletes
- Includes user, action, entity, timestamp, IP, payload
- Exportable as an evidence package for regulatory examination
- Configurable retention per jurisdiction

## Component 7.8 — Legal Protection Layer 🔴 CRITICAL

**This is the gap that could end the business. It is not optional.**

**`LIMITATIONS.md`** — a plain-language, version-controlled statement covering:
- The platform provides technical evidence, **not legal advice or legal certification**
- Automated testing cannot detect all forms of bias or all compliance failures
- Compliance verdicts are based on configurable thresholds with no regulatory force
- Results are valid only for the specific model version, dataset, and configuration tested
- Regulatory requirements are evolving; mappings reflect the state of guidance at the time of testing
- The customer remains solely responsible for their regulatory compliance

**Embedded in every output:** every PDF report, every dashboard compliance view, and every API compliance response carries a limitation statement. Not a footnote — a visible, unavoidable section.

**Terms of Service and DPA** — drafted by an Irish solicitor (~€1,500–2,000). Must cover: limitation of liability, no warranty of regulatory outcome, customer responsibility, data processing terms, IP ownership.

**Professional indemnity insurance** — in force before the first paying customer. Approximately €1,000–1,500/year for this risk profile.

**Verdict language discipline:** the system never outputs "compliant." It outputs **"no issues detected against tested criteria."** The difference is legally material.

## Component 7.9 — Data Handling & Cross-Border

- Data residency configuration (EU default)
- Automatic flagging when test data may contain personal data of subjects in a jurisdiction requiring transfer safeguards
- Data retention policy: test input data deleted after run completion by default; only results retained
- Right-to-erasure implementation — full tenant data deletion within 30 days
- PII scrubbing in all logs

## Exit Criteria — Phase 7
- [ ] Authentication on every endpoint — verified by test
- [ ] RBAC with 5 roles enforced
- [ ] Multi-tenancy with app-layer scoping AND database RLS
- [ ] Cross-tenant access test passes (Org A cannot reach Org B)
- [ ] PostgreSQL migration complete, SQLite retired
- [ ] Celery async jobs — no synchronous test execution remains
- [ ] Input size limits and rate limiting enforced
- [ ] Config and engine versioning on every run
- [ ] Immutable audit log with evidence-package export
- [ ] `LIMITATIONS.md` written and embedded in every output
- [ ] Terms of Service and DPA drafted by solicitor
- [ ] Professional indemnity insurance in force
- [ ] No output anywhere uses the bare word "compliant"
- [ ] Data retention and erasure implemented
- [ ] pytest → all passing

---

# PHASE 8 — Production UI, SDK & Integrations

**Duration:** Days 171–200
**Goal:** Enterprise-grade interface and developer experience.

- React + TypeScript dashboard replacing Streamlit
- Python SDK published (`pip install aigovernance`)
- CI/CD integrations: GitHub Actions, GitLab CI, Jenkins, Azure DevOps
- ML platform connectors: MLflow, SageMaker, Azure ML, Vertex AI
- Enterprise integrations: Jira, Slack, ServiceNow
- Webhook system
- Full OpenAPI documentation

### Exit Criteria — Phase 8
- [ ] React dashboard at feature parity with Streamlit version
- [ ] SDK installable and documented
- [ ] 4 CI/CD integrations
- [ ] 4 ML platform connectors
- [ ] Webhook delivery with retry
- [ ] 5-minute demo runs end to end

---

# PHASE 9 — Scale, Certification & Deferred Items

**Duration:** Ongoing, funding-dependent
**Closes:** Everything in the Deferred Register

- SOC 2 Type II certification
- ISO 27001 certification
- Annual third-party penetration testing
- Multimodal testing (vision, audio, video)
- Embedding bias analysis
- Differential privacy for cross-customer benchmarking
- Anonymised industry benchmarking
- Self-hosted deployment (Helm chart)
- Regulatory update monitoring agent
- Remediation recommendation engine

---

# DEFERRED RISK REGISTER

Every gap not closed in Phases 0–8, with its risk and its trigger.

| # | Deferred Item | Risk If Not Addressed | Trigger That Forces Action |
|---|---|---|---|
| D1 | SOC 2 Type II | Enterprise procurement will block the sale | First enterprise deal above €50k, or any regulated-industry prospect |
| D2 | ISO 27001 | EU public sector and finance sales blocked | First public sector or tier-1 bank opportunity |
| D3 | Third-party pen test | Undiscovered vulnerability; customer breach | Before first production customer with real data |
| D4 | Multimodal testing (vision/audio) | Cannot serve customers using facial recognition, video interview AI, voice systems | First prospect deploying a vision or voice model |
| D5 | Embedding bias analysis | Bias inherited from embedding layers goes undetected | First customer using a RAG or embedding-based system in a high-risk context |
| D6 | Differential privacy | Cross-customer benchmarking would leak statistical information about customer populations | Before any cross-customer benchmarking feature ships |
| D7 | Cross-customer benchmarking | Customers lack peer context for their scores | 20+ customers in the same sector |
| D8 | Self-hosted deployment | Cannot serve banks/insurers who refuse cloud | First regulated-industry customer requiring on-prem |
| D9 | Regulatory update agent | Rule sets go stale as regulations evolve | Manual rule maintenance exceeds 4 hours/month |
| D10 | Remediation recommendation engine | Product identifies problems without suggesting fixes | Customer feedback requesting fix guidance |
| D11 | Federated / privacy-preserving testing | Customers unwilling to share model or data at all | First prospect refusing to upload artifacts |
| D12 | Real-time inference monitoring | Only batch testing; no live production monitoring | Customer requires live decision monitoring |

**Rule:** no deferred item may be silently dropped. This register is reviewed at every phase completion and any triggered item is escalated into the active plan.

---

# THE 9 GAP CATEGORIES — WHERE EACH IS CLOSED

| Category | Closed In | Status |
|---|---|---|
| 1. Statistical holes | Phase 2 | Fully closed |
| 2. Missing model types | Phase 4 (+ D4, D5 deferred) | Closed except multimodal/embeddings |
| 3. LLM & agentic gaps | Phase 3 | Fully closed |
| 4. Security vulnerabilities | Phase 4 (+ D3 deferred) | Closed except third-party pen test |
| 5. Regulatory coverage | Phase 5 (+ D9 deferred) | Closed except automated update monitoring |
| 6. Data & privacy | Phase 4 + Phase 7 (+ D6 deferred) | Closed except differential privacy |
| 7. Architectural holes | Phase 7 (+ D8, D12 deferred) | Closed except self-hosted & real-time |
| 8. Product & business gaps | Phase 6 + Phase 7 (+ D7, D10 deferred) | Closed except benchmarking & remediation AI |
| 9. Rare gaps | Phases 2, 4, 5, 6 | Fully closed |

---

# REVISED TIMELINE

| Phase | Days | Focus | Status |
|---|---|---|---|
| 0 | 1–5 | Foundation | ✅ Complete |
| 1 | 6–25 | Structured ML testing | ✅ Complete (Week 3 validated 2026-09-02) |
| 2 | 26–40 | Statistical rigour | ⬜ Not started |
| 3 | 41–65 | LLM testing & red teaming | ⬜ Not started |
| 4 | 66–90 | Security, data governance, model coverage | ⬜ Not started |
| 5 | 91–115 | Compliance mapper & explainability | ⬜ Not started |
| 6 | 116–140 | Monitoring, workflow, dashboard | ⬜ Not started |
| 7 | 141–170 | Production architecture & legal | ⬜ Not started |
| 8 | 171–200 | Production UI, SDK, integrations | ⬜ Not started |
| 9 | Ongoing | Scale & certification | ⬜ Funding-dependent |

**Prototype demonstrable at end of Phase 6 (Day 140).**
**Sellable at end of Phase 7 (Day 170).**

Note: this is roughly double the original 90-day plan. That is the honest cost of closing the gaps. A 90-day prototype that ignores statistical rigour, LLM testing, and liability protection is faster to build and impossible to sell into a regulated industry.

---

# TOP RISKS TO THE BUILD

1. **Scope explosion** → Phase gates are mandatory. No phase starts before the previous phase's exit criteria are fully met.
2. **Statistical implementation errors** → every statistical function validated against `scipy`/`statsmodels` reference outputs with hand-computed test cases.
3. **LLM testing costs** → use Ollama locally at zero cost. Estimate and display API cost before any paid-model run.
4. **Regulatory mapping errors** → every article reference traced to primary source text. Legal review before first customer.
5. **Compliance overclaim** → the word "compliant" is banned from all outputs. Enforced by an automated test that greps all output templates.
6. **Deferred items forgotten** → Deferred Register reviewed at every phase completion.
7. **Burnout** → this is a 200-day plan, not 90. Sustainable pace matters more than any single sprint.

---

## Progress Tracking

`docs/PROGRESS.md` — updated every session.
`docs/GAP_CHECKLIST.md` — the authoritative 9-category tracker, updated at every phase completion.
