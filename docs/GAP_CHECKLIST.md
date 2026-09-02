# Gap Closure Checklist — All 9 Categories

**Purpose:** Every gap identified in the architectural review, tracked to closure.
**Rule:** No phase is complete until every gap assigned to it is ticked.
**Review cadence:** At every phase completion, and before any customer conversation.

Legend: ⬜ Not started · 🟡 In progress · ✅ Closed · ⏸️ Deferred (see Deferred Register)

---

## CATEGORY 1 — Statistical Holes
**Assigned to: Phase 2** · **Target: Day 40**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 1.1 | Multiple comparisons problem — 5 simultaneous tests inflate false positives | Bonferroni + Benjamini-Hochberg correction; compliance uses corrected threshold | 2 | ⬜ |
| 1.2 | Thresholds arbitrary and legally undefended | THRESHOLDS.md with honest "no regulatory basis" statement; per-system configurable; changes audit-logged | 2 | ⬜ |
| 1.3 | No confidence intervals — bare point estimates only | Bootstrap CI (1,000 iterations) on every metric; stored on every TestResult | 2 | ⬜ |
| 1.4 | No significance testing — cannot distinguish real bias from noise | Chi-squared, Fisher's exact, two-proportion z-test, permutation test; p-value on every metric | 2 | ✅ |
| 1.5 | Simpson's Paradox undetected — aggregate hides subgroup reversal | Automatic stratification by all low-cardinality columns; flag reversals | 2 | ⬜ |
| 1.6 | Label bias unaddressed — biased ground truth treated as correct | label_bias.py — assess label distribution vs external base rates; flag enforcement-proxy labels | 4 | ⬜ |
| 1.7 | Feedback loop bias undetectable | feedback_loop.py — correlate past decisions with training data composition shifts | 6 | ⬜ |
| 1.8 | No reliability assessment — unstable metrics reported as fact | N-run stability testing; three-tier flag; "insufficient_data" blocks verdict | 2 | ⬜ |
| 1.9 | Impossibility theorem ignored — conflicting metrics presented without explanation | tensions.py — base rate comparison; explain mathematically unachievable combinations | 2 | ⬜ |

---

## CATEGORY 2 — Missing Model Types
**Assigned to: Phase 4** · **Target: Day 90**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 2.1 | Regression models only partially supported | regression.py — mean difference, variance ratio, Wasserstein, calibration by group, error parity | 4 | ⬜ |
| 2.2 | Ranking models not supported at all | ranking.py — exposure parity, nDCG parity, rank-position disparity, top-k representation | 4 | ⬜ |
| 2.3 | Multimodal models absent (vision, audio, video) | Deferred — D4 | 9 | ⏸️ |
| 2.4 | Embedding models / vector bias undetected | Deferred — D5 | 9 | ⏸️ |
| 2.5 | Continuously learning models — verdicts never expire | Expiring verdicts; mandatory retest cadence for continuous_learning systems | 6 | ⬜ |
| 2.6 | Fine-tuned model alignment degradation | Alignment regression testing — base vs fine-tuned probe comparison | 3 | ⬜ |
| 2.7 | Quantisation behavioural drift | Quantisation drift testing — full precision vs deployed precision | 3 | ⬜ |

---

## CATEGORY 3 — LLM & Agentic Gaps
**Assigned to: Phase 3** · **Target: Day 65**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 3.1 | No LLM testing capability at all | Full llm/ module — adapter, probes, engine | 3 | ⬜ |
| 3.2 | Prompt injection untested | 40+ injection probes | 3 | ⬜ |
| 3.3 | Jailbreak resistance untested | 50+ jailbreak probes (DAN, roleplay, hypothetical) | 3 | ⬜ |
| 3.4 | System prompt extraction untested | 20+ extraction probes | 3 | ⬜ |
| 3.5 | PII leakage from LLM untested | 25+ leakage probes | 3 | ⬜ |
| 3.6 | Hallucination rate unmeasured | Three methods — known-answer, self-consistency, retrieval grounding | 3 | ⬜ |
| 3.7 | Toxicity unmeasured | detoxify classification on all outputs, local on M4 | 3 | ⬜ |
| 3.8 | Generative demographic bias unmeasured | Demographic-varied prompts; sentiment/length/lexical/recommendation disparity with Phase 2 statistics | 3 | ⬜ |
| 3.9 | Encoding bypass attacks untested | 20+ base64/rot13/unicode obfuscation probes | 3 | ⬜ |
| 3.10 | Over-refusal unmeasured (usability failure) | 20+ legitimate-request probes | 3 | ⬜ |
| 3.11 | RAG-specific attacks untested | rag_attacks.py — KB poisoning, retrieval manipulation, context stuffing, citation fabrication | 3 | ⬜ |
| 3.12 | Indirect prompt injection untested | 25+ probes via processed content | 3 | ⬜ |
| 3.13 | Agent tool misuse untested | agent_safety.py with mock tools — injection-to-tool-call, privilege escalation, confused deputy | 3 | ⬜ |
| 3.14 | Multi-agent manipulation untested | Multi-agent probe scenarios | 3 | ⬜ |
| 3.15 | No automated red teaming | Garak (NVIDIA) integration, results parsed to our schema | 3 | ⬜ |
| 3.16 | LLM non-determinism unhandled | generate_n() with variance measurement; temperature 0.0 default | 3 | ⬜ |

---

## CATEGORY 4 — Security Vulnerabilities
**Assigned to: Phase 4** · **Target: Day 90**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 4.1 | Adversarial examples untested | adversarial.py — FGSM/PGD via IBM ART; robustness score | 4 | ⬜ |
| 4.2 | Group-conditional robustness untested | Robustness measured per demographic group — rarely tested fairness issue | 4 | ⬜ |
| 4.3 | Model inversion attacks undetected | inversion.py via ml-privacy-meter | 4 | ⬜ |
| 4.4 | Membership inference undetected | membership.py — GDPR-relevant privacy attack | 4 | ⬜ |
| 4.5 | Model stealing vulnerability unassessed | extraction.py — query-count-to-clone estimation | 4 | ⬜ |
| 4.6 | Backdoor / trojan attacks undetected | backdoor.py — activation anomaly detection, trigger scanning | 4 | ⬜ |
| 4.7 | Supply chain attacks on pre-trained models | provenance.py — artifact hashing, HuggingFace card checks, unverified source warnings | 4 | ⬜ |
| 4.8 | No third-party penetration testing | Deferred — D3 | 9 | ⏸️ |

---

## CATEGORY 5 — Regulatory Coverage
**Assigned to: Phase 5** · **Target: Day 115**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 5.1 | EU AI Act implementing acts still unpublished — overclaiming compliance | `status` field on every rule; pending standards cap verdict at "partial — awaiting technical standard" | 5 | ⬜ |
| 5.2 | UK AI regulation absent | uk_ai_framework.json — five principles + ICO guidance | 5 | ⬜ |
| 5.3 | US state law absent (NYC LL144, Colorado, California, Illinois) | Four US state rule sets | 5 | ⬜ |
| 5.4 | ISO 42001 mentioned but never mapped | iso_42001.json — all clauses including organisational | 5 | ⬜ |
| 5.5 | NIST AI RMF absent | nist_ai_rmf.json — GOVERN/MAP/MEASURE/MANAGE | 5 | ⬜ |
| 5.6 | Sector regulation incomplete (DORA, ECOA, FHA, FDA) | sector_finance.json, sector_health.json | 5 | ⬜ |
| 5.7 | Article 11 documentation completeness unchecked | documentation.py — 9-item checklist feeding Art.11 verdict | 5 | ⬜ |
| 5.8 | Article 14 human oversight unverifiable | Honest scope statement; check documented mechanism + override rate; state operational verification requires manual audit | 5 | ⬜ |
| 5.9 | EU AI Act GPAI provisions (Title VIII) absent | Added to eu_ai_act.json | 5 | ⬜ |
| 5.10 | No mechanism to update rules as regulations evolve | Versioned JSON, no code change required; automated monitoring deferred — D9 | 5 | ⬜ |

---

## CATEGORY 6 — Data & Privacy
**Assigned to: Phases 4 and 7** · **Target: Day 170**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 6.1 | Training data completely untested — only test data examined | data_governance/ module — full training data audit suite | 4 | ⬜ |
| 6.2 | Dataset imbalance unhandled (class, group, intersectional) | balance.py — imbalance detection, min-30-per-group warning, stratified sampling | 4 | ⬜ |
| 6.3 | Data lineage untracked | lineage.py — source, date, consent basis, transformations, retention | 4 | ⬜ |
| 6.4 | GDPR Art.22 right to explanation not implemented | explanation.py — plain-language natural language explanation with counterfactuals | 5 | ⬜ |
| 6.5 | Data minimisation unchecked | minimisation.py — utility vs privacy-risk feature ranking | 4 | ⬜ |
| 6.6 | Cross-border transfer unaddressed | Data residency config; automatic transfer-safeguard flagging | 7 | ⬜ |
| 6.7 | Proxy variable chains undetected (multi-step) | proxy_chains.py — causal graph analysis via dowhy/networkx | 4 | ⬜ |
| 6.8 | Test input data retained indefinitely | Delete input data after run; retain results only | 7 | ⬜ |
| 6.9 | PII in logs | Structured logging with PII scrubbing | 7 | ⬜ |
| 6.10 | No right-to-erasure implementation | Full tenant data deletion within 30 days | 7 | ⬜ |
| 6.11 | Differential privacy for benchmarking | Deferred — D6 | 9 | ⏸️ |

---

## CATEGORY 7 — Architectural Holes
**Assigned to: Phase 7** · **Target: Day 170**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 7.1 | No authentication — any caller can hit any endpoint | Keycloak OIDC; auth required on every endpoint, verified by test | 7 | ⬜ |
| 7.2 | No authorisation / RBAC | 5 roles: Owner, Admin, Analyst, Viewer, API Key | 7 | ⬜ |
| 7.3 | No multi-tenancy — all data commingled | org_id on every table; app-layer scoping + PostgreSQL RLS backstop | 7 | ⬜ |
| 7.4 | No cross-tenant isolation test | Automated test: Org A cannot reach Org B by any path | 7 | ⬜ |
| 7.5 | Synchronous job processing — HTTP will time out on real models | Celery + Redis; async jobs with progress polling | 7 | ⬜ |
| 7.6 | SQLite single-writer concurrency limit | PostgreSQL migration via Alembic | 7 | ⬜ |
| 7.7 | No rate limiting | Per-tenant rate limits on all endpoints | 7 | ⬜ |
| 7.8 | No input size limits — 2GB upload would crash the process | Max upload size, max rows with disclosed sampling, request timeout | 7 | ⬜ |
| 7.9 | No test configuration versioning — results not comparable over time | config_version + engine_version pinned on every run; UI states comparability | 7 | ⬜ |
| 7.10 | No audit log | Immutable append-only audit_log with evidence-package export | 7 | ⬜ |
| 7.11 | No self-hosted deployment option | Deferred — D8 | 9 | ⏸️ |
| 7.12 | No real-time inference monitoring | Deferred — D12 | 9 | ⏸️ |

---

## CATEGORY 8 — Product & Business Gaps
**Assigned to: Phases 6 and 7** · **Target: Day 170**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 8.1 | 🔴 No liability protection — compliance asserted with no disclaimer | LIMITATIONS.md embedded in every report, dashboard view, and API response | 7 | ⬜ |
| 8.2 | 🔴 No Terms of Service or DPA | Drafted by Irish solicitor before first customer | 7 | ⬜ |
| 8.3 | 🔴 No professional indemnity insurance | In force before first paying customer | 7 | ⬜ |
| 8.4 | 🔴 Product says "compliant" — legally dangerous overclaim | Banned word; output is "no issues detected against tested criteria"; enforced by automated grep test | 7 | ⬜ |
| 8.5 | No remediation tracking — cannot answer "when did you know" | remediation.py — full lifecycle, accepted-risk requires named approver | 6 | ⬜ |
| 8.6 | No human review workflow — automated verdicts are final | human_review.py — confirm/override/escalate, mandatory reasoning, audit-logged | 6 | ⬜ |
| 8.7 | No pre-deployment approval gate — only documents harm after the fact | approval_gate.py — blocks production promotion without passing recent test; CI/CD webhook | 6 | ⬜ |
| 8.8 | No trend analysis — point-in-time only | Metrics plotted over time; direction and rate surfaced | 6 | ⬜ |
| 8.9 | No peer benchmarking | Internal benchmarking in Phase 6; cross-customer deferred — D7 | 6 / 9 | ⬜ |
| 8.10 | No remediation recommendations | Deferred — D10 | 9 | ⏸️ |

---

## CATEGORY 9 — Rare Gaps
**Assigned to: Phases 2, 4, 5, 6** · **Target: Day 140**

| # | Gap | Fix | Phase | Status |
|---|---|---|---|---|
| 9.1 | Intersectional bias mathematically invisible | intersectional.py — 2-way and 3-way combinations, small-cell guarding, Bonferroni across intersections | 4 | ⬜ |
| 9.2 | Aggregation bias — one model for a diverse population | Subgroup performance comparison; flag when per-group models would outperform | 4 | ⬜ |
| 9.3 | Temporal bias — fair at training, unfair later | temporal_bias.py — fairness metrics tracked over time with degradation alerts | 6 | ⬜ |
| 9.4 | Measurement invariance unchecked | Feature-outcome relationship consistency testing across groups | 4 | ⬜ |
| 9.5 | Validation datasets frozen in time (1994, 2016, 1990s) | Documented limitation in VALIDATION_RESULTS.md; supplement with a modern dataset when available | 5 | ⬜ |
| 9.6 | Proxy variable chains (multi-step) undetected | proxy_chains.py — causal graph analysis | 4 | ⬜ |
| 9.7 | Model documentation completeness unchecked | documentation.py — Article 11 checklist | 5 | ⬜ |
| 9.8 | Human-in-the-loop operational effectiveness unverifiable | Override-rate analysis + explicit honest scope statement | 5 | ⬜ |
| 9.9 | Simpson's Paradox | Covered in 1.5 | 2 | ⬜ |
| 9.10 | Model version lineage untracked | model_versions table with hash chain | 4 | ⬜ |

---

# PHASE GATE SUMMARY

No phase may begin until the previous phase's gaps are all ticked.

| Phase | Gaps Closed | Count | Must All Be ✅ Before |
|---|---|---|---|
| Phase 1 (Week 3) | Validation ✅ passed 2026-09-02 | — | Phase 2 |
| Phase 2 | 1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 1.9, 9.9 | 8 | Phase 3 |
| Phase 3 | 2.6, 2.7, 3.1–3.16 | 18 | Phase 4 |
| Phase 4 | 1.6, 2.1, 2.2, 4.1–4.7, 6.1, 6.2, 6.3, 6.5, 6.7, 9.1, 9.2, 9.4, 9.6, 9.10 | 21 | Phase 5 |
| Phase 5 | 5.1–5.10, 6.4, 9.5, 9.7, 9.8 | 14 | Phase 6 |
| Phase 6 | 1.7, 2.5, 8.5, 8.6, 8.7, 8.8, 8.9, 9.3 | 8 | Phase 7 |
| Phase 7 | 6.6, 6.8, 6.9, 6.10, 7.1–7.10, 8.1, 8.2, 8.3, 8.4 | 18 | First customer |
| Deferred | 2.3, 2.4, 4.8, 6.11, 7.11, 7.12, 8.10 | 7 | See Deferred Register |

**Total gaps identified: 94**
**Closed in build plan: 87**
**Explicitly deferred with risk statement: 7**
**Silently ignored: 0**

---

# THE FOUR RED-FLAG GAPS

These four must be closed before a single euro changes hands. If forced to choose, close these first.

| Gap | Why It Is Existential |
|---|---|
| **8.1 — No liability protection** | A customer relying on a "compliant" verdict who then faces regulatory action may hold you responsible. No disclaimer, no defence. |
| **8.2 — No Terms of Service / DPA** | Processing customer personal data with no legal agreement is itself a GDPR violation. |
| **8.3 — No professional indemnity insurance** | One claim without insurance ends the business personally. |
| **8.4 — The word "compliant"** | Asserting legal compliance without being a legal service is the single most dangerous sentence the product can output. |

---

# REVIEW LOG

| Date | Phase Completed | Gaps Closed | Reviewer | Notes |
|---|---|---|---|---|
| 2026-09-02 | Phase 1 (Week 3 validation) | — (validation gate, no gap rows) | Rhishikumar | Adult 0.1745 / COMPAS 0.1752 / German 0.1149 — all in published range. `docs/VALIDATION_RESULTS.md` written. pytest 20/20. Deferred Register reviewed — no triggers fired. Phase 2 unblocked. |
| 2026-09-02 | Phase 2 Component 2.1 (partial — significance testing) | 1.4 | Rhishikumar | `governance/testing/statistics.py`: chi-squared / Fisher / z-test / permutation + "auto" selector + always-on permutation cross-check. Chi-squared bit-identical to `scipy.chi2_contingency`; Fisher & z-test match to 1e-12; permutation converges within 0.005 across seeds. Permutation p-value uses add-one smoothing (Phipson & Smyth 2010) — bounded away from 0. pytest 28/28. Phase 2 NOT complete — 1.1, 1.2, 1.3, 1.5, 1.8, 1.9, 9.9 still open. |

*Add a row at every phase completion. Review the Deferred Register at the same time and escalate any triggered item.*
