# THRESHOLDS.md — Justification for Every Numeric Threshold

**Purpose:** every number in this platform that turns a measurement into a
pass / warn / fail / block decision, documented with its exact value, where it
came from, who set it, and whether it can be changed.

**Compiled:** 2026-09-02, during Phase 2 Component 2.2 (gap 1.2), by reading the
current code in `governance/testing/bias.py` and `governance/testing/statistics.py`
directly — not from memory. Every value below was verified against the source at
that date.

**Owner:** Rhishikumar Ayyappan (architect / product owner). Values proposed
during the build and confirmed by the owner on 2026-09-02.

---

## THE HONEST LIMITATION STATEMENT — read this first

None of the numeric thresholds in this document are mandated by the EU AI Act,
GDPR, or any regulation this platform maps against. The EU AI Act requires that
AI systems be tested for bias and that risks be identified and managed — it does
not specify what magnitude of disparity constitutes unacceptable bias. Every
threshold below is an internal engineering convention, documented here for
transparency and auditability, not asserted as a legal standard. Customers
deploying this platform in a regulated context should establish their own
thresholds with appropriate legal counsel, informed by their specific sector,
jurisdiction, and risk tolerance.

---

## How the fairness thresholds are applied

For the four "gap" metrics (demographic parity, equalized odds, equal
opportunity, predictive parity) a **lower value is better** — it is a disparity,
and 0 means no measured disparity. The status logic in
`BiasTestSuite._get_status()` is:

| Measured value | Status |
|---|---|
| `value ≤ threshold × _PASS_BAND_FRACTION`  (i.e. `≤ 0.07` at the default) | **pass** |
| `0.07 < value ≤ threshold`  (i.e. `≤ 0.10`) | **warn** |
| `value > threshold`  (i.e. `> 0.10`) | **fail** |

**The real pass boundary is 0.07, not 0.10.** The `0.10` in
`BiasTestSuite.DEFAULT_THRESHOLDS` is only the fail boundary; a warn band sits
between 0.07 and 0.10. See threshold #6 below (`_PASS_BAND_FRACTION`, gap 9.11).

For `overall_accuracy_floor` the logic is **inverted** — it is a floor, higher
is better, and there is no warn band: `pass` if `value ≥ 0.80`, else `fail`.

---

## Part 1 — Fairness thresholds (`governance/testing/bias.py`)

### #1–#4 — The four gap metrics — default `0.10`

| Metric | Code key | Default |
|---|---|---|
| Demographic parity difference | `demographic_parity_difference` | `0.10` |
| Equalized odds difference | `equalized_odds_difference` | `0.10` |
| Equal opportunity difference | `equal_opportunity_difference` | `0.10` |
| Predictive parity difference | `predictive_parity_difference` | `0.10` |

**Source / reasoning — stated honestly:**

The `0.10` default traces to the **US EEOC "four-fifths rule"** (Uniform
Guidelines on Employee Selection Procedures, 1978; 29 CFR § 1607.4(D)). That rule
says: if one group's selection rate is less than **80%** of the highest group's
rate, that is treated as evidence of adverse impact. Expressed as a *ratio* the
rule is 0.80; expressed loosely as a *difference* in selection rates it is often
associated with a gap of about 0.20.

The `0.10` used here is **half of that 0.20** — a deliberately stricter internal
choice. Two honesty points:

1. **It is a US employment-law convention being reused here**, not a fairness
   standard with independent statistical grounding. The four-fifths rule was
   written for hiring and promotion decisions under US civil-rights law; there
   is no reason it transfers cleanly to, e.g., a credit-scoring model in the EU.
2. **The halving from 0.20 to 0.10 has no external basis** — it is an internal
   decision to be conservative.
3. For metrics #2, #3 and #4 the convention is applied even further from its
   origin: the four-fifths rule concerns *selection rates*, whereas equalized
   odds and equal opportunity are gaps in *error rates* (TPR / FPR) and
   predictive parity is a gap in *precision*. Using `0.10` for these is
   convention-by-analogy only — the four-fifths rule says nothing about error or
   precision parity.

**Who / when:** proposed during Phase 1 Week 2 (BiasTestSuite build), confirmed
by the owner 2026-09-02.

**Configurable:** the value can be overridden per `BiasTestSuite` instance via
the `__init__(thresholds=...)` argument (present since Phase 1 Week 2). It is
**not yet configurable per AI system with persistence or an audit trail** — see
the Configuration Roadmap below.

**What changing it means:** a lower threshold (say 0.05) flags smaller
disparities but raises the false-positive rate — more fair models get a warn or
fail on noise. That interacts with the Phase 2 multiple-comparison correction
(`apply_multiple_comparisons_correction`): the correction already tightens the
*significance* bar across the 5 tests, but the *effect-size* threshold here is
separate and a lower value works against it. A higher threshold (say 0.15) does
the reverse — fewer false alarms, more real disparities passed as acceptable.

---

### #5 — `overall_accuracy_floor` — default `0.80` (inverted floor)

> **Renamed from `individual_fairness_score` on 2026-09-02 (gap 9.12).** The old
> name implied a consistency property the metric does not have. The metric,
> value and threshold are **unchanged** — this was a pure rename; Phase 1
> validation numbers are bit-identical (see `docs/VALIDATION_RESULTS.md`).

**Value:** `0.80`. Passes if `value ≥ 0.80`, fails otherwise. No warn band.

**What it measures:** `MetricFrame(metrics=accuracy_score, ...).overall` — the
model's **overall predictive accuracy over the whole test set**, checked against
a floor. It is *not* a fairness metric — it does not compare groups (the
`sensitive_features` argument only feeds an informational, unused `by_group`
breakdown in the result detail). A genuine individual-fairness consistency
metric ("do similar individuals receive similar predictions") is tracked
separately as gap **9.13**, assigned to Phase 4.

**Source / reasoning — there is no external source. This is an internal
engineering default.**

- There is no regulation or widely-accepted standard that sets a minimum model
  accuracy of 80% (or any figure). Acceptable accuracy is entirely
  task-dependent — an 80%-accurate model can be worthless under class imbalance,
  and a naive predictor of a rare event can score 99%.
- The most likely origin of `0.80` is numerical coincidence with the four-fifths
  (80%) rule. If so, that is a category error: the four-fifths rule is about the
  *ratio of selection rates between groups*, not a model's absolute accuracy.
- The academic concept of *individual fairness* (Dwork et al., *Fairness Through
  Awareness*, 2012) — the origin of the old name — prescribes no numeric
  threshold and is not what this metric computes.

**Who / when:** default set during Phase 1 Week 2; renamed 2026-09-02.

**Configurable:** same as #1–#4 — `__init__` override only, no per-system
persistence yet.

**What changing it means:** raising it (say 0.90) rejects more models on
accuracy grounds; lowering it (say 0.70) passes more. It remains an
accuracy floor either way, not a fairness measure.

---

### #6 — `_PASS_BAND_FRACTION = 0.7` — pass/warn band-splitter

**Value:** `0.7`. In `BiasTestSuite._get_status()`, a gap metric passes when
`value <= threshold * self._PASS_BAND_FRACTION` and warns between there and
`threshold`. The same constant is `_PASS_BAND_FRACTION` in `statistics.py`
(`detect_simpsons_paradox`).

**Effect:** for the default `0.10` metrics, the actual **pass** boundary is
`0.10 × 0.7 = 0.07`. Values between 0.07 and 0.10 are **warn**. Anyone reading
`DEFAULT_THRESHOLDS` alone would reasonably conclude the pass/fail line is 0.10 —
it is not; 0.10 is the fail line and 0.07 is the pass line.

**Source / reasoning:** none. It is an internal decision to carve a
"close to the limit" warn band at 70% of the threshold. No external basis, no
regulatory basis, no statistical basis. It is a reasonable idea (a buffer zone
before outright failure).

**Who / when:** introduced during Phase 1 Week 2 as an inline `0.7`; discovered
and documented 2026-09-02 (gap **9.11**); extracted to the named class constant
`_PASS_BAND_FRACTION` with an explanatory comment on 2026-09-02 — **gap 9.11
closed.**

**Configurable:** **no** — it is not in `DEFAULT_THRESHOLDS`; it is a class-level
constant. Changing it changes the width of the warn band for all four gap
metrics at once.

---

## Part 2 — Statistical thresholds (`governance/testing/statistics.py`)

### #7 — `ALPHA = 0.05` — significance level

The raw significance level, before multiple-comparison correction. A p-value
below 0.05 is called "significant".

**Source:** R. A. Fisher, *Statistical Methods for Research Workers* (1925). The
p < 0.05 convention is the single most widely used threshold in applially all of
practically all of empirical science — and it is exactly as arbitrary as any
alternative (0.01, 0.10). Its value is that everyone uses it, not that 5% is a
principled cutoff.

**Configurable:** yes, per call (`alpha=` argument on
`apply_multiple_comparisons_correction`). Not persisted per system.

### #8 — `MIN_EXPECTED_CELL_COUNT = 5` — chi-squared validity

If any expected cell count in the 2×2 contingency table is below 5, the
chi-squared approximation is unreliable and the code falls back to Fisher's exact
test.

**Source:** the standard rule of thumb for chi-squared validity, usually
attributed to W. G. Cochran (1952, 1954). A genuine statistical convention, not
an internal invention.

**Configurable:** no — module constant. Changing it would change when the
Fisher's-exact fallback triggers.

### #9 — `Z_TEST_MIN_COUNT = 5` — normal-approximation validity

For the two-proportion z-test, the normal approximation is considered valid only
when `n·p̂ > 5` and `n·(1−p̂) > 5` for both groups (using the pooled proportion).
Below that the result carries `assumptions_met = False`.

**Source:** the standard textbook rule for the normal approximation to the
binomial (some texts use 5, some 10). Genuine convention.

**Configurable:** no — module constant.

### #10 — `min_group_size = 30` — reliability hard floor

In `assess_reliability`, if any group has fewer than 30 samples the result is
`insufficient_data` and **cannot produce a compliance verdict**.

**Source / reasoning:** the classic "n ≥ 30" rule of thumb for the central limit
theorem to give a usable normal approximation. It is a rule of thumb, not a
theorem — 30 is not magic. The reasoning established in Component 2.3: below
roughly this size, no fairness estimate on that group is trustworthy regardless
of what p-value or interval it produces. It matches the minimum-cell guard
planned for Phase 4 intersectional testing.

**Configurable:** yes, per call (`min_group_size=` argument). Not persisted.

**What changing it means:** a lower floor (say 20) lets thinner data through to a
verdict; the CI-width (#11) and bootstrap-SD (#12) rules are the backstop that
should still catch genuinely imprecise estimates in the 20–30 range. A higher
floor (say 50) is more conservative — more results come back "indeterminate".

### #11 — `CI_WIDTH_UNSTABLE_THRESHOLD = 0.15`

In `assess_reliability`, if the 95% bootstrap CI is wider than 0.15 the result is
`unstable` (reported, but flagged as imprecise — not blocked).

**Source:** engineering judgment, made during this build (Component 2.3,
2026-09-02). No external standard. Reasoning: a fairness gap estimate whose 95%
interval spans more than 0.15 (e.g. [0.03, 0.19]) cannot support a precise
pass/fail against a 0.10 threshold — the answer could be either side.

**Configurable:** no — module constant today.

### #12 — `BOOTSTRAP_SD_UNSTABLE_THRESHOLD = 0.05`

In `assess_reliability`, if the standard deviation of the bootstrap distribution
exceeds 0.05 the result is `unstable`.

**Source:** engineering judgment, this build (Component 2.3). No external
standard. Complementary to #11 — catches a skewed distribution whose SD is high
even if the percentile interval happens to look narrower.

**Configurable:** no — module constant today.

### #13 — `SINGLE_GROUP_SKIP_THRESHOLD = 0.05` and `SINGLE_GROUP_SKIP_BLOCK_THRESHOLD = 0.05`

Two separate constants, both currently `0.05` ("more than 5% of bootstrap
resamples degenerated to a single group"):

- `SINGLE_GROUP_SKIP_THRESHOLD` — in `bootstrap_confidence_interval`, above this
  the `BootstrapResult` carries a `reliability_warning`.
- `SINGLE_GROUP_SKIP_BLOCK_THRESHOLD` — in `assess_reliability`, above this the
  result is `insufficient_data` and blocks the verdict.

**Source:** engineering judgment, this build. No external standard. They are kept
as two constants (not one) because they govern different consequences and could
diverge later.

**Configurable:** no — module constants today.

### #14 — `DEFAULT_BOOTSTRAP_ITERATIONS = 1000` and `DEFAULT_N_PERMUTATIONS = 1000`

Number of resamples for the bootstrap CI and for the permutation test / cross-check.

**Source:** common practice. Efron & Tibshirani (*An Introduction to the
Bootstrap*, 1993) suggest 1,000–2,000 resamples for confidence intervals; 1,000
is the usual floor. Enough for stable 2.5 / 97.5 percentiles; cheap enough to run
in a test suite.

**Configurable:** yes, per call (`n_iterations=`, `n_permutations=`). More
iterations = tighter Monte-Carlo precision, linearly more runtime.

### #15 — `confidence_level = 0.95` (default) — bootstrap CI coverage

Paired with `ALPHA = 0.05`: a 95% interval reports the 2.5th and 97.5th
percentiles of the bootstrap distribution.

**Source:** the same convention as #7, and just as arbitrary. 95% is universal by
agreement, not by principle.

**Configurable:** yes, per call (`confidence_level=`).

### #16 — Permutation p-values are bounded away from zero — by design

Not a threshold, but a deliberate numeric floor that belongs in this file.

The permutation test p-value is computed with **add-one smoothing**:
`p = (1 + k) / (1 + n)` where `k` is the number of shuffles at least as extreme
as observed and `n` is the number of shuffles. With `n = 1000` the smallest
possible p-value is `1 / 1001 ≈ 0.000999`.

**Why:** a raw `k / n` can return exactly `0.0`, which would assert that a
fairness gap "could never occur by chance" — a claim that is statistically
indefensible and dangerous to state as fact in a legal or regulatory context.
The add-one estimator (Phipson & Smyth, *Permutation P-values Should Never Be
Zero*, Statistical Applications in Genetics and Molecular Biology 9(1), 2010) is
the standard fix.

**Configurable:** no — it is intrinsic to the estimator.

### #17 — Implementation constants (not decision thresholds)

- `POSITIVE_LABEL = 1` — the code assumes binary predictions labelled `{0, 1}`
  and treats `1` as the positive / favourable outcome. A dataset that encodes
  the favourable outcome as `0` would invert every metric's sign (magnitudes are
  unaffected because the gap metrics take absolute values, but per-group rates in
  the detail would read backwards). Not configurable.
- `_FP_TOLERANCE = 1e-12` — floating-point slack in the permutation test's
  "as large or larger" comparison. Pure numerical hygiene. Not configurable.

### #18 — `BASE_RATE_DIFFERENCE_THRESHOLD = 0.05` — impossibility-theorem trigger

In `detect_metric_tensions` (Component 2.4), if the largest gap between any two
groups' base rates (positive rate in `y_true`) exceeds 0.05, base rates are
treated as "differing significantly" and known metric tensions (demographic
parity vs predictive parity, etc.) are reported as *mathematically expected*
rather than as a fault. Below 0.05, a disagreement between fairness metrics is
flagged as `unexplained_disagreement = True` — a genuine finding, not a
mathematical artefact.

**Source:** engineering judgment, this build (2026-09-02), following the
threshold named in BUILD_PLAN Component 2.4. No external standard sets the point
at which base-rate differences make the impossibility theorem "bite" — the
theorem applies at *any* non-zero difference; 0.05 is a pragmatic cutoff below
which the effect is treated as negligible. Stated as such.

**Configurable:** no — module constant today.

**What changing it means:** a lower value (say 0.02) treats more metric
disagreements as mathematically excused, fewer as genuine findings — more
lenient. A higher value (say 0.10) does the reverse — more disagreements are
surfaced as real problems needing investigation.

---

## Structural findings from this audit

Two issues surfaced while compiling this file. Both are tracked in
`docs/GAP_CHECKLIST.md`; recorded here so the reasoning is not lost.

### Gap 9.11 — the `0.7` band-splitter was undocumented in code — **CLOSED**

The real pass boundary for the four gap metrics is **0.07**, not the `0.10` in
`DEFAULT_THRESHOLDS`. Until this file, that fact existed nowhere in writing.

- **Documented:** Part 1, #6 (2026-09-02).
- **Code fixed:** the inline `0.7` in `BiasTestSuite._get_status()` was extracted
  to the named class constant `_PASS_BAND_FRACTION` with an explanatory comment
  (2026-09-02, same commit as the 9.12 rename). `statistics.py` already used
  `_PASS_BAND_FRACTION`.
- **Status:** ✅ closed.

### Gap 9.12 — `individual_fairness_score` was mislabelled — **CLOSED (rename)**

The metric measures **overall model accuracy** (`MetricFrame.overall`), not
individual consistency. Individual fairness in the literature means "similar
individuals receive similar predictions" — a consistency property.

- **Fix (owner decision, 2026-09-02): rename only.** `individual_fairness_score`
  → `overall_accuracy_floor` across `bias.py`, `statistics.py`, test files,
  `THRESHOLDS.md` #5, `VALIDATION_RESULTS.md`, and the `detect_metric_tensions`
  note. Pure rename — metric, value, threshold and Phase 1 validation numbers
  unchanged.
- **Status:** ✅ closed. Building a *genuine* consistency metric is a separate
  piece of work — gap **9.13**, assigned to Phase 4 (X-threading through
  `BiasTestSuite.run()`, nearest-neighbour comparison, benchmark validation,
  ~1–2 days).

---

## Configuration Roadmap — current reality as of 2026-09-02

**Where thresholds live today:**

- The five fairness thresholds are hard-coded as class-level defaults in
  `BiasTestSuite.DEFAULT_THRESHOLDS` (`governance/testing/bias.py`).
- Per-instance override has been supported since Phase 1 Week 2 Component 1 via
  `BiasTestSuite(thresholds={...})` — the `__init__` merges the supplied dict
  over the defaults.
- **There is no persistence and no audit trail.** Nothing records who changed a
  threshold, when, or why. Nothing stores a per-AI-system threshold set. A
  threshold override lives only for the lifetime of the `BiasTestSuite` object
  that received it.
- `_PASS_BAND_FRACTION` (#6) is a class constant, not overridable per instance.
- The statistical constants in `statistics.py` (#8, #9, #11, #12, #13, #16, #17)
  are module-level and not configurable per call; #7, #10, #14, #15 are function
  arguments with defaults.

**What still needs to be built (a remaining Phase 2 task, not this one):**

- Per-`AISystem` threshold storage on the system record.
- An audit-log entry every time a threshold changes — who, when, old value, new
  value, stated reason.
- A statement, once that lands, of **who is allowed to change thresholds**,
  **whether changes are logged** (they must be), and **whether historical
  results remain comparable** after a threshold change (they do not — a result
  computed under threshold 0.10 is not comparable to one computed under 0.15, and
  the UI must say so).

**This file must be updated when that wiring lands.**

---

## Review

| Date | Change | By |
|---|---|---|
| 2026-09-02 | Initial version. All 17 thresholds/constants documented from source. Gaps 9.11 and 9.12 raised. | Claude (Sonnet 5) + Rhishikumar |
| 2026-09-02 | Added #18 (`BASE_RATE_DIFFERENCE_THRESHOLD = 0.05`) with Component 2.4 (metric tension detection). | Claude (Sonnet 5) + Rhishikumar |
| 2026-09-02 | **Gap 9.12 rename** — `individual_fairness_score` → `overall_accuracy_floor` throughout (#5 retitled; pure rename, validation numbers bit-identical). **Gap 9.11 closed** — inline `0.7` in `_get_status()` extracted to `_PASS_BAND_FRACTION` class constant (#6 retitled). | Claude (Sonnet 5) + Rhishikumar |
