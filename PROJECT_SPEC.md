# Supply-Chain Information Diffusion Alpha in Chinese Equities
## Autonomous Research & Implementation Specification for Codex

**Version:** 2.0  
**Role:** Single source of truth and execution contract  
**Primary frequency:** Daily  
**Primary language:** Python 3.10+  
**Market:** Shanghai + Shenzhen A-shares  
**Research objective:** Test whether firm-specific information shocks diffuse with delay across disclosed customer-supplier links, then determine whether any surviving signal is economically implementable after China-market execution constraints and costs.

---

# 0. How Codex Must Use This File

This specification is not a research outline. It is an **autonomous execution contract**.

Codex must:

1. read this file before modifying research code;
2. inspect the current repository and existing status artifacts;
3. resume from the first phase whose status is not `PASS` or `SKIPPED_BY_DESIGN`;
4. implement phases in order;
5. run the validation criteria for each phase;
6. write a machine-readable phase-status artifact;
7. stop when a hard gate fails or required external data are unavailable;
8. never continue by inventing data, weakening a gate after seeing results, or silently changing the research question.

The project is allowed to finish with:

```text
POSITIVE
NULL
INCONCLUSIVE
INFEASIBLE_DATA
```

A profitable strategy is **not** required for project success.

The project is successful if it produces a reproducible and scientifically valid conclusion.

---

# 1. Research Question and Frozen Hypotheses

## 1.1 Primary question

> After removing broad systematic return components, do firm-specific shocks to disclosed customers predict future abnormal returns of their suppliers?

The primary economic mechanism is delayed information incorporation along real economic links.

---

## 1.2 Primary hypothesis

For supplier \(i\) with customer set \(C_i(t)\):

\[
S^{C\rightarrow S}_{i,t}
=
\frac{1}{|C_i(t)|}
\sum_{j\in C_i(t)} \epsilon_{j,t}
\]

where \(\epsilon_{j,t}\) is the point-in-time residual return of customer \(j\).

Primary mechanism test:

\[
S^{C\rightarrow S}_{i,t}
\rightarrow
\epsilon_{i,t+1}.
\]

Primary horizon:

```text
1 trading day
```

Pre-specified decay diagnostics:

```text
3 trading days
5 trading days
10 trading days
```

---

## 1.3 Secondary directional hypothesis

Supplier shocks may predict customer returns:

\[
S^{S\rightarrow C}_{i,t}
=
\frac{1}{|S_i(t)|}
\sum_{j\in S_i(t)} \epsilon_{j,t}.
\]

This direction must be reported separately.

It must never replace the primary direction merely because it performs better.

---

## 1.4 Primary edge weighting

**Primary v1 signal uses equal-weight valid neighbors.**

Reason:

Supply-chain disclosures may report exposure from different economic perspectives:

- supplier sales share to a customer;
- customer procurement share from a supplier.

Those quantities are not automatically comparable after combining evidence from both endpoints.

Economic-weight signals are therefore robustness extensions only when the weight is interpretable from the target company's perspective.

---

## 1.5 What the project is not

This is not primarily:

- a pair-trading project;
- a graph-neural-network project;
- a generic stock-return ML project;
- a search over many graph transformations.

Graph-guided pair mean reversion is an optional benchmark only.

GNN/GraphSAGE/GAT models are post-v1 extensions.

---

# 2. Frozen Research Period and Information Timing

## 2.1 Target period

```text
2015-01-01 to 2025-12-31
```

Initial split:

```text
2015                 warm-up
2016-2020            development
2021-2022            validation
2023-2025            final holdout
```

The raw start date may move only if data availability makes 2015 impossible.

Any date change must occur:

1. before any forward-return result is inspected;
2. with a written reason in `reports/research_decisions.md`;
3. without moving the holdout boundary to improve results.

---

## 2.2 Signal timing

Baseline:

```text
day t close
    observe permitted day-t market information
    use supply-chain edges public by the signal cutoff
    compute residual returns
    compute graph diffusion signal

day t+1
    mechanism return already begins at the next close-to-close interval
    tradable portfolio may enter only at the permitted t+1 execution price
```

Mechanism returns and tradable returns must remain separate.

---

## 2.3 Disclosure timing

For every relationship:

```text
effective_start >= actual_publication_datetime
```

Never backdate a relationship to:

- fiscal year end;
- accounting period start;
- report period;
- relationship start inferred from prose.

If a disclosure is published after the configured day-t signal cutoff, it becomes usable on the next eligible signal date.

---

# 3. Global Engineering Contract

## 3.1 Status states

Every phase must end in exactly one state:

```text
PASS
FAIL
BLOCKED
SKIPPED_BY_DESIGN
```

Research conclusions use a different field:

```text
POSITIVE
NULL
INCONCLUSIVE
NOT_EVALUATED
```

Do not use `PASS` to mean positive alpha.

---

## 3.2 Required phase-status artifact

After each phase, write:

```text
reports/status/phase_<N>.json
```

Required fields:

```json
{
  "phase": 1,
  "name": "repository_and_schemas",
  "state": "PASS",
  "started_at": "...",
  "finished_at": "...",
  "git_commit": "...",
  "config_sha256": "...",
  "inputs": [],
  "outputs": [],
  "tests_run": [],
  "metrics": {},
  "criteria": {},
  "blockers": [],
  "notes": []
}
```

If Git is unavailable, write `git_commit = null` and include a SHA-256 source-tree manifest.

---

## 3.3 Allowed state transitions

```text
not_run -> PASS
not_run -> FAIL
not_run -> BLOCKED
not_run -> SKIPPED_BY_DESIGN
BLOCKED -> PASS     only after blocker resolution
FAIL -> PASS        only after code/data defect correction and revalidation
```

A phase must not be manually relabeled without rerunning its validation.

---

## 3.4 Orchestrator

Codex must implement:

```bash
python scripts/run_pipeline.py
```

and:

```bash
python scripts/run_pipeline.py --through-phase N
```

The orchestrator must:

1. read existing status artifacts;
2. verify upstream statuses;
3. execute the next permissible phase;
4. stop on `FAIL` or `BLOCKED`;
5. skip conditional phases only when this spec explicitly permits it;
6. never open final-holdout data before Phase 9 freeze.

---

## 3.5 Exit codes

Use:

```text
0 = requested work completed successfully
1 = engineering/test failure
2 = external-data or credential blocker
3 = research gate failed by design
4 = holdout integrity violation
```

A research gate failure is not an engineering error.

---

## 3.6 Reproducibility

Every full run must write:

```text
reports/run_manifest.json
```

with:

- UTC run timestamp;
- Python version;
- package versions;
- config hash;
- source-code hash;
- raw-data manifest hash;
- random seed;
- phase states;
- final artifact paths.

---

## 3.7 Raw-data policy

`data/raw/` is append-only.

Every raw file must have:

```text
source
source_url_or_identifier
retrieval_datetime
file_size
sha256
```

Never edit raw data in place.

Derived values belong in `data/interim/` or `data/processed/`.

---

## 3.8 Secrets

Credentials and tokens:

- must come from environment variables;
- must never be committed;
- must never be printed in logs;
- must never be embedded in notebooks.

If a required paid source needs unavailable credentials, use an allowed public fallback or mark the phase `BLOCKED`.

---

# 4. Repository Architecture

Codex should converge to:

```text
supply-chain-alpha/
├── README.md
├── PROJECT_SPEC.md
├── PHASE_1_3_STATUS.md
├── pyproject.toml
├── config/
│   ├── project.yaml
│   ├── entity_overrides.csv
│   └── market_rules.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── reports/
│   ├── status/
│   ├── decisions/
│   ├── phase3_coverage/
│   ├── phase7_validation/
│   ├── holdout/
│   └── final/
├── notebooks/
│   ├── 01_graph_coverage.ipynb
│   ├── 02_entity_resolution_audit.ipynb
│   ├── 03_signal_diagnostics.ipynb
│   └── 04_portfolio_diagnostics.ipynb
├── scripts/
│   ├── run_pipeline.py
│   ├── acquire_security_master.py
│   ├── acquire_disclosures.py
│   ├── ingest_disclosures.py
│   ├── build_supply_chain_graph.py
│   ├── run_phase3_coverage.py
│   ├── build_market_panel.py
│   ├── build_residual_returns.py
│   ├── build_signals.py
│   ├── run_signal_research.py
│   ├── run_portfolio_research.py
│   ├── freeze_research.py
│   ├── run_holdout.py
│   ├── build_final_report.py
│   └── validate_all.py
├── src/
│   └── supply_chain_alpha/
│       ├── data/
│       ├── entities/
│       ├── graph/
│       ├── factors/
│       ├── signals/
│       ├── evaluation/
│       ├── portfolio/
│       ├── backtest/
│       └── reporting/
└── tests/
    ├── unit/
    ├── integration/
    └── regression/
```

Core logic belongs in `src/`.

Notebooks are diagnostic/reporting surfaces only.

---

# 5. Canonical Data Contracts

## 5.1 `security_master`

Primary key:

```text
security_id
```

Required:

```text
security_id
ticker
exchange
company_name
listing_date
delisting_date
board
```

Preferred:

```text
industry_code
industry_name
industry_valid_from
industry_valid_to
```

Codes are strings.

Dates use ISO date types.

---

## 5.2 `company_alias`

Primary key:

```text
entity_id, alias, valid_from
```

Required:

```text
entity_id
canonical_name
alias
alias_type
valid_from
valid_to
source
```

Alias validity must be enforced at the disclosure publication timestamp.

---

## 5.3 `disclosure_raw`

One row per extracted relationship mention.

Required:

```text
source_company_id
counterparty_raw_name
relationship_type
source_period_end
publication_datetime
source_document_id
source_document_url_or_path
evidence_text
exposure_value
exposure_share
```

`relationship_type`:

```text
customer
supplier
```

`exposure_share` uses decimal units:

```text
0.20 = 20%
```

Values outside `[0, 1]` fail validation.

---

## 5.4 `resolution_audit`

Required:

```text
source_document_id
source_company_id
counterparty_raw_name
resolved_entity_id
resolution_method
resolution_confidence
resolution_status
candidate_count
notes
```

---

## 5.5 `supply_chain_edge`

Direction is frozen:

```text
supplier_id -> customer_id
```

Required:

```text
supplier_id
customer_id
effective_start
effective_end
source_document_id
source_period_end
source_company_id
relationship_confidence
economic_weight
weight_source
```

Duplicate evidence must not create duplicate graph exposure on one date.

---

## 5.6 `equity_daily`

Primary key:

```text
date, security_id
```

Required:

```text
date
security_id
open
close
volume
amount
tradable
```

Preferred:

```text
high
low
market_cap
free_float_market_cap
suspended
limit_up
limit_down
st_status
```

---

## 5.7 `returns_daily`

Required:

```text
date
security_id
raw_return
market_return
industry_return
residual_return
beta_market
beta_industry
residual_model
```

Unavailable industry fields remain null only if the explicitly documented fallback model is used.

---

## 5.8 `signal_daily`

Required:

```text
date
security_id
customer_shock
supplier_shock
customer_neighbor_count
supplier_neighbor_count
graph_snapshot_date
```

Optional:

```text
customer_shock_weighted
supplier_shock_weighted
edge_age_mean
edge_age_max
```

---

# 6. Data-Source Hierarchy

## 6.1 General rule

Prefer:

```text
official source
    -> reproducible structured public source
    -> licensed local dataset already supplied
```

Never scrape an unstable third-party page if an official or structured source is available.

---

## 6.2 Disclosures

Primary target:

- CNINFO / 巨潮资讯 or equivalent official listed-company disclosure source.

Every downloaded disclosure must retain:

- document id;
- publication timestamp;
- source URL;
- local raw path;
- hash.

If automated access is blocked:

1. retry with respectful rate limiting;
2. use official downloadable endpoints if available;
3. use already supplied structured disclosure data;
4. otherwise mark Phase 2 `BLOCKED`.

Do not bypass access controls.

---

## 6.3 Security master and market data

Preferred:

- SSE / SZSE official data;
- reproducible structured public adapters;
- supplied licensed data.

Public-library adapters such as AKShare are permitted if:

1. the endpoint is reproducible;
2. provenance is stored;
3. a small sample is cross-checked against an official source.

---

## 6.4 Industry classification

Preferred:

point-in-time industry history with validity dates.

If point-in-time industry classification cannot be obtained:

- do not use current classification retroactively;
- use the market-only residual model as the primary fallback;
- mark same-industry placebo and industry-neutral portfolio as unavailable;
- report this limitation explicitly.

Lack of industry history alone does not block the core project.

---

# 7. Phase 1 — Repository, Schemas, and Validation Framework

## Objective

Create the reproducible engineering foundation.

## Required outputs

- repository structure;
- project config;
- canonical schemas;
- PK assertions;
- raw-data manifest;
- deterministic test fixture;
- status writer;
- pipeline orchestrator skeleton.

## Validation criteria — PASS

Phase 1 is `PASS` only if all are true:

- package installs in a clean environment;
- `pytest` for Phase-1 tests passes;
- canonical schemas reject duplicate primary keys;
- `exposure_share > 1` is rejected;
- security identifiers remain strings;
- raw-manifest mutation test catches changed files;
- project config parses and required fields exist;
- no absolute machine-specific paths exist in production code;
- no secrets are present in tracked files;
- the status artifact itself validates against its required schema.

## FAIL

Any deterministic schema/test/reproducibility defect.

## BLOCKED

Only if the runtime cannot install required open-source dependencies.

## Current checkpoint

Existing implementation is expected to satisfy most Phase-1 requirements. Codex must re-run, not assume, `PASS`.

---

# 8. Phase 2 — Real Data Acquisition, Disclosure Extraction, and Entity Resolution

## Objective

Produce auditable real-world relationship mentions and entity mappings.

This phase includes four submodules.

---

## 8.1 Security-master acquisition

Build all historical Shanghai/Shenzhen listed common equities needed for the target window.

Must preserve:

```text
listing_date
delisting_date
ticker history if available
company-name history if available
```

### Success criteria

- duplicate `security_id` count = 0;
- `listing_date < delisting_date` whenever delisting exists;
- ticker stored as string;
- exchange is valid and non-null for 100% of rows;
- at least 95% of securities expected from the chosen source have non-null listing dates;
- source provenance and raw hash are present.

---

## 8.2 Disclosure acquisition

Acquire annual reports and/or other public disclosures likely to contain named top customers/suppliers.

The acquisition layer must cache documents.

### Success criteria

- every downloaded document has publication timestamp, document id, source location, retrieval timestamp, and SHA-256 hash;
- duplicate document ids are deduplicated deterministically;
- rerunning acquisition does not redownload unchanged documents unnecessarily;
- failed downloads are logged with explicit reason;
- no synthetic disclosure enters production raw data.

---

## 8.3 Relationship extraction

Extract both named and anonymized customer/supplier mentions.

Anonymous mentions must be recorded as mentions but never resolved to securities.

Accepted mention must contain source-grounded evidence text.

### Source-section coverage audit

Among source documents identified as containing customer/supplier disclosure sections:

```text
captured_section_rate >= 0.80
```

A section is "captured" if the parser produces either:

- at least one named mention; or
- at least one explicit anonymous mention.

This measures parser coverage without pretending anonymized names are usable edges.

### Evidence audit

On a deterministic stratified sample of up to 200 accepted named mentions:

```text
evidence_support_rate >= 0.98
```

`evidence_support_rate` means the source snippet or structured source field visibly supports the extracted raw counterparty name.

If fewer than 50 accepted named mentions exist, mark the audit `LIMITED_SAMPLE` and continue to Phase 3 only if all other criteria pass.

---

## 8.4 Entity resolution

Order:

```text
normalized exact alias
    -> deterministic identifier
    -> high-confidence unique fuzzy match
    -> manual override
    -> unresolved
```

Rules:

- anonymous labels never resolve;
- ambiguous exact aliases remain unresolved;
- fuzzy matching requires both score and uniqueness margin;
- alias validity is filtered at publication time;
- every accepted mapping retains method and confidence;
- manual overrides are version-controlled.

### Resolution QA

On a deterministic stratified sample of up to 200 accepted mappings:

```text
source-supported mapping rate >= 0.98
```

A mapping is source-supported when:

- exact legal/alias evidence is present; or
- the counterparty string and resolved canonical entity are unambiguous under the stored rules.

No self-generated LLM explanation counts as evidence.

---

## Phase-2 PASS criteria

All are required:

- security-master checks pass;
- disclosure provenance completeness = 100%;
- anonymous-to-listed mappings = 0;
- accepted mappings without audit trail = 0;
- source-section coverage >= 80%;
- evidence support >= 98% when auditable;
- alias-validity regression tests pass;
- future data appended to alias history do not alter earlier mappings;
- production raw-data manifest verifies.

## FAIL

Examples:

- anonymous labels map to securities;
- publication timestamps are missing;
- entity mappings are not reproducible;
- raw-data provenance is lost;
- extraction evidence materially disagrees with source.

## BLOCKED

Examples:

- official disclosure source cannot be accessed and no legitimate fallback exists;
- required market/security master source is unavailable.

If blocked, write a blocker report and stop.

---

# 9. Phase 3 — Point-in-Time Graph and Real Coverage Gate

## Objective

Determine whether the real data can support a daily cross-sectional diffusion study.

No return data may be used to choose graph-coverage thresholds.

---

## 9.1 Edge construction

For company X disclosure:

```text
X names customer Y  -> X -> Y
X names supplier Y  -> Y -> X
```

with graph convention:

```text
supplier -> customer
```

Baseline:

```text
effective_start = publication_datetime
effective_end   = next contradictory/replacement evidence
                  or configured max age
                  or listing-interval end
```

Initial maximum edge age:

```text
550 calendar days
```

This may be changed only in development/validation after Phase 3 passes, never based on final holdout.

---

## 9.2 Duplicate-evidence rule

Multiple documents supporting the same directed pair on date \(t\) create one graph edge.

Preserve all evidence separately, but collapse graph exposure.

---

## 9.3 Mandatory coverage report

Produce:

```text
total relationship mentions
explicit-name rate
resolution rate conditional on explicit name
unique directed listed-to-listed edges
active edges by year
connected listed stocks by year
supplier-node count
customer-node count
degree distribution
edge-age distribution
industry distribution if PIT industry exists
source-document distribution
```

---

## 9.4 Frozen feasibility thresholds

These are engineering minimums, not alpha-tuned hyperparameters.

Phase 3 requires at least **three consecutive calendar years** within development+validation with:

```text
active listed-to-listed directed edges at year-end >= 200
connected listed stocks at year-end              >= 150
```

Across those usable years:

```text
median connected stocks >= 200
```

If point-in-time industry data are available:

```text
active edges span >= 5 industries
largest single-industry share of active edges <= 0.60
```

If PIT industry data are unavailable, defer only the concentration criterion.

Additionally:

```text
timing violations = 0
anonymous resolved nodes = 0
duplicate graph exposure for same pair/date = 0
missing source provenance on active edges = 0
```

---

## Phase-3 PASS criteria

All mandatory integrity checks pass and the frozen feasibility thresholds are met.

## FAIL

Coverage is genuinely below the frozen thresholds, or graph integrity fails.

If coverage fails, the project concludes:

```text
INFEASIBLE_DATA
```

and Codex must create a short feasibility report instead of building alpha models.

## BLOCKED

Real disclosures or security-master data are not yet available.

## Current checkpoint

Engineering and synthetic tests exist, but the real-data gate is currently expected to be `BLOCKED` until production A-share inputs are acquired.

---

# 10. Phase 4 — Point-in-Time Market Panel and Tradability

## Objective

Create corporate-action-safe daily returns and execution-state fields for graph-connected securities.

---

## 10.1 Market data

At minimum:

```text
date
security_id
open
close
volume
amount
```

The panel should additionally identify, directly or through validated rules:

```text
suspension
price-limit state
ST/*ST state
listing age
tradability
```

---

## 10.2 Corporate actions

Preferred:

- source-provided total-return/adjustment factors with documented semantics; or
- explicit split/dividend/rights data.

Do not assume a backward-adjusted price history is PIT-safe.

Test it.

### Future-action invariance test

Compute historical one-period returns through \(T\).

Append later corporate actions and rerun.

Returns through \(T\) must remain unchanged within numerical tolerance.

---

## 10.3 Trading calendar

Use actual trading dates.

No calendar-day lagging.

---

## 10.4 Market-rules history

Create:

```text
config/market_rules.yaml
```

with validity intervals for rules used in execution, including where applicable:

- T+1 restrictions;
- board-specific price limits;
- ST price-limit treatment;
- transaction taxes/fees.

Each rule must have a source note.

Do not apply today's rule retroactively to the whole sample.

---

## 10.5 Market-panel coverage criteria

Across development+validation:

```text
>= 95% of graph-connected security-days
```

should have valid daily close/return data when the stock is listed and not suspended.

At least:

```text
>= 90% of trading days
```

must have usable market data for at least 90% of graph-connected listed securities expected to trade that day.

---

## Phase-4 PASS criteria

- primary key duplicates = 0;
- invalid negative volume/amount values = 0 after source-specific validation;
- future-action invariance test passes;
- trading-calendar alignment tests pass;
- listing-date universe tests pass;
- market-panel coverage thresholds pass;
- market-rule provenance exists;
- appending future market data does not change historical features.

## FAIL

Examples:

- historical returns change when later data are appended;
- market coverage is below threshold;
- execution-state logic cannot distinguish suspended/untradeable observations.

## BLOCKED

No reproducible market-data source is available.

---

# 11. Phase 5 — Residual-Return Engine

## Objective

Separate firm-specific shocks from broad systematic returns using only information available by day \(t\).

---

## 11.1 Preferred residual model

If valid PIT industry history exists:

\[
r_{i,t}
=
\alpha_{i,t-1}
+
\beta^M_{i,t-1}r^M_t
+
\beta^I_{i,t-1}r^{I(i)}_{t}
+
\epsilon_{i,t}.
\]

Beta estimation window:

```text
252 trailing trading days
```

Minimum valid observations:

```text
126
```

Betas for residual on day \(t\) must use data ending no later than \(t-1\).

---

## 11.2 Industry-factor self-inclusion

If an internally computed industry return is used, it must be **leave-one-out** for target stock \(i\).

Do not mechanically subtract part of the target's own return through the industry factor.

---

## 11.3 Fallback residual model

If PIT industry history is unavailable:

\[
r_{i,t}
=
\alpha_{i,t-1}
+
\beta^M_{i,t-1}r^M_t
+
\epsilon_{i,t}.
\]

Label:

```text
residual_model = market_only
```

Do not fabricate historical industry labels.

---

## 11.4 Residual diagnostics

Required:

```text
beta distributions
residual mean/std
raw vs residual market correlation
raw vs residual industry correlation when available
coverage through time
extreme residual frequency
```

---

## Phase-5 PASS criteria

Across development+validation:

- beta-estimation leakage tests pass;
- residual dated \(t\) uses betas estimated through \(t-1\);
- at least 90% of otherwise eligible graph target-days have residual returns after warm-up;
- finite residuals on valid observations >= 99.9%;
- at least 80% of sufficiently observed stocks show lower absolute market correlation for residual returns than raw returns;
- median absolute residual-vs-market correlation <= 0.15;
- if industry model is used, at least 70% of sufficiently observed stocks show reduced absolute industry correlation;
- future-data invariance tests pass.

These criteria validate the residualization engine, not alpha.

## FAIL

Residualization materially fails to remove the intended systematic component or contains leakage.

## BLOCKED

Required market data are unavailable.

---

# 12. Phase 6 — Baseline Diffusion Signal Engine

## Objective

Build the two pre-specified directional signals without using future returns.

---

## 12.1 Customer-to-supplier signal

Graph convention:

```text
supplier -> customer
```

For target supplier \(i\), use active outgoing customer neighbors:

\[
S^{customer}_{i,t}
=
\frac{1}{N_{i,t}}
\sum_{j \in Customers(i,t)} \epsilon_{j,t}.
\]

---

## 12.2 Supplier-to-customer signal

For target customer \(i\), use active incoming supplier neighbors:

\[
S^{supplier}_{i,t}
=
\frac{1}{N_{i,t}}
\sum_{j \in Suppliers(i,t)} \epsilon_{j,t}.
\]

---

## 12.3 Missing-neighbor rule

A neighbor contributes only if:

- edge is active at \(t\);
- neighbor is listed at \(t\);
- neighbor has valid day-t residual return.

Do not fill missing neighbor shocks with zero.

---

## 12.4 Weighted extensions

Only after equal-weight signal is generated.

A weighted signal is permitted only when `weight_source` is economically interpretable for the target exposure.

Do not mix incompatible sales-share and procurement-share weights without explicit normalization logic.

---

## 12.5 Signal-coverage gate

Across development+validation:

```text
median daily eligible target count >= 100
25th percentile daily eligible target count >= 50
```

for the primary customer-to-supplier signal.

---

## Phase-6 PASS criteria

- synthetic edge-direction tests pass;
- graph snapshot never uses future edges;
- target stock never appears as its own neighbor;
- equal-weight formula exactly matches hand-computed fixtures;
- signal-return alignment has no future overlap;
- median and 25th-percentile eligible-target thresholds pass;
- future graph/market observations do not change historical signals.

## FAIL

Signal logic is incorrect or the real graph-market intersection is too sparse.

If coverage fails:

```text
research_conclusion = INCONCLUSIVE
reason = insufficient_daily_signal_coverage
```

and do not build a portfolio.

---

# 13. Phase 7 — Development and Validation Research

## Objective

Determine whether the pre-specified signal contains incremental predictive information before final holdout.

This phase uses only:

```text
development + validation
```

No 2023-2025 data may be loaded into research-result code.

---

## 13.1 Primary evaluation

For each date:

\[
IC_t =
SpearmanCorr(S^{customer}_{i,t}, \epsilon_{i,t+1}).
\]

Report:

```text
mean IC
median IC
ICIR
positive-IC fraction
Newey-West uncertainty
block-bootstrap confidence interval
```

Pearson IC is diagnostic only.

---

## 13.2 Forward-return horizons

Pre-specified:

```text
1d primary
3d decay
5d decay
10d diagnostic
```

Do not add new horizons after validation merely to improve significance.

---

## 13.3 Fama-MacBeth-style regression

Daily cross-section:

\[
\epsilon_{i,t+h}
=
a_t
+
b_t GraphShock_{i,t}
+
c_t Controls_{i,t}
+
u_{i,t+h}.
\]

Core controls:

```text
target residual return at t
target 5d residual momentum/reversal
size if PIT available
liquidity
volatility
graph degree
edge age
```

Unavailable optional controls are omitted and documented.

---

## 13.4 Quantile test

Daily signal ranks:

```text
Q1 Q2 Q3 Q4 Q5
```

Required outputs:

- average forward residual return by quantile;
- Q5-Q1 spread;
- monotonicity score;
- day-level spread series.

---

## 13.5 Mandatory placebos

### Placebo A — Same-industry shock

Required only if PIT industry history is valid.

### Placebo B — Trailing-correlation network

Neighbors selected from historical returns ending no later than \(t-1\).

### Placebo C — Matched randomized graphs

At least:

```text
500 random seeds
```

approximately preserving:

- degree distribution;
- industry mix when available;
- size/liquidity bucket where feasible.

Random seeds must be fixed before validation results are finalized.

### Placebo D — Target own-return baseline

Include contemporaneous target residual and short-term reversal/momentum.

---

## 13.6 Robustness allowed before freeze

Pre-specified robustness only:

```text
customer -> supplier vs supplier -> customer
equal vs interpretable economic weights
fresh vs old edges
high vs low liquidity
large vs small cap
1-neighbor vs multi-neighbor
market-only vs market+industry residual
development vs validation
```

Exploratory analyses must be labeled `EXPLORATORY`.

They cannot redefine the primary signal.

---

## 13.7 Validation promotion gate for portfolio research

The primary signal is promoted to Phase 8 only if all are true:

```text
development mean rank IC > 0
validation mean rank IC > 0
validation 1d Q5-Q1 residual return > 0
validation block-bootstrap one-sided p-value < 0.10
real-graph validation IC exceeds the 95th percentile
of matched-random placebo mean ICs
```

If a required placebo is impossible because a non-core field is unavailable, document and use the remaining placebos; the random-graph placebo remains mandatory.

---

## Phase-7 PASS criteria

Engineering/research phase is `PASS` when:

- all pre-specified primary/secondary tests run;
- all available mandatory placebos run;
- uncertainty estimates are generated;
- development and validation are reported separately;
- no holdout date is loaded;
- every tested variant is recorded in `reports/phase7_validation/test_registry.csv`;
- a portfolio-promotion decision is written.

`PASS` does not require the promotion gate to be positive.

Possible result:

```text
portfolio_candidate = true | false
```

---

# 14. Phase 8 — Portfolio, Execution, and Cost Model

## Entry condition

Run only if:

```text
portfolio_candidate = true
```

Otherwise:

```text
state = SKIPPED_BY_DESIGN
```

The project still proceeds to the holdout mechanism test.

---

## 14.1 Research portfolio

Diagnostic portfolio:

```text
long top signal quantile
short bottom signal quantile
```

This is a **simulated research portfolio**.

Do not call it live executable unless borrow/funding availability is modeled.

---

## 14.2 Neutralization

Preferred constraints when fields are available:

```text
market beta approximately neutral
industry approximately neutral
size exposure controlled
single-name weight limited
```

Optimization must have deterministic fallbacks.

If the optimizer fails, do not silently use unconstrained weights.

---

## 14.3 Execution

Signal uses day-t close.

Earliest baseline entry:

```text
t+1 open
```

For a newly purchased A-share long position, exit timing must respect historical T+1 constraints.

Reject/defer fills when the historical market state makes them unrealistic, including:

- suspension;
- unavailable execution price;
- relevant price-limit condition;
- configured minimum-liquidity failure.

---

## 14.4 Costs

Separate:

```text
statutory/venue fees
taxes
commission assumption
slippage assumption
```

Historical statutory changes must use validity intervals.

If reliable bid/ask data are unavailable, run slippage sensitivity:

```text
5 bps/side
10 bps/side
20 bps/side
```

plus documented statutory costs.

Also report:

```text
break-even slippage
```

---

## 14.5 Development/validation strategy diagnostics

Required:

```text
gross return
net return
annualized volatility
Sharpe
max drawdown
turnover
hit rate
average holding period
gross/net exposure
beta exposure
industry exposure if available
break-even cost
```

---

## 14.6 Strategy promotion gate

A portfolio is promoted to final holdout strategy evaluation only if, on validation:

```text
net cumulative return > 0
break-even total trading cost > baseline modeled total trading cost
no persistent material market-beta exposure
no execution-timing violation
```

Do not require a high validation Sharpe.

The gate asks only whether the signal survives a minimal economic-implementability check.

---

## Phase-8 PASS criteria

- execution timing tests pass;
- T+1 tests pass;
- no-fill logic tests pass;
- position constraints pass;
- cost calculations reconcile from trades to portfolio P&L;
- turnover is reproducible;
- validation diagnostics are complete;
- `strategy_candidate = true | false` is frozen.

The phase can `PASS` even when `strategy_candidate = false`.

---

# 15. Phase 9 — Research Freeze and Holdout Lock

## Objective

Freeze every decision that could affect final results before any holdout statistic is computed.

---

## 15.1 Required freeze artifact

Create:

```text
reports/research_freeze.json
```

containing:

```text
research question
primary direction
primary signal formula
edge-aging rule
edge weighting
residual model
beta window
forward horizons
controls
placebos
portfolio_candidate
strategy_candidate
portfolio rules if applicable
cost assumptions
all hyperparameters
config SHA-256
source-tree SHA-256
raw-data-manifest SHA-256
validation artifact hashes
freeze timestamp
```

---

## 15.2 Holdout lock

Before the freeze file exists:

- any function requesting dates >= `holdout_start` must raise a hard error in research-result modules.

After freeze:

- holdout run is allowed.

The first holdout run writes:

```text
reports/holdout/holdout_run_manifest.json
```

A rerun is allowed only if code/config/freeze hashes are identical.

If holdout-result-affecting code changes after the first run:

```text
state = FAIL
exit_code = 4
research_integrity = HOLDOUT_CONTAMINATED
```

The project may still report the contaminated analysis, but it must not call it untouched OOS.

---

## Phase-9 PASS criteria

- freeze artifact exists;
- all required fields are non-null or explicitly `not_applicable`;
- hashes verify;
- holdout-lock unit test passes;
- no pre-freeze holdout output exists;
- validation artifacts are immutable or hash-tracked.

---

# 16. Phase 10 — Final Holdout

## Objective

Run the frozen research once on 2023-2025 data and classify the evidence.

No tuning is allowed.

---

## 16.1 Primary holdout mechanism test

Required:

```text
1d customer->supplier rank IC
block-bootstrap 95% CI
Newey-West inference
Q1-Q5 forward residual returns
Q5-Q1 spread
Fama-MacBeth coefficient
matched-random placebo distribution
```

Secondary direction is reported separately.

---

## 16.2 Evidence classification

### POSITIVE

Classify primary mechanism evidence as `POSITIVE` only if all are true:

```text
holdout mean 1d rank IC > 0
95% block-bootstrap CI lower bound > 0
holdout Q5-Q1 residual return > 0
real-graph holdout mean IC exceeds the 95th percentile
of the frozen matched-random placebo distribution
development, validation, and holdout point estimates
have the same primary sign
```

### NULL

Classify as `NULL` when:

```text
95% CI includes 0
```

and the study is sufficiently powered to rule out a practically meaningful effect.

Use empirical block-bootstrap variability to report the minimum detectable mean rank IC.

If:

```text
minimum_detectable_abs_IC <= 0.005
```

and the observed effect remains statistically indistinguishable from zero with no placebo advantage, `NULL` is appropriate.

### INCONCLUSIVE

Use `INCONCLUSIVE` when:

- effective sample size is too small;
- CI is wide enough that economically relevant effects remain plausible;
- signs materially conflict across periods;
- results are highly sensitive to one robustness slice;
- real graph is indistinguishable from matched placebo despite a positive raw IC.

Do not force a binary positive/negative story.

---

## 16.3 Holdout strategy evaluation

Run only if:

```text
strategy_candidate = true
```

Use frozen:

- weights;
- neutralization;
- execution;
- costs;
- holding rules.

Report gross and net results.

Do not optimize after seeing holdout P&L.

---

## 16.4 Strategy evidence labels

Separate from mechanism evidence:

```text
ECONOMICALLY_VIABLE
NOT_ECONOMICALLY_VIABLE
NOT_EVALUATED
```

`ECONOMICALLY_VIABLE` requires:

```text
holdout net cumulative return > 0
break-even total cost > modeled baseline total cost
no material persistent forbidden factor exposure
```

This does not imply live executability.

---

## Phase-10 PASS criteria

The phase passes if:

- the frozen holdout run completes;
- no freeze hash changed;
- all primary outputs are generated;
- an evidence classification is assigned;
- strategy evaluation is run or correctly marked `NOT_EVALUATED`;
- no post-holdout parameter selection is performed.

`PASS` is compatible with `NULL`.

---

# 17. Phase 11 — Final Report, Reproducibility, and Project Closure

## Objective

Produce the final research package without overstating evidence.

---

## 17.1 Required final artifacts

```text
reports/final/FINAL_REPORT.md
reports/final/EXECUTIVE_SUMMARY.md
reports/final/results_summary.json
reports/final/limitations.md
reports/final/reproducibility.md
README.md
```

Optional rendered PDF is allowed after Markdown is finalized.

---

## 17.2 Required final tables

At minimum:

1. data-source and provenance summary;
2. disclosure/entity-resolution QA;
3. graph coverage by year;
4. market-panel coverage;
5. residual-model diagnostics;
6. development/validation signal results;
7. placebo comparison;
8. final holdout mechanism results;
9. portfolio results if evaluated;
10. cost/turnover sensitivity if evaluated.

---

## 17.3 Required final figures

At minimum:

1. graph coverage through time;
2. degree distribution;
3. representative directed subgraph;
4. daily IC time series;
5. signal decay;
6. quantile forward returns;
7. primary vs secondary direction;
8. real graph vs matched-placebo distribution;
9. holdout cumulative P&L if strategy evaluated;
10. factor exposure / turnover if strategy evaluated.

---

## 17.4 Claim discipline

Allowed wording depends on result.

### If POSITIVE

May state:

> Found out-of-sample evidence consistent with delayed information diffusion from customers to suppliers in the tested A-share disclosure network.

Do not state causality unless the design supports causality.

### If NULL

State:

> Did not find robust out-of-sample evidence that the tested public supply-chain network adds predictive information at the pre-specified horizons.

### If INCONCLUSIVE

State exactly why inference is limited.

### If INFEASIBLE_DATA

Report the graph/data coverage failure as the main result.

---

## 17.5 Resume bullet generation

Generate resume wording only from actual final results.

Never insert:

- Sharpe;
- IC;
- return;
- significance;
- "alpha";
- "profitable"

unless supported by frozen final outputs.

---

## Phase-11 PASS criteria

- every numerical claim traces to a saved artifact;
- report and `results_summary.json` agree;
- no holdout tuning is hidden;
- limitations are explicit;
- `python scripts/validate_all.py` passes;
- a clean environment can reproduce the final summary from canonical processed data;
- README contains exact run instructions;
- final research conclusion is one of the allowed states.

---

# 18. Phase 12 — Optional Extensions After v1 Closure

These are not required for project completion.

Run only after Phase 11 is frozen.

Possible extensions:

```text
event-conditioned diffusion
multi-hop A^2 shocks
edge-age decay
interpretable economic weights
graph embeddings
GraphSAGE
GAT
temporal GNN
graph-guided pair-trading benchmark
```

Rules:

- v1 holdout remains frozen;
- extensions require a new experiment registry;
- do not retroactively redefine v1;
- compare incremental value against the simple 1-hop baseline;
- use a new future holdout or nested validation if making new performance claims.

---

# 19. Mandatory Test Registry

Codex must implement and maintain at least these tests.

## Data and schema

- duplicate primary-key rejection;
- code/string preservation;
- percentage-unit validation;
- raw-manifest mutation detection.

## Entity resolution

- anonymous-label blocking;
- ambiguous exact alias remains unresolved;
- alias validity at publication time;
- manual override reproducibility;
- future alias append invariance.

## Graph

- supplier/customer direction;
- no pre-publication edge;
- listing-interval intersection;
- duplicate-evidence collapse;
- no self-edge unless explicitly permitted and justified;
- edge-expiry behavior.

## Market

- trading-calendar alignment;
- listing-period validity;
- corporate-action future invariance;
- suspended/untradeable state;
- historical market-rule validity interval.

## Residuals

- beta window ends at \(t-1\);
- no future scaler/regression input;
- leave-one-out industry factor;
- residual future-data invariance.

## Signals

- hand-computed customer shock;
- hand-computed supplier shock;
- missing-neighbor exclusion;
- no self-neighbor contamination;
- signal date / future-return alignment.

## Statistics

- IC agrees with hand-computed fixture;
- bootstrap deterministic under seed;
- placebo graph preserves required properties within tolerance;
- test registry captures every run.

## Portfolio

- signal at t cannot fill before t+1;
- T+1 exit constraint;
- no-fill on invalid market state;
- weight/exposure constraints;
- trade-level costs reconcile to portfolio P&L.

## Holdout

- holdout inaccessible before freeze;
- identical-hash rerun allowed;
- changed-code/config rerun rejected as contaminated.

---

# 20. Module Validation Matrix

| Phase | Module | PASS means | If not PASS |
|---|---|---|---|
| 1 | Repo / schemas | Tests, schemas, manifests, status protocol are reproducible | `FAIL` / fix engineering |
| 2 | Real data + entity resolution | Source-grounded disclosures and reproducible mappings meet QA | `BLOCKED` if source unavailable; `FAIL` if QA invalid |
| 3 | PIT graph coverage | Integrity clean and frozen minimum real coverage reached | `INFEASIBLE_DATA`, stop alpha research |
| 4 | Market panel | PIT-safe returns/tradability and required coverage | `FAIL/BLOCKED`, stop |
| 5 | Residual returns | Leakage-safe residualization removes broad market component | `FAIL`, fix model/data |
| 6 | Diffusion signals | Correct direction/timing and sufficient daily target coverage | `INCONCLUSIVE`, no portfolio |
| 7 | Dev/validation research | All pre-specified tests/placebos complete; promotion decision frozen | Continue regardless of positive/negative |
| 8 | Portfolio model | Conditional execution/cost engine valid; strategy decision frozen | May `SKIPPED_BY_DESIGN` |
| 9 | Freeze | Hash-locked research design before holdout | Holdout forbidden |
| 10 | Holdout | Frozen OOS run completed and evidence classified | `FAIL` if integrity broken |
| 11 | Final report | All claims reproducible and consistent with evidence | Project not complete |
| 12 | Extensions | New experiments separated from v1 | Optional only |

---

# 21. Automated Decision Tree

Codex must follow:

```text
Phase 1 PASS?
    no -> fix / stop
    yes
        ↓
Phase 2 PASS?
    blocked -> blocker report / stop
    fail -> fix / stop
    yes
        ↓
Phase 3 PASS?
    no coverage -> INFEASIBLE_DATA / final feasibility report / stop
    yes
        ↓
Phase 4 PASS?
    no -> stop
    yes
        ↓
Phase 5 PASS?
    no -> fix / stop
    yes
        ↓
Phase 6 PASS?
    no coverage -> INCONCLUSIVE / final data-limitation report / stop
    yes
        ↓
Phase 7 PASS
    portfolio candidate?
        yes -> Phase 8
        no  -> Phase 8 SKIPPED_BY_DESIGN
        ↓
Phase 9 FREEZE PASS
        ↓
Phase 10 HOLDOUT
        -> POSITIVE / NULL / INCONCLUSIVE
        -> strategy viable / not viable / not evaluated
        ↓
Phase 11 FINAL REPORT
```

---

# 22. Multiple Testing and Research Registry

Every model/signal test after Phase 6 must append to:

```text
reports/experiment_registry.csv
```

Required fields:

```text
experiment_id
timestamp
phase
sample_period
signal_name
direction
horizon
edge_weighting
residual_model
controls
purpose
pre_specified
result_artifact
```

Rules:

- never delete failed experiments;
- exploratory tests remain visible;
- final primary claim uses only the frozen primary specification;
- do not present the best exploratory variant as if pre-specified.

---

# 23. Logging and Data Quality

Every production script must:

- use `logging`;
- log input row counts;
- log output row counts;
- log duplicate count;
- log missingness in required fields;
- log date range;
- log source version/identifier;
- fail loudly on schema violation.

Never silently:

- coerce invalid dates;
- convert percentages without unit rules;
- drop duplicates without a documented deduplication key;
- fill missing prices;
- map unresolved entities;
- substitute another dataset.

---

# 24. Performance and Scalability

Correctness dominates speed.

Preferred:

- Parquet canonical tables;
- vectorized Pandas/NumPy where clear;
- partition large daily panels by date/year if needed;
- cache expensive graph snapshots;
- avoid rebuilding unchanged raw sources.

Before optimization:

- pass the tiny deterministic integration test;
- pass all PIT tests.

Do not introduce distributed computing unless required by measured runtime/memory.

---

# 25. Required Tiny End-to-End Fixture

Maintain a deterministic fixture with approximately:

```text
10-30 securities
3-10 directed edges
120-300 trading days
named and anonymous counterparties
one company rename
one future listing
one suspension
one price-limit case
one corporate action
```

The fixture must run:

```text
raw source
-> disclosure extraction
-> entity resolution
-> PIT graph
-> market panel
-> residuals
-> signals
-> IC
-> portfolio fill rules
-> P&L
```

before full-scale production research.

---

# 26. Config Values That Must Be Frozen Before Holdout

At minimum:

```text
research periods
signal cutoff
edge max age
entity-resolution thresholds
primary edge weighting
residual model
beta window
minimum beta observations
signal horizons
minimum graph coverage
minimum daily signal coverage
controls
placebo count and seeds
quantile count
portfolio promotion rule
portfolio constraints
execution price
holding rule
cost assumptions
market-rule schedule
```

No code-level hidden defaults for research decisions.

---

# 27. Current Repository Checkpoint

As of the Phase 1-3 engineering checkpoint:

- Phase 1 engineering exists;
- Phase 2 structured ingestion/entity-resolution engineering exists;
- Phase 3 graph/coverage engineering exists;
- synthetic tests have passed in the existing checkpoint;
- real Phase-3 coverage has not yet been established because production A-share inputs were not present.

Codex must **revalidate the current repository**, then continue with real-data acquisition.

Do not treat synthetic success as empirical graph coverage.

---

# 28. Definition of Project Completion

The v1 project is complete only when one of the following terminal paths is reached.

## Path A — Full empirical completion

```text
Phase 1-11 complete
final conclusion = POSITIVE / NULL / INCONCLUSIVE
```

## Path B — Data infeasibility

```text
Phase 1-3 valid
real graph fails frozen coverage
final conclusion = INFEASIBLE_DATA
feasibility report produced
```

## Path C — Later hard blocker

A required legitimate data source becomes unavailable after Phase 3.

Required output:

```text
BLOCKER_REPORT.md
```

containing:

- exact missing dependency;
- attempted legitimate sources;
- why substitution would violate the spec;
- last valid completed phase.

This is not considered a completed empirical project, but it is a valid engineering stop.

---

# 29. Final Research Narrative Template

The final report should answer only these questions:

1. Could a reproducible point-in-time listed-company supply-chain graph be constructed?
2. How broad and stable was the graph?
3. Did customer shocks predict supplier abnormal returns?
4. Was the result directional?
5. Did the real economic graph beat generic correlation and random-network placebos?
6. Did the effect decay as expected?
7. Did it survive the untouched holdout?
8. If a strategy was promoted, did the signal survive execution and costs?
9. What market/data limitations prevent stronger claims?

The project should be understandable without mentioning a GNN.

---

# 30. Codex Prohibitions

Codex must not:

- use holdout data before the freeze artifact;
- move the holdout boundary after inspecting results;
- backdate supply-chain relationships;
- map anonymous customers/suppliers to listed companies;
- use current company aliases before their valid dates;
- use current industry classification retroactively without historical validity;
- fit residual betas using day-t or future returns when constructing day-t betas;
- use target stock inside its own internally calculated industry factor;
- forward/backward fill missing stock prices;
- treat missing neighbor shocks as zero;
- mix incompatible edge weights without justification;
- search many horizons and report only the best;
- delete failed experiments from the registry;
- tune thresholds after holdout;
- call a statistical relationship causal without a causal design;
- call a cash-equity long/short simulation executable without modeling borrow/funding;
- implement a GNN before v1 closes;
- weaken a failed phase criterion simply to continue the project;
- fabricate data to satisfy a gate.

---

# 31. Final Validation Command

Codex must make this command the final project check:

```bash
python scripts/validate_all.py
```

It must verify:

```text
all required phase status files exist
all upstream phases are valid
raw manifest verifies
canonical table PKs validate
PIT regression tests pass
experiment registry exists
research freeze hash verifies
holdout manifest hash verifies
final numerical claims trace to artifacts
README run instructions exist
```

Expected final output format:

```text
PROJECT VALIDATION: PASS
RESEARCH CONCLUSION: POSITIVE | NULL | INCONCLUSIVE | INFEASIBLE_DATA
STRATEGY EVIDENCE: ECONOMICALLY_VIABLE | NOT_ECONOMICALLY_VIABLE | NOT_EVALUATED
```

Anything else means the project is not yet closed.
