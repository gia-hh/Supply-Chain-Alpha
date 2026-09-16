# Supply-Chain Information Diffusion Alpha in Chinese Equities

A point-in-time research prototype testing whether disclosed customer–supplier relationships among Chinese A-share companies provide enough economically grounded network coverage to support systematic information-diffusion and pair-trading research.

The project was designed with a **pre-registered feasibility gate before return data, signal evaluation, or production backtesting**. The graph failed that gate, so the study stopped with an auditable `INFEASIBLE_DATA` conclusion rather than relaxing the sample ex post.

---

## Project Summary

This project asks a simple question:

> Can public annual-report disclosures be converted into a sufficiently dense, point-in-time supply-chain graph for systematic equity research?

To answer it, the pipeline:

1. audits and processes official annual-report data;
2. extracts customer / supplier relationship mentions;
3. resolves counterparties to listed securities using historical names and listing intervals;
4. constructs validity-dated directed supply-chain edges;
5. evaluates graph coverage against a frozen feasibility threshold;
6. permits return-based alpha research only if that threshold passes.

The project reached Step 5.

It **did not** proceed to production return testing because the real-data graph was too sparse under the precommitted research criteria.

---

## Key Results

| Metric | Result |
|---|---:|
| Annual reports processed | **30,126** |
| Official raw data processed | **115.66 GiB** |
| Customer / supplier mentions extracted | **171,041** |
| Listed-company directed relationships | **386** |
| Connected A-share securities | **487** |
| PDF extraction success rate | **99.92%** |
| Mentions with explicit counterparty names | **21.65%** |
| Explicit names mapped to listed entities | **1.86%** |
| Best yearly active-edge count | **116** |
| Frozen feasibility requirement | **≥200 active edges/year for 3 consecutive years** |
| Automated tests | **442** |
| Final research status | **`INFEASIBLE_DATA`** |

The main bottleneck was therefore **not document extraction**.

The evidence instead pointed to limited explicit counterparty disclosure and difficulty mapping disclosed economic entities to tradable listed securities.

### Why 30,126 reports but far fewer graph edges?

The pipeline audits the full annual-report corpus, but only a small fraction of reports ultimately contain disclosures that can be resolved into valid listed-company-to-listed-company relationships.

In other words:

```text
30,126 annual reports
        ↓
relationship disclosures
        ↓
explicitly named counterparties
        ↓
historically resolvable entities
        ↓
listed-company ↔ listed-company edges
        ↓
386 directed graph relationships
```

This distinction is important: **document-processing scale is not the same thing as usable graph coverage**.

---

## Research Design

The core design principle is:

> **Do not inspect return outcomes until the real-data graph is demonstrably rich enough to support the intended experiment.**

The workflow is therefore deliberately gated:

```text
Official raw disclosures
        ↓
SHA-256 / provenance audit
        ↓
Disclosure extraction
        ↓
Historical entity resolution
        ↓
Point-in-time directed graph
        ↓
Coverage diagnostics
        ↓
Frozen feasibility gate
        │
        ├── PASS → return / signal / portfolio research may proceed
        │
        └── FAIL → stop with INFEASIBLE_DATA
```

The real-data pipeline followed the second path.

---

## Point-in-Time Discipline

Supply-chain information is only allowed to enter the graph after it was publicly available.

The implementation tracks:

- actual disclosure publication timing;
- historical company names;
- security-code mappings;
- listing intervals;
- relationship validity periods;
- anonymous / unresolved counterparties;
- future-data invariance.

A graph snapshot cannot contain an edge before the underlying disclosure becomes observable.

This prevents retrospective knowledge from contaminating historical research.

---

## Data Provenance and Reproducibility

The data pipeline was designed to make each stage auditable.

Key safeguards include:

- per-file SHA-256 verification;
- source and provenance records;
- canonical schemas;
- deterministic entity normalization;
- manual-override audit paths;
- stage-level validation gates;
- explicit separation between raw, intermediate, and production artifacts.

The raw market / annual-report datasets are **not included in this repository**.

The repository contains the research code, schemas, diagnostics, tests, and reproducible pipeline structure.

---

## Entity Resolution

A major challenge is converting annual-report language into securities that existed and were tradable at the relevant historical date.

The resolution layer uses:

- normalized company names;
- historical aliases;
- security codes;
- listing intervals;
- validity-dated mappings;
- conservative blocking of anonymous or generic counterparties.

The design intentionally prefers **unresolved** over an aggressive false match.

That conservatism contributed to lower graph coverage but reduces false economic relationships.

---

## Feasibility Gate

Before any production return research, the project froze a minimum graph-coverage requirement:

> **At least 200 active supply-chain edges per year for three consecutive years.**

The best observed year contained only:

> **116 active edges**

The threshold therefore failed.

Crucially, the project stopped **before accessing production return outcomes**.

The threshold was not relaxed after observing the data.

This prevents the research process from turning into:

```text
data too sparse
→ lower threshold
→ inspect returns
→ lower threshold again
→ eventually obtain a backtest
```

The resulting status is:

```text
INFEASIBLE_DATA
```

not:

```text
NO_ALPHA
```

The project did not obtain enough graph coverage to run the pre-registered production alpha test, so it does **not** claim that supply-chain information has no predictive value.

---

## Diagnosing the Bottleneck

The project used statistical diagnostics including:

- Wilson confidence intervals;
- Kendall–Theil–Sen trend analysis;
- Holm multiple-testing correction.

The pipeline achieved:

- **99.92% PDF extraction success**
- **21.65% explicit counterparty naming**
- **1.86% explicit-name-to-listed-entity resolution**

The result suggests that the main constraint was downstream of PDF extraction.

The largest losses in usable graph information occurred because:

1. many disclosures did not explicitly identify counterparties;
2. disclosed entities were not always listed companies;
3. entity names could not always be mapped to a historically valid tradable security;
4. public disclosure coverage was insufficient to create a dense longitudinal graph.

This conclusion applies to the dataset and research design used here; it should not be interpreted as a universal statement about all commercial supply-chain datasets.

---

## Implemented Research Components

Although the real-data feasibility gate prevented production alpha testing, the downstream research infrastructure was implemented and exercised through deterministic fixtures.

Components include:

### Graph research

- directed `supplier → customer` graph construction;
- historical graph snapshots;
- customer-shock propagation;
- graph-neighbor aggregation;
- factor residualization.

### Statistical research

- hypothesis-testing infrastructure;
- bootstrap analysis;
- placebo tests;
- experiment registration;
- holdout locking.

### Execution and portfolio simulation

The execution engine supports constraints including:

- trading suspensions;
- limit-up / limit-down states;
- T+1 execution rules;
- liquidity filtering;
- transaction costs;
- portfolio P&L accounting.

Synthetic / fixture outputs are used only for engineering validation and are never reported as empirical alpha results.

---

## Testing

The project includes **442 unit, integration, and regression tests**.

Coverage includes:

- future-data invariance;
- time-travel / information-leakage prevention;
- disclosure publication timing;
- entity-validity intervals;
- anonymous-entity blocking;
- graph-edge validity;
- market execution rules;
- portfolio accounting;
- stage-gate behavior;
- research-boundary enforcement.

The objective is not only to test whether the software runs, but also whether it is capable of producing an invalid research result silently.

---

## Repository Structure

```text
Supply-Chain-Alpha/
├── config/                     # Research and pipeline configuration
├── data/                       # Canonical metadata / permitted repository artifacts
├── docs/
│   └── raw_data/              # Data-source documentation
├── fixtures/                  # Deterministic test fixtures
├── notebooks/                 # Exploratory / diagnostic analysis
├── reports/                   # Research diagnostics and final summaries
├── scripts/                   # Pipeline entry points
├── src/
│   └── supply_chain_alpha/    # Core package
├── tests/                     # Unit, integration, and regression tests
├── PHASE_1_3_STATUS.md        # Gated research status
├── pyproject.toml
└── README.md
```

---

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -e ".[dev]"
pytest
```

Run the mandatory tiny end-to-end preflight:

```bash
python -m pytest -q tests/integration/test_tiny_e2e_pipeline.py
```

Run the complete gated pipeline:

```bash
python scripts/run_pipeline.py
```

Stop at the real-data feasibility decision:

```bash
python scripts/run_pipeline.py --through-phase 3
```

Validate the resulting project state:

```bash
python scripts/validate_all.py
```

If engineering integrity checks pass but the frozen real-data coverage gate fails, the production pipeline intentionally exits through the valid research-gate path.

That outcome represents an infeasible empirical dataset under the frozen specification—not an engineering crash.

---

## What This Project Demonstrates

The main output of the project is not a trading strategy.

It demonstrates a research process for deciding **whether an alpha hypothesis deserves to be backtested at all**.

The project combines:

- large-scale document processing;
- alternative-data construction;
- point-in-time entity resolution;
- graph analytics;
- reproducible research engineering;
- anti-leakage controls;
- pre-registered decision rules;
- statistical data-quality diagnosis;
- execution-aware quantitative research.

The final result is intentionally a negative one:

> Public annual-report supply-chain disclosures, under this extraction and entity-resolution design, did not provide sufficient longitudinal listed-company graph coverage to justify the planned production pair-trading test.

Rather than weakening the research gate until a backtest became possible, the project records that limitation explicitly and identifies the data improvements required for a future go / no-go decision.

---

## Possible Next Steps

The research could be reopened if a future dataset materially improves graph coverage.

Potential extensions include:

- higher-quality commercial supply-chain datasets;
- additional legally available disclosure sources;
- improved economic-entity-to-security mapping;
- broader relationship types;
- supplier / customer disclosure from multiple jurisdictions;
- alternative graph-based information-diffusion hypotheses that require less dense pair coverage.

Any extension should retain the same point-in-time and pre-registration discipline.

---

## Research Status

```text
Data pipeline:        COMPLETE
Entity resolution:    COMPLETE
PIT graph:            COMPLETE
Coverage diagnostics: COMPLETE
Feasibility gate:     FAIL
Production alpha test: NOT EVALUATED

Final status: INFEASIBLE_DATA
```

This repository should therefore be read as a **completed feasibility study and research-engineering prototype**, not as evidence of a profitable trading strategy.