# Supply-Chain Information Diffusion Alpha in Chinese Equities

Implementation follows `PROJECT_SPEC.md`. The repository enforces the Phase 3
real-data coverage gate before any return, signal, portfolio, or holdout work.

## Implemented

- repository/config and canonical schema validation;
- raw-file SHA-256 manifest convention;
- structured disclosure ingestion;
- conservative counterparty candidate extraction;
- deterministic entity normalization/resolution with anonymous-name blocking;
- manual override audit path;
- point-in-time directed edge construction (`supplier -> customer`);
- graph snapshots that cannot predate publication;
- coverage, yearly, degree, industry diagnostics;
- Phase-3 gate status;
- deterministic in-memory market, residual, diffusion-signal, IC/bootstrap/placebo,
  portfolio execution, cost, P&L, and holdout-lock primitives;
- the mandatory 12-security / 160-trading-day tiny end-to-end preflight;
- automated unit/integration/regression tests.

## Intentionally not implemented

Production Phase 4+ artifacts and empirical claims remain blocked until **real-data
Phase 3 coverage passes**. The market-through-P&L implementation is exercised only
by the deterministic test fixture before that gate; synthetic outputs never enter
canonical production tables, statuses, or reports.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest
```

Run the mandatory tiny full-chain preflight by itself:

```bash
python -m pytest -q tests/integration/test_tiny_e2e_pipeline.py
```

## Production workflow

Run the complete gated pipeline from the repository root:

```bash
python scripts/run_pipeline.py
```

To stop at the real-data feasibility decision:

```bash
python scripts/run_pipeline.py --through-phase 3
```

Validate the resulting project state independently:

```bash
python scripts/validate_all.py
```

Production Phase 3 reads only the canonical Phase 2 artifacts and is the default;
there is no `--real-data` option. `--synthetic` is diagnostic-only, writes only to
the trial area, and can never create a production phase status.

If the engineering integrity checks pass but the frozen real-data coverage checks
fail, `run_pipeline.py` exits with code `3`. That is the specification's valid
Path B research-gate outcome, not an engineering crash. In that state,
`validate_all.py` must still exit with code `0` and report
`INFEASIBLE_DATA` / `NOT_EVALUATED` after verifying every required artifact.

## Canonical Phase 3 inputs

Phase 3 requires the Phase 2 `PASS` status and these validated canonical tables:

1. `security_master`: point-in-time A-share listing master;
2. `company_alias`: canonical legal names and validity-dated aliases;
3. `disclosure_raw`: source-grounded customer/supplier mentions with actual
   publication timestamps;
4. `resolution_audit`: the frozen Phase 2 decision for every mention key.

Canonical fields are defined in `src/supply_chain_alpha/data/schemas.py`.
