# Phase 1–3 Implementation Status

This page describes the implemented control surface. It intentionally does not
copy volatile run counts or conclusions. The authoritative current result is in:

- `reports/status/phase_1.json`
- `reports/status/phase_2.json`
- `reports/status/phase_3.json`
- `reports/phase3_coverage/summary.json`
- `reports/final/results_summary.json` when the Phase-3 gate selects Path B

## Phase 1 — Repository and contracts

- Canonical schemas, primary-key checks, decimal percentage units, Parquet I/O,
  deterministic source/config hashes, and the phase-state protocol are enforced.
- Raw official assets are append-only, provenance sidecars are mandatory, and an
  exact-inventory SHA-256 manifest detects additions, deletions, or mutations.
- The declared package dependencies and runtime versions are checked before the
  empirical pipeline can proceed.
- The explicit Section-19 test registry is fail-closed if any required family is
  missing, and the Section-25 tiny fixture must pass extraction through portfolio
  P&L before production acquisition proceeds. Its outputs remain in memory only.

## Phase 2 — Official data and entity resolution

- SSE, SZSE, and CNINFO public endpoints are acquired through a rate-limited,
  retry-bounded cache. Network targets are restricted to exact official HTTPS
  hosts and endpoint paths; redirects and non-official attachment URLs fail closed.
- Annual-report extraction runs in bounded process isolation and retains document,
  section, evidence, source identity, timestamp, and failure audits.
- CNINFO organization IDs and code-change evidence reconcile historical disclosure
  issuers without guessing across missing or ambiguous candidates.
- Named counterparties use publication-time-valid aliases. Anonymous labels never
  resolve; ambiguous exact matches remain unresolved; fuzzy matches require the
  frozen score and uniqueness margin; manual overrides are version-controlled.
- Date-precision publication timestamps are conservatively delayed before becoming
  usable, while the raw source epoch and timing rule remain auditable.

## Phase 3 — Point-in-time graph gate

- Direction is `supplier -> customer`; edges cannot predate safe publication
  availability and expire after the frozen 550-calendar-day maximum age.
- Listing intervals are intersected at both endpoints, self-edges are excluded, and
  duplicate evidence cannot inflate a graph snapshot.
- The frozen real-data coverage gate is evaluated before any returns, signals,
  portfolio construction, or holdout access.
- If integrity passes but coverage does not, the workflow terminates on the spec's
  Path B with a feasibility report. Phase 4+ remains inaccessible unless Phase 3
  genuinely passes every frozen threshold.

Run `python scripts/validate_all.py` to independently verify the current artifacts,
hashes, statuses, tests, and permitted terminal path.

## Post-run diagnosis and handoff

The frozen Path-B result is supplemented by post-run interpretation artifacts that
do not alter any phase status, threshold, or canonical result:

- `reports/final/INFEASIBILITY_DIAGNOSIS_ZH.md`: statistical diagnosis plus
  macroeconomic and microeconomic interpretation;
- `reports/final/infeasibility_diagnostics.json`: machine-readable statistics,
  methods, limitations, and hashes of every frozen input;
- `reports/final/IMPLEMENTATION_RECORD_ZH.md`: implementation scope, validation,
  provenance, and truthful Git-history notes.

These artifacts use only existing formal outputs. They do not download additional
data or read returns, signals, portfolios, or holdout results.
