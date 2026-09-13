# Supply-chain graph feasibility result

Research conclusion: `INFEASIBLE_DATA`

The point-in-time integrity checks and frozen real-data coverage gate were run without using returns.
The disclosed listed-to-listed graph did not meet every pre-specified V2 coverage requirement, so no
return, signal, IC, portfolio, or holdout research was performed.

All mandatory graph-integrity checks passed. This conclusion is caused only by frozen coverage
thresholds and is not an engineering-failure classification.

## Coverage summary

```json
{
  "mentions": 171041,
  "explicit_name_mentions": 37030,
  "explicit_name_rate": 0.2164977987733935,
  "resolved_mentions": 690,
  "resolution_rate_given_explicit": 0.018633540372670808,
  "listed_to_listed_edges": 386,
  "unique_connected_stocks": 487,
  "first_publication": "2015-01-15T16:00:00+00:00",
  "last_publication": "2022-12-30T16:00:00+00:00",
  "supplier_nodes": 276,
  "customer_nodes": 278,
  "source_documents": 560
}
```

## Frozen annual coverage

| Year | Active directed edges at year-end | Connected listed stocks at year-end |
|---:|---:|---:|
| 2016 | 90 | 143 |
| 2017 | 74 | 117 |
| 2018 | 90 | 146 |
| 2019 | 86 | 136 |
| 2020 | 76 | 122 |
| 2021 | 62 | 105 |
| 2022 | 116 | 179 |

## Gate checks

```json
{
  "consecutive_usable_years": {
    "actual": 0,
    "threshold": 3,
    "operator": ">=",
    "passed": false
  },
  "usable_years_median_connected_stocks": {
    "actual": null,
    "threshold": 200,
    "operator": ">=",
    "passed": false
  },
  "pit_industry_count": {
    "actual": null,
    "threshold": 5,
    "operator": ">=",
    "passed": null,
    "deferred": true,
    "reason": "PIT_INDUSTRY_UNAVAILABLE"
  },
  "pit_largest_industry_share": {
    "actual": null,
    "threshold": 0.6,
    "operator": "<=",
    "passed": null,
    "deferred": true,
    "reason": "PIT_INDUSTRY_UNAVAILABLE"
  },
  "timing_violations": {
    "actual": 0,
    "threshold": 0,
    "operator": "<=",
    "passed": true
  },
  "edge_max_age_violations": {
    "actual": 0,
    "threshold": 0,
    "operator": "==",
    "passed": true
  },
  "invalid_counterparty_mentions": {
    "actual": 0,
    "threshold": 0,
    "operator": "==",
    "passed": true
  },
  "invalid_resolution_audit_names": {
    "actual": 0,
    "threshold": 0,
    "operator": "==",
    "passed": true
  },
  "resolution_status_classification_mismatches": {
    "actual": 0,
    "threshold": 0,
    "operator": "==",
    "passed": true
  },
  "anonymous_resolved_nodes": {
    "actual": 0,
    "threshold": 0,
    "operator": "<=",
    "passed": true
  },
  "duplicate_pair_date_exposures": {
    "actual": 0,
    "threshold": 0,
    "operator": "<=",
    "passed": true
  },
  "missing_source_provenance_on_active_edges": {
    "actual": 0,
    "threshold": 0,
    "operator": "<=",
    "passed": true
  },
  "missing_security_master_endpoints": {
    "actual": 0,
    "threshold": 0,
    "operator": "==",
    "passed": true
  },
  "missing_listing_dates_on_edge_endpoints": {
    "actual": 0,
    "threshold": 0,
    "operator": "==",
    "passed": true
  },
  "pit_industry_unclassified_active_endpoints": {
    "actual": null,
    "threshold": 0,
    "operator": "==",
    "passed": null,
    "deferred": true,
    "reason": "PIT_INDUSTRY_UNAVAILABLE"
  },
  "pit_industry_ambiguous_active_nodes": {
    "actual": null,
    "threshold": 0,
    "operator": "==",
    "passed": null,
    "deferred": true,
    "reason": "PIT_INDUSTRY_UNAVAILABLE"
  }
}
```

This is a data-feasibility conclusion, not evidence for or against the economic hypothesis.
