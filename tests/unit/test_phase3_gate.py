import json

import pandas as pd
import pytest

from supply_chain_alpha.graph.coverage import (
    evaluate_phase3_gate,
    node_role_diagnostics,
    source_document_diagnostics,
    yearly_diagnostics,
)
from supply_chain_alpha.graph.snapshots import graph_snapshot


def _frozen_thresholds(**overrides):
    thresholds = {
        "evaluation_start": "2016-01-01",
        "evaluation_end": "2022-12-31",
        "min_consecutive_calendar_years": 3,
        "min_year_end_active_directed_edges": 200,
        "min_year_end_connected_stocks": 150,
        "min_median_connected_stocks_across_usable_years": 200,
        "min_industries_if_pit_available": 5,
        "max_largest_industry_edge_share_if_pit_available": 0.60,
        "max_timing_violations": 0,
        "max_anonymous_resolved_nodes": 0,
        "max_duplicate_pair_date_exposures": 0,
        "max_missing_source_provenance": 0,
    }
    thresholds.update(overrides)
    return thresholds


def _phase3_inputs(*, nodes: int = 200, edge_count: int = 200):
    security_ids = [f"S{index:03d}" for index in range(nodes)]
    master = pd.DataFrame(
        {
            "security_id": security_ids,
            "ticker": [f"{index:06d}" for index in range(nodes)],
            "exchange": ["SSE"] * nodes,
            "company_name": [f"Company {index}" for index in range(nodes)],
            "listing_date": ["2010-01-01"] * nodes,
            "delisting_date": [None] * nodes,
            "board": ["MAIN"] * nodes,
        }
    )

    pairs = [(index, (index + 1) % nodes) for index in range(nodes)]
    offset = 2
    while len(pairs) < edge_count:
        for index in range(nodes):
            pair = (index, (index + offset) % nodes)
            if pair not in pairs:
                pairs.append(pair)
                if len(pairs) == edge_count:
                    break
        offset += 1
    edge_rows = []
    for year in (2018, 2019, 2020):
        publication = pd.Timestamp(f"{year}-04-30 18:00:00", tz="Asia/Shanghai")
        for row, (supplier, customer) in enumerate(pairs):
            edge_rows.append(
                {
                    "supplier_id": security_ids[supplier],
                    "customer_id": security_ids[customer],
                    "effective_start": publication,
                    "effective_end": publication + pd.Timedelta(days=550),
                    "publication_datetime": publication,
                    "source_document_id": f"D{year}-{row:03d}",
                    "source_period_end": f"{year - 1}-12-31",
                    "source_company_id": security_ids[supplier],
                    "relationship_confidence": 1.0,
                    "economic_weight": 1.0,
                    "weight_source": "equal",
                }
            )
    edges = pd.DataFrame(edge_rows)
    industry = pd.DataFrame(
        {
            "security_id": security_ids,
            "industry_code": [f"I{index % 5}" for index in range(nodes)],
            "industry_name": [f"Industry {index % 5}" for index in range(nodes)],
            "industry_valid_from": ["2010-01-01"] * nodes,
            "industry_valid_to": ["2030-01-01"] * nodes,
        }
    )
    return edges, master, industry


def test_phase3_gate_passes_exact_frozen_coverage_minimums():
    edges, master, industry = _phase3_inputs()

    result = evaluate_phase3_gate(
        edges,
        master,
        pit_industry=industry,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["status"] == "PASS"
    assert result["outcome"] == "PROCEED"
    assert result["integrity_passed"] is True
    assert result["coverage_passed"] is True
    assert result["usable_years"] == [2018, 2019, 2020]
    assert result["checks"]["consecutive_usable_years"]["actual"] == 3
    assert result["checks"]["usable_years_median_connected_stocks"]["actual"] == 200.0
    assert result["checks"]["pit_industry_count"]["actual"] == 5
    assert result["checks"]["pit_largest_industry_share"]["actual"] == 0.2
    roles = result["diagnostics"]["latest_usable_year_node_roles"]
    assert roles["supplier_node_count"] == 200
    assert roles["customer_node_count"] == 200
    assert result["diagnostics"]["development_validation_node_roles"] == roles
    assert len(result["diagnostics"]["source_documents"]) == 600
    json.dumps(result)


def test_phase3_gate_defaults_to_blocked_without_explicit_real_data():
    edges, master, industry = _phase3_inputs()

    result = evaluate_phase3_gate(
        edges,
        master,
        pit_industry=industry,
    )

    assert result["status"] == "BLOCKED_NO_REAL_DISCLOSURE_DATA"
    assert result["outcome"] == "BLOCKED"
    assert result["failure_class"] == "NO_REAL_DATA"
    assert result["passed"] is False
    assert result["integrity_passed"] is True
    assert result["coverage_passed"] is True


def test_real_data_gate_rejects_implicit_compatibility_thresholds():
    edges, master, _ = _phase3_inputs()

    with pytest.raises(ValueError, match="requires explicit frozen_thresholds"):
        evaluate_phase3_gate(edges, master, real_data=True)


def test_phase3_gate_requires_median_200_connected_across_usable_years():
    edges, master, _ = _phase3_inputs(nodes=151, edge_count=200)

    result = evaluate_phase3_gate(
        edges,
        master,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["checks"]["consecutive_usable_years"]["passed"] is True
    assert result["checks"]["usable_years_median_connected_stocks"] == {
        "actual": 151.0,
        "threshold": 200,
        "operator": ">=",
        "passed": False,
    }
    assert result["status"] == "FAIL"
    assert result["outcome"] == "INFEASIBLE_DATA"
    assert result["failure_class"] == "COVERAGE_INFEASIBLE"
    assert result["integrity_passed"] is True
    assert result["coverage_passed"] is False


def test_phase3_gate_defers_pit_industry_only_when_history_is_unavailable():
    edges, master, _ = _phase3_inputs()

    result = evaluate_phase3_gate(
        edges,
        master,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["status"] == "PASS"
    assert result["checks"]["pit_industry_count"]["passed"] is None
    assert result["checks"]["pit_largest_industry_share"]["passed"] is None


def test_phase3_gate_enforces_all_zero_tolerance_integrity_checks():
    edges, master, _ = _phase3_inputs()
    snapshot = graph_snapshot(edges, "2020-12-31 23:59:59+08:00")
    edges.loc[0, "effective_start"] = edges.loc[
        0, "publication_datetime"
    ] - pd.Timedelta(days=1)
    edges.loc[1, "source_document_id"] = ""
    audit = pd.DataFrame(
        {
            "counterparty_raw_name": ["Customer A", "*****"],
            "resolution_status": ["resolved", "resolved"],
            "resolved_entity_id": ["S000", "S001"],
        }
    )
    duplicate_snapshot = pd.concat([snapshot, snapshot.iloc[[2]]], ignore_index=True)

    result = evaluate_phase3_gate(
        edges,
        master,
        audit=audit,
        frozen_thresholds=_frozen_thresholds(),
        graph_exposures={"2020-12-31": duplicate_snapshot},
        real_data=True,
    )

    assert result["checks"]["timing_violations"]["actual"] == 1
    assert result["checks"]["anonymous_resolved_nodes"]["actual"] == 2
    assert result["checks"]["duplicate_pair_date_exposures"]["actual"] == 1
    assert result["checks"]["missing_source_provenance_on_active_edges"]["actual"] == 1
    assert result["status"] == "FAIL"
    assert result["outcome"] == "ENGINEERING_FAILURE"
    assert result["failure_class"] == "ENGINEERING_INTEGRITY"
    assert result["integrity_passed"] is False


def test_duplicate_evidence_is_reported_but_collapsed_before_graph_exposure():
    edges, master, _ = _phase3_inputs()
    duplicate_evidence = edges.iloc[[0]].copy()
    duplicate_evidence["source_document_id"] = "D-SECOND-SOURCE"
    combined = pd.concat([edges, duplicate_evidence], ignore_index=True)

    result = evaluate_phase3_gate(
        combined,
        master,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["integrity"]["raw_duplicate_pair_date_evidence_rows"] == 1
    assert result["checks"]["duplicate_pair_date_exposures"]["actual"] == 0
    assert result["status"] == "PASS"


def test_source_provenance_requires_a_nonblank_raw_document_location():
    edges, master, _ = _phase3_inputs()
    mentions = pd.DataFrame(
        {
            "source_document_id": edges["source_document_id"],
            "counterparty_raw_name": [
                f"Named Company {index}" for index in range(len(edges))
            ],
            "source_document_url_or_path": [
                f"raw://D{index:03d}" for index in range(len(edges))
            ],
        }
    )
    mentions.loc[0, "source_document_url_or_path"] = ""

    result = evaluate_phase3_gate(
        edges,
        master,
        mentions=mentions,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    provenance = result["checks"]["missing_source_provenance_on_active_edges"]
    assert provenance["actual"] == 1
    assert provenance["passed"] is False


def test_phase3_gate_checks_pit_industry_breadth_and_concentration():
    edges, master, industry = _phase3_inputs()
    industry["industry_code"] = [
        "DOMINANT" if index < 121 else f"I{index % 4}" for index in range(200)
    ]

    result = evaluate_phase3_gate(
        edges,
        master,
        pit_industry=industry,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["checks"]["pit_industry_count"]["passed"] is True
    assert result["checks"]["pit_largest_industry_share"]["actual"] == 0.605
    assert result["checks"]["pit_largest_industry_share"]["passed"] is False

    industry.loc[120, "industry_code"] = "I0"
    boundary = evaluate_phase3_gate(
        edges,
        master,
        pit_industry=industry,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )
    assert boundary["checks"]["pit_largest_industry_share"]["actual"] == 0.6
    assert boundary["checks"]["pit_largest_industry_share"]["passed"] is True


def test_yearly_and_document_diagnostics_include_required_roles_and_sources():
    edges, master, _ = _phase3_inputs()
    yearly = yearly_diagnostics(edges, master, start_year=2017, end_year=2021)
    roles = node_role_diagnostics(edges)
    documents = source_document_diagnostics(edges)

    assert yearly["year"].tolist() == [2017, 2018, 2019, 2020, 2021]
    assert yearly.loc[yearly["year"].eq(2017), "year_end_active_edges"].item() == 0
    assert yearly.loc[yearly["year"].eq(2018), "supplier_node_count"].item() == 200
    assert roles["is_supplier_node"].sum() == 200
    assert roles["is_customer_node"].sum() == 200
    assert documents["source_document_id"].nunique() == 600
    assert documents["unique_directed_edges"].eq(1).all()


def test_nondefault_frozen_coverage_threshold_is_the_only_gate_truth():
    edges, master, _ = _phase3_inputs()
    thresholds = _frozen_thresholds(min_year_end_active_directed_edges=201)

    result = evaluate_phase3_gate(
        edges,
        master,
        frozen_thresholds=thresholds,
        real_data=True,
    )

    assert result["frozen_thresholds"]["min_year_end_active_directed_edges"] == 201
    assert result["checks"]["consecutive_usable_years"]["actual"] == 0
    assert result["coverage_passed"] is False
    assert result["outcome"] == "INFEASIBLE_DATA"


def test_nondefault_integrity_tolerance_drives_the_check():
    edges, master, _ = _phase3_inputs()
    edges.loc[0, "publication_datetime"] = edges.loc[
        0, "publication_datetime"
    ] + pd.Timedelta(hours=1)
    thresholds = _frozen_thresholds(max_timing_violations=1)

    result = evaluate_phase3_gate(
        edges,
        master,
        frozen_thresholds=thresholds,
        real_data=True,
    )

    assert result["checks"]["timing_violations"] == {
        "actual": 1,
        "threshold": 1,
        "operator": "<=",
        "passed": True,
    }
    assert result["outcome"] == "PROCEED"


def test_explicit_frozen_threshold_mapping_requires_every_config_key():
    edges, master, _ = _phase3_inputs()
    thresholds = _frozen_thresholds()
    thresholds.pop("min_year_end_connected_stocks")

    with pytest.raises(ValueError, match="missing frozen thresholds"):
        evaluate_phase3_gate(
            edges,
            master,
            frozen_thresholds=thresholds,
            real_data=True,
        )


def test_edge_older_than_configured_max_is_an_engineering_failure():
    edges, master, _ = _phase3_inputs()
    edges.loc[0, "effective_end"] = edges.loc[0, "effective_end"] + pd.Timedelta(days=1)

    result = evaluate_phase3_gate(
        edges,
        master,
        frozen_thresholds=_frozen_thresholds(),
        max_edge_age_days=550,
        real_data=True,
    )

    assert result["checks"]["edge_max_age_violations"]["actual"] == 1
    assert result["integrity_passed"] is False
    assert result["outcome"] == "ENGINEERING_FAILURE"


@pytest.mark.parametrize(
    ("fault", "check_name"),
    [
        ("missing_master", "missing_security_master_endpoints"),
        ("missing_listing", "missing_listing_dates_on_edge_endpoints"),
    ],
)
def test_edge_endpoints_require_master_rows_and_listing_dates(fault, check_name):
    edges, master, _ = _phase3_inputs()
    if fault == "missing_master":
        master = master.loc[master["security_id"].ne("S000")].copy()
    else:
        master.loc[master["security_id"].eq("S000"), "listing_date"] = None

    result = evaluate_phase3_gate(
        edges,
        master,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["checks"][check_name]["actual"] == 1
    assert result["integrity_passed"] is False
    assert result["outcome"] == "ENGINEERING_FAILURE"


def test_entity_ids_are_validated_by_master_membership_not_name_semantics() -> None:
    edges, master, _ = _phase3_inputs()
    opaque_entity_id = "Customer A"
    for endpoint in ("supplier_id", "customer_id"):
        edges.loc[edges[endpoint].eq("S000"), endpoint] = opaque_entity_id
    master.loc[master["security_id"].eq("S000"), "security_id"] = opaque_entity_id

    result = evaluate_phase3_gate(
        edges,
        master,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["checks"]["missing_security_master_endpoints"]["actual"] == 0
    assert result["checks"]["anonymous_resolved_nodes"]["actual"] == 0
    assert result["status"] == "PASS"


def test_phase3_name_boundary_rejects_invalid_and_inconsistent_audit_rows() -> None:
    edges, master, _ = _phase3_inputs()
    mentions = pd.DataFrame(
        {
            "source_document_id": edges["source_document_id"],
            "counterparty_raw_name": [
                f"Named Company {index}" for index in range(len(edges))
            ],
            "source_document_url_or_path": [
                f"raw://D{index:03d}" for index in range(len(edges))
            ],
        }
    )
    mentions.loc[0, "counterparty_raw_name"] = "不适用"
    audit = pd.DataFrame(
        {
            "counterparty_raw_name": [
                "不适用",
                "华东企业",
                "供应商A",
                "华南企业",
            ],
            "normalized_name": [None, "华东企业", None, "华南企业"],
            "resolution_status": [
                "unresolved",
                "anonymous",
                "anonymous",
                "resolved ",
            ],
            "resolved_entity_id": [None, None, "S001", "S002"],
            "resolution_method": [
                "none",
                "blocked_anonymous",
                "blocked_anonymous",
                "exact_alias",
            ],
        }
    )

    result = evaluate_phase3_gate(
        edges,
        master,
        mentions=mentions,
        audit=audit,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    assert result["checks"]["invalid_counterparty_mentions"]["actual"] == 1
    assert result["checks"]["invalid_resolution_audit_names"]["actual"] == 1
    assert (
        result["checks"]["resolution_status_classification_mismatches"]["actual"] == 3
    )
    assert result["checks"]["anonymous_resolved_nodes"]["actual"] == 1
    assert result["outcome"] == "ENGINEERING_FAILURE"


def test_available_pit_industry_history_must_classify_every_active_endpoint():
    edges, master, industry = _phase3_inputs()
    industry.loc[industry["security_id"].eq("S000"), "industry_code"] = pd.NA

    result = evaluate_phase3_gate(
        edges,
        master,
        pit_industry=industry,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    completeness = result["checks"]["pit_industry_unclassified_active_endpoints"]
    assert completeness["actual"] > 0
    assert completeness["passed"] is False
    assert result["outcome"] == "ENGINEERING_FAILURE"


def test_overlapping_pit_industry_labels_are_an_engineering_failure():
    edges, master, industry = _phase3_inputs()
    conflicting = industry.loc[industry["security_id"].eq("S000")].copy()
    conflicting["industry_code"] = "CONFLICT"
    industry = pd.concat([industry, conflicting], ignore_index=True)

    result = evaluate_phase3_gate(
        edges,
        master,
        pit_industry=industry,
        frozen_thresholds=_frozen_thresholds(),
        real_data=True,
    )

    ambiguity = result["checks"]["pit_industry_ambiguous_active_nodes"]
    assert ambiguity["actual"] > 0
    assert ambiguity["passed"] is False
    assert result["outcome"] == "ENGINEERING_FAILURE"
