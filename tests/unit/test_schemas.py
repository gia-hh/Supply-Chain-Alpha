import pandas as pd
import pytest

from supply_chain_alpha.data.schemas import (
    CNINFO_DOCUMENT_AUDIT,
    DISCLOSURE_RAW,
    EQUITY_DAILY,
    RETURNS_DAILY,
    SCHEMAS,
    SECURITY_MASTER,
    SIGNAL_DAILY,
    coerce_disclosure_types,
    validate_table,
)


def _security_row(**overrides):
    row = {
        "security_id": "S1",
        "ticker": "600001",
        "exchange": "SSE",
        "company_name": "A",
        "listing_date": "2010-01-01",
        "delisting_date": None,
        "board": "MAIN",
    }
    row.update(overrides)
    return row


def _disclosure_row(**overrides):
    row = {
        "source_company_id": "S1",
        "counterparty_raw_name": "B",
        "relationship_type": "customer",
        "source_period_end": "2020-12-31",
        "publication_datetime": "2021-04-01T18:00:00+08:00",
        "source_document_id": "D",
        "source_document_url_or_path": "fixture://D",
        "evidence_text": "主要客户：B",
        "exposure_value": None,
        "exposure_share": None,
    }
    row.update(overrides)
    return row


def test_duplicate_primary_key_raises():
    row = _security_row()
    with pytest.raises(ValueError, match="duplicate primary key"):
        validate_table(pd.DataFrame([row, row]), SECURITY_MASTER)


def test_security_industry_fields_are_optional_and_declared():
    validate_table(pd.DataFrame([_security_row()]), SECURITY_MASTER, allow_extra=False)
    assert {
        "industry_code",
        "industry_name",
        "industry_valid_from",
        "industry_valid_to",
    }.issubset(SECURITY_MASTER.optional_columns)


@pytest.mark.parametrize("column", ["security_id", "ticker"])
def test_security_identifiers_must_remain_strings(column):
    row = _security_row(**{column: 600001})
    with pytest.raises(ValueError, match=f"{column} must contain strings"):
        validate_table(pd.DataFrame([row]), SECURITY_MASTER)


def test_security_exchange_and_listing_interval_are_validated():
    with pytest.raises(ValueError, match="invalid exchange"):
        validate_table(pd.DataFrame([_security_row(exchange="NYSE")]), SECURITY_MASTER)
    with pytest.raises(ValueError, match="listing_date must be < delisting_date"):
        validate_table(
            pd.DataFrame(
                [_security_row(listing_date="2020-01-01", delisting_date="2019-01-01")]
            ),
            SECURITY_MASTER,
        )
    with pytest.raises(ValueError, match="ISO YYYY-MM-DD"):
        validate_table(
            pd.DataFrame([_security_row(listing_date="01/02/2020")]), SECURITY_MASTER
        )


def test_disclosure_v2_evidence_is_required():
    row = _disclosure_row()
    del row["evidence_text"]
    with pytest.raises(ValueError, match="missing required columns.*evidence_text"):
        validate_table(pd.DataFrame([row]), DISCLOSURE_RAW)


def test_disclosure_period_date_compares_with_timezone_aware_publication():
    validated = validate_table(pd.DataFrame([_disclosure_row()]), DISCLOSURE_RAW)
    assert len(validated) == 1

    with pytest.raises(
        ValueError, match="source_period_end must be <= publication_datetime"
    ):
        validate_table(
            pd.DataFrame(
                [
                    _disclosure_row(
                        source_period_end="2022-12-31",
                        publication_datetime="2021-04-01T18:00:00+08:00",
                    )
                ]
            ),
            DISCLOSURE_RAW,
        )


def test_exposure_share_must_be_decimal():
    with pytest.raises(ValueError, match="exposure_share must be decimal"):
        coerce_disclosure_types(pd.DataFrame([_disclosure_row(exposure_share=20.0)]))


@pytest.mark.parametrize("column", ["exposure_value", "exposure_share"])
def test_malformed_exposure_fails_loudly(column):
    with pytest.raises(ValueError, match=f"{column} must be numeric"):
        coerce_disclosure_types(pd.DataFrame([_disclosure_row(**{column: "unknown"})]))


def test_v2_daily_schemas_are_registered_with_composite_keys():
    assert SCHEMAS["cninfo_document_audit"] is CNINFO_DOCUMENT_AUDIT
    assert SCHEMAS["equity_daily"] is EQUITY_DAILY
    assert SCHEMAS["returns_daily"] is RETURNS_DAILY
    assert SCHEMAS["signal_daily"] is SIGNAL_DAILY
    assert EQUITY_DAILY.primary_key == ("date", "security_id")
    assert RETURNS_DAILY.primary_key == ("date", "security_id")
    assert SIGNAL_DAILY.primary_key == ("date", "security_id")


def test_cninfo_document_audit_rejects_unknown_reconciliation_method():
    row = {column: None for column in CNINFO_DOCUMENT_AUDIT.required_columns}
    row.update(
        {
            "source_document_id": "D1",
            "source_company_id_reconciliation_method": "guess_from_name",
            "source_company_id_reconciliation_candidate_count": 0,
        }
    )

    with pytest.raises(ValueError, match="invalid security-ID reconciliation method"):
        validate_table(pd.DataFrame([row]), CNINFO_DOCUMENT_AUDIT)


def test_daily_schema_rejects_duplicate_date_security_key():
    row = {
        "date": "2020-01-02",
        "security_id": "S1",
        "open": 10.0,
        "close": 10.1,
        "volume": 1000,
        "amount": 10100.0,
        "tradable": True,
    }
    with pytest.raises(ValueError, match="duplicate primary key"):
        validate_table(pd.DataFrame([row, row]), EQUITY_DAILY)
