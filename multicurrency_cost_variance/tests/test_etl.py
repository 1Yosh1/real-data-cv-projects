from __future__ import annotations

import pandas as pd
import pytest

from etl.etl_pipeline import run_etl
from etl.fetch_ecb_fx import forward_fill_rates


def test_forward_fill_rates(sample_fx_rates_with_gaps: pd.DataFrame):
    filled = forward_fill_rates(sample_fx_rates_with_gaps)
    assert not filled.empty
    assert "date" in filled.columns
    assert "rate_to_eur" in filled.columns
    assert filled["rate_to_eur"].isna().sum() == 0


def test_etl_reconciliation_clean_data(
    sample_ledger: pd.DataFrame, sample_fx_rates: pd.DataFrame
):
    enriched, dlq, stats = run_etl(sample_ledger, sample_fx_rates)
    assert not enriched.empty
    assert stats["reconciliation_rate"] >= 99.8
    assert len(dlq) == 0


def test_etl_dlq_invalid_currency(
    sample_ledger_with_invalid_currency: pd.DataFrame, sample_fx_rates: pd.DataFrame
):
    enriched, dlq, stats = run_etl(sample_ledger_with_invalid_currency, sample_fx_rates)
    assert len(dlq) == 3
    assert set(dlq["currency"].unique()) == {"XYZ"}
    assert len(enriched) == 10


def test_eur_amount_conversion():
    ledger = pd.DataFrame(
        [
            {
                "transaction_id": "TX-001",
                "invoice_date": pd.to_datetime("2024-01-02"),
                "vendor_country": "USA",
                "currency": "USD",
                "base_amount": 110.0,
                "expense_category": "Aviation Fuel",
                "vendor_name": "FuelCorp",
                "payment_terms": "NET30",
                "department": "Operations",
            }
        ]
    )
    fx_rates = pd.DataFrame(
        [
            {
                "date": pd.to_datetime("2024-01-02"),
                "currency_code": "USD",
                "rate_to_eur": 1.10,
            }
        ]
    )
    enriched, dlq, _ = run_etl(ledger, fx_rates)
    assert len(enriched) == 1
    assert pytest.approx(enriched.iloc[0]["amount_eur"], 0.01) == 100.0
