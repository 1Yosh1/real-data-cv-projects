"""
Shared test fixtures for the multicurrency_cost_variance test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simulation.cost_drivers import CostDriver, get_all_drivers


@pytest.fixture
def sample_fx_rates() -> pd.DataFrame:
    """Provide a small, deterministic FX rates DataFrame for testing.

    Covers 7 days (Mon 2024-01-01 → Sun 2024-01-07) with three currencies.
    Weekdays have real rates; weekend rows are present to test forward-fill.
    """
    dates = pd.date_range("2024-01-01", "2024-01-07", freq="D")
    currencies = ["USD", "GBP", "JPY"]
    rates = {
        "USD": [1.1050, 1.1060, 1.1045, 1.1070, 1.1080, 1.1080, 1.1080],
        "GBP": [0.8620, 0.8610, 0.8625, 0.8630, 0.8640, 0.8640, 0.8640],
        "JPY": [158.50, 158.60, 158.45, 158.70, 158.80, 158.80, 158.80],
    }

    rows: list[dict] = []
    for i, dt in enumerate(dates):
        for ccy in currencies:
            rows.append(
                {"date": dt, "currency_code": ccy, "rate_to_eur": rates[ccy][i]}
            )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_fx_rates_with_gaps() -> pd.DataFrame:
    """FX rates with missing weekend rows — used to test forward-fill logic."""
    # Only weekdays
    dates = pd.bdate_range("2024-01-01", "2024-01-12")
    currencies = ["USD", "GBP"]
    rng = np.random.default_rng(99)

    rows: list[dict] = []
    for dt in dates:
        for ccy in currencies:
            rate = 1.10 + rng.normal(0, 0.005) if ccy == "USD" else 0.86 + rng.normal(0, 0.003)
            rows.append({"date": dt, "currency_code": ccy, "rate_to_eur": round(rate, 4)})

    return pd.DataFrame(rows)


@pytest.fixture
def sample_ledger(sample_fx_rates: pd.DataFrame) -> pd.DataFrame:
    """Small deterministic ledger that should fully match sample_fx_rates."""
    rng = np.random.default_rng(7)
    n = 50
    dates = sample_fx_rates["date"].unique()
    currencies = ["EUR", "USD", "GBP", "JPY"]

    rows: list[dict] = []
    for i in range(n):
        ccy = currencies[i % len(currencies)]
        rows.append(
            {
                "transaction_id": f"TX-{i:04d}",
                "invoice_date": rng.choice(dates),
                "vendor_country": "Test Country",
                "currency": ccy,
                "base_amount": round(rng.uniform(100, 50_000), 2),
                "expense_category": "IT Infrastructure",
                "vendor_name": f"Vendor {i}",
                "payment_terms": "NET30",
                "department": "Engineering",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_ledger_with_invalid_currency(sample_fx_rates: pd.DataFrame) -> pd.DataFrame:
    """Ledger that includes a currency not present in FX rates → should go to DLQ."""
    rng = np.random.default_rng(8)
    dates = sample_fx_rates["date"].unique()

    rows: list[dict] = []
    # 10 valid USD rows
    for i in range(10):
        rows.append(
            {
                "transaction_id": f"TXV-{i:04d}",
                "invoice_date": rng.choice(dates),
                "vendor_country": "US",
                "currency": "USD",
                "base_amount": round(rng.uniform(500, 10_000), 2),
                "expense_category": "Aviation Fuel",
                "vendor_name": f"Valid Vendor {i}",
                "payment_terms": "NET30",
                "department": "Operations",
            }
        )
    # 3 invalid XYZ rows — should end up in DLQ
    for i in range(3):
        rows.append(
            {
                "transaction_id": f"TXI-{i:04d}",
                "invoice_date": rng.choice(dates),
                "vendor_country": "Atlantis",
                "currency": "XYZ",
                "base_amount": round(rng.uniform(500, 10_000), 2),
                "expense_category": "Telecom",
                "vendor_name": f"Invalid Vendor {i}",
                "payment_terms": "PREPAID",
                "department": "IT",
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def sample_drivers() -> list[CostDriver]:
    """Return the canonical list of 15 cost drivers."""
    return get_all_drivers()
