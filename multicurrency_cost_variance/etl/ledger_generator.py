"""
Ledger Generator Module.

Generates realistic enterprise procurement transactions for aviation cost
analysis, including vendor details, multi-currency amounts, expense categories,
and payment terms.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Expense categories aligned with the 15 operational cost drivers
# ---------------------------------------------------------------------------
EXPENSE_CATEGORIES: list[str] = [
    "Aviation Fuel",
    "Direct Labor",
    "Aircraft Maintenance",
    "Airport Handling",
    "Air Navigation Fees",
    "IT Infrastructure",
    "Software Licenses",
    "Commercial Insurance",
    "Legal Compliance",
    "Facility Leases",
    "Marketing",
    "Customer Care",
    "Telecom",
    "Admin Staff",
    "Financing Fees",
]

# ---------------------------------------------------------------------------
# Category → mean base_amount (EUR) for log-normal sampling
# ---------------------------------------------------------------------------
CATEGORY_MEAN_AMOUNT: dict[str, float] = {
    "Aviation Fuel": 55_000.0,
    "Direct Labor": 8_500.0,
    "Aircraft Maintenance": 42_000.0,
    "Airport Handling": 12_000.0,
    "Air Navigation Fees": 6_500.0,
    "IT Infrastructure": 18_000.0,
    "Software Licenses": 4_200.0,
    "Commercial Insurance": 35_000.0,
    "Legal Compliance": 9_000.0,
    "Facility Leases": 22_000.0,
    "Marketing": 7_500.0,
    "Customer Care": 3_800.0,
    "Telecom": 2_200.0,
    "Admin Staff": 5_000.0,
    "Financing Fees": 15_000.0,
}

# ---------------------------------------------------------------------------
# Vendor country → currency mapping
# ---------------------------------------------------------------------------
COUNTRY_CURRENCY: dict[str, str] = {
    "Germany": "EUR",
    "France": "EUR",
    "Italy": "EUR",
    "Spain": "EUR",
    "Netherlands": "EUR",
    "Belgium": "EUR",
    "Austria": "EUR",
    "Ireland": "EUR",
    "Finland": "EUR",
    "Portugal": "EUR",
    "United States": "USD",
    "Canada": "CAD",
    "United Kingdom": "GBP",
    "Japan": "JPY",
    "Switzerland": "CHF",
    "Australia": "AUD",
    "China": "CNY",
    "Sweden": "SEK",
    "Norway": "NOK",
}

# Probability weights per country (approx 40% EUR, 25% USD, 15% GBP, etc.)
COUNTRY_WEIGHTS: dict[str, float] = {
    "Germany": 0.10,
    "France": 0.08,
    "Italy": 0.07,
    "Spain": 0.05,
    "Netherlands": 0.04,
    "Belgium": 0.02,
    "Austria": 0.01,
    "Ireland": 0.01,
    "Finland": 0.01,
    "Portugal": 0.01,
    "United States": 0.20,
    "Canada": 0.05,
    "United Kingdom": 0.15,
    "Japan": 0.10,
    "Switzerland": 0.03,
    "Australia": 0.03,
    "China": 0.02,
    "Sweden": 0.01,
    "Norway": 0.01,
}

# ---------------------------------------------------------------------------
# Vendor name patterns
# ---------------------------------------------------------------------------
VENDOR_PREFIXES: list[str] = [
    "Aero", "Sky", "Global", "Trans", "Euro", "Prime", "Atlas",
    "Pinnacle", "Apex", "Zenith", "Delta", "Orbital", "Nordic",
    "Pacific", "Alpine", "Titan", "Vanguard", "Meridian", "Falcon", "Eagle",
]
VENDOR_SUFFIXES: list[str] = [
    "Systems", "Solutions", "Services", "Group", "Corp", "Industries",
    "Technologies", "Logistics", "Partners", "Supply Co", "Holdings",
    "Engineering", "Aviation", "Consulting", "GmbH", "Ltd", "S.A.", "Inc",
]

DEPARTMENTS: list[str] = [
    "Operations", "Engineering", "Finance", "Marketing", "Legal", "IT", "HR",
]

PAYMENT_TERMS: list[str] = ["NET30", "NET60", "NET90", "PREPAID"]
PAYMENT_TERM_WEIGHTS: list[float] = [0.45, 0.30, 0.15, 0.10]


def _generate_vendor_name(rng: np.random.Generator) -> str:
    """Create a realistic company name from prefix + suffix."""
    prefix = rng.choice(VENDOR_PREFIXES)
    suffix = rng.choice(VENDOR_SUFFIXES)
    return f"{prefix} {suffix}"


def generate_ledger(
    n_transactions: int = 100_000,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    output_dir: Optional[str] = "data/",
) -> pd.DataFrame:
    """Generate a realistic procurement transaction ledger.

    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate.
    start_date, end_date : str
        ISO date range for invoice dates.
    seed : int
        Random seed for reproducibility.
    output_dir : str | None
        If not *None*, save the ledger as Parquet in this directory.

    Returns
    -------
    pd.DataFrame
        Ledger with columns: ``transaction_id``, ``invoice_date``,
        ``vendor_country``, ``currency``, ``base_amount``,
        ``expense_category``, ``vendor_name``, ``payment_terms``,
        ``department``.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Invoice dates — weighted toward business days
    # ------------------------------------------------------------------
    date_range = pd.date_range(start_date, end_date, freq="D")
    # Assign higher weight to weekdays (Mon–Fri)
    weights = np.where(date_range.dayofweek < 5, 5.0, 1.0)
    weights = weights / weights.sum()
    invoice_dates = rng.choice(date_range, size=n_transactions, p=weights)

    # ------------------------------------------------------------------
    # 2. Vendor country & currency
    # ------------------------------------------------------------------
    countries = list(COUNTRY_WEIGHTS.keys())
    country_probs = np.array(list(COUNTRY_WEIGHTS.values()))
    country_probs = country_probs / country_probs.sum()  # normalise
    vendor_countries = rng.choice(countries, size=n_transactions, p=country_probs)
    currencies = np.array([COUNTRY_CURRENCY[c] for c in vendor_countries])

    # ------------------------------------------------------------------
    # 3. Expense categories (uniform across 15 drivers)
    # ------------------------------------------------------------------
    expense_categories = rng.choice(EXPENSE_CATEGORIES, size=n_transactions)

    # ------------------------------------------------------------------
    # 4. Base amounts — log-normal, mean depends on category
    # ------------------------------------------------------------------
    base_amounts = np.empty(n_transactions, dtype=float)
    for cat in EXPENSE_CATEGORIES:
        mask = expense_categories == cat
        n_cat = mask.sum()
        if n_cat == 0:
            continue
        mu = np.log(CATEGORY_MEAN_AMOUNT[cat])
        sigma = 0.6  # moderate spread
        base_amounts[mask] = rng.lognormal(mean=mu, sigma=sigma, size=n_cat)
    base_amounts = np.round(base_amounts, 2)

    # ------------------------------------------------------------------
    # 5. Other fields
    # ------------------------------------------------------------------
    vendor_names = np.array([_generate_vendor_name(rng) for _ in range(n_transactions)])
    payment_terms = rng.choice(
        PAYMENT_TERMS, size=n_transactions, p=PAYMENT_TERM_WEIGHTS
    )
    departments = rng.choice(DEPARTMENTS, size=n_transactions)
    transaction_ids = np.array([str(uuid.uuid4()) for _ in range(n_transactions)])

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    ledger = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "invoice_date": pd.to_datetime(invoice_dates),
            "vendor_country": vendor_countries,
            "currency": currencies,
            "base_amount": base_amounts,
            "expense_category": expense_categories,
            "vendor_name": vendor_names,
            "payment_terms": payment_terms,
            "department": departments,
        }
    )
    ledger.sort_values("invoice_date", inplace=True)
    ledger.reset_index(drop=True, inplace=True)

    # Save to Parquet
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ledger.to_parquet(out_path / "ledger.parquet", index=False, engine="pyarrow")

    return ledger
