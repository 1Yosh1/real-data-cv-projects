"""
ETL Pipeline Module.

Loads transaction ledger and FX rates into DuckDB, performs currency conversion
joins, data quality checks, and produces an enriched output with a Dead Letter
Queue (DLQ) for unmatched/rejected records.
"""

from __future__ import annotations

import time
from typing import Any

import duckdb
import pandas as pd


def run_etl(
    ledger_df: pd.DataFrame,
    fx_rates_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the full ETL pipeline using DuckDB for performant SQL joins.

    Steps
    -----
    1. Load ledger and FX rates into DuckDB in-memory tables.
    2. Join transactions to FX rates by ``invoice_date`` → ``date`` and
       ``currency`` → ``currency_code``.
    3. Convert amounts to EUR and USD equivalents.
    4. Route unmatched records to a Dead Letter Queue.
    5. Compute reconciliation statistics.

    Parameters
    ----------
    ledger_df : pd.DataFrame
        Raw transaction ledger.
    fx_rates_df : pd.DataFrame
        Tidy, forward-filled FX rates.

    Returns
    -------
    enriched_df : pd.DataFrame
        Successfully matched & converted transactions.
    dlq_df : pd.DataFrame
        Records that could not be matched to an FX rate.
    stats : dict
        Reconciliation and summary statistics.
    """
    t_start = time.perf_counter()
    con = duckdb.connect(database=":memory:")

    # ------------------------------------------------------------------
    # 1. Register source DataFrames as DuckDB tables
    # ------------------------------------------------------------------
    con.register("ledger_raw", ledger_df)
    con.register("fx_rates_raw", fx_rates_df)

    con.execute("""
        CREATE TABLE ledger AS SELECT * FROM ledger_raw
    """)
    con.execute("""
        CREATE TABLE fx_rates AS SELECT * FROM fx_rates_raw
    """)

    # ------------------------------------------------------------------
    # 2. Determine unique currencies in ledger
    # ------------------------------------------------------------------
    total_records = int(con.execute("SELECT COUNT(*) FROM ledger").fetchone()[0])
    currencies_in_ledger = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT currency FROM ledger"
        ).fetchall()
    ]

    # ------------------------------------------------------------------
    # 3. Join ledger to FX rates & compute EUR / USD amounts
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE enriched AS
        SELECT
            l.transaction_id,
            l.invoice_date,
            l.vendor_country,
            l.currency,
            l.base_amount,
            l.expense_category,
            l.vendor_name,
            l.payment_terms,
            l.department,
            fx.rate_to_eur  AS fx_rate_used,
            l.invoice_date  AS conversion_date,
            -- EUR conversion: for EUR transactions, amount stays the same;
            -- otherwise divide by the rate (rate_to_eur = X currency per 1 EUR)
            CASE
                WHEN l.currency = 'EUR' THEN l.base_amount
                ELSE l.base_amount / fx.rate_to_eur
            END AS amount_eur,
            -- USD conversion: first to EUR then multiply by USD rate
            CASE
                WHEN l.currency = 'EUR' THEN l.base_amount * usd_fx.rate_to_eur
                ELSE (l.base_amount / fx.rate_to_eur) * usd_fx.rate_to_eur
            END AS amount_usd
        FROM ledger l
        LEFT JOIN fx_rates fx
            ON l.invoice_date = fx.date
            AND l.currency = fx.currency_code
        LEFT JOIN fx_rates usd_fx
            ON l.invoice_date = usd_fx.date
            AND usd_fx.currency_code = 'USD'
        WHERE l.currency = 'EUR'
           OR fx.rate_to_eur IS NOT NULL
    """)

    # ------------------------------------------------------------------
    # 4. Dead Letter Queue — records that didn't match
    # ------------------------------------------------------------------
    con.execute("""
        CREATE TABLE dlq AS
        SELECT
            l.transaction_id,
            l.invoice_date,
            l.vendor_country,
            l.currency,
            l.base_amount,
            l.expense_category,
            l.vendor_name,
            l.payment_terms,
            l.department,
            'MISSING_FX_RATE' AS rejection_reason
        FROM ledger l
        LEFT JOIN fx_rates fx
            ON l.invoice_date = fx.date
            AND l.currency = fx.currency_code
        WHERE l.currency != 'EUR'
          AND fx.rate_to_eur IS NULL
    """)

    # ------------------------------------------------------------------
    # 5. Fetch results
    # ------------------------------------------------------------------
    enriched_df = con.execute("SELECT * FROM enriched").fetchdf()
    dlq_df = con.execute("SELECT * FROM dlq").fetchdf()

    matched_records = len(enriched_df)
    dlq_records = len(dlq_df)
    reconciliation_rate = (matched_records / total_records) * 100 if total_records > 0 else 0.0

    t_elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # 6. Statistics
    # ------------------------------------------------------------------
    stats: dict[str, Any] = {
        "total_records": total_records,
        "matched_records": matched_records,
        "dlq_records": dlq_records,
        "reconciliation_rate": round(reconciliation_rate, 4),
        "currencies_processed": sorted(currencies_in_ledger),
        "execution_time_seconds": round(t_elapsed, 3),
        "total_eur_exposure": round(enriched_df["amount_eur"].sum(), 2) if len(enriched_df) > 0 else 0.0,
        "total_usd_exposure": round(enriched_df["amount_usd"].sum(), 2) if len(enriched_df) > 0 else 0.0,
    }

    # Assert reconciliation quality gate
    if reconciliation_rate < 99.8:
        import warnings
        warnings.warn(
            f"Reconciliation rate {reconciliation_rate:.2f}% is below the 99.8% threshold. "
            f"DLQ contains {dlq_records} records."
        )

    con.close()
    return enriched_df, dlq_df, stats


def get_etl_summary(
    enriched_df: pd.DataFrame,
    dlq_df: pd.DataFrame,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Build a human-readable summary dictionary from ETL outputs.

    Parameters
    ----------
    enriched_df : pd.DataFrame
        Successfully enriched records.
    dlq_df : pd.DataFrame
        Dead Letter Queue records.
    stats : dict
        Raw reconciliation statistics from ``run_etl``.

    Returns
    -------
    dict
        Summary suitable for rendering in a dashboard or report.
    """
    currency_exposure = (
        enriched_df.groupby("currency")["amount_eur"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    ) if len(enriched_df) > 0 else {}

    department_totals = (
        enriched_df.groupby("department")["amount_eur"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    ) if len(enriched_df) > 0 else {}

    return {
        "etl_stats": stats,
        "currency_exposure_eur": currency_exposure,
        "department_totals_eur": department_totals,
        "dlq_sample": dlq_df.head(5).to_dict(orient="records") if len(dlq_df) > 0 else [],
        "date_range": {
            "min": str(enriched_df["invoice_date"].min()) if len(enriched_df) > 0 else None,
            "max": str(enriched_df["invoice_date"].max()) if len(enriched_df) > 0 else None,
        },
    }
