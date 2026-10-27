"""
fetch_fred_data.py
==================
Downloads macroeconomic time-series from the St. Louis Federal Reserve (FRED)
and corporate financial data from Yahoo Finance, merges them into a single
panel DataFrame, and persists the result as a Parquet file.

Series
------
- CPIAUCSL  : Consumer Price Index for All Urban Consumers
- FEDFUNDS  : Effective Federal Funds Rate
- INDPRO    : Industrial Production Index
- DGS10     : 10-Year Treasury Constant Maturity Rate
- M2SL      : M2 Money Stock

Corporate
---------
- DAL       : Delta Air Lines quarterly cash flow from operations

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_SERIES: Dict[str, str] = {
    "CPIAUCSL": "CPI",
    "FEDFUNDS": "FedFunds",
    "INDPRO": "IndustrialProduction",
    "DGS10": "Treasury10Y",
    "M2SL": "M2MoneyStock",
}

DATA_DIR = Path(__file__).resolve().parent
PARQUET_PATH = DATA_DIR / "macro_data.parquet"


# ---------------------------------------------------------------------------
# FRED helpers
# ---------------------------------------------------------------------------

def _fetch_single_series(series_id: str) -> pd.Series:
    """Download a single FRED series CSV and return as a named pd.Series."""
    url = f"{FRED_BASE_URL}?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(
        io.StringIO(response.text),
        parse_dates=["DATE"],
        index_col="DATE",
        na_values=["."],
    )
    col = df.columns[0]
    series = df[col].astype(float)
    series.name = FRED_SERIES.get(series_id, series_id)
    return series


def fetch_fred_data() -> pd.DataFrame:
    """Fetch all FRED series and merge on the DATE index.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame with columns for each macro indicator.
    """
    frames = []
    for series_id in FRED_SERIES:
        print(f"  Fetching FRED series: {series_id} …")
        s = _fetch_single_series(series_id)
        frames.append(s)
    merged = pd.concat(frames, axis=1, join="outer")
    merged.index = pd.DatetimeIndex(merged.index, freq="infer")
    merged = merged.sort_index()
    merged = merged.ffill()
    return merged


# ---------------------------------------------------------------------------
# Corporate data (Delta Air Lines)
# ---------------------------------------------------------------------------

def fetch_dal_cashflow() -> pd.DataFrame:
    """Fetch Delta Air Lines quarterly cash flow from operations via yfinance.

    Returns
    -------
    pd.DataFrame
        Quarterly series resampled to monthly frequency (forward-filled).
    """
    print("  Fetching Delta Air Lines (DAL) cash-flow data …")
    ticker = yf.Ticker("DAL")
    try:
        cf = ticker.quarterly_cashflow
        if cf is None or cf.empty:
            raise ValueError("No cashflow data returned for DAL")
        # Look for operating cash flow row
        possible_labels = [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
        ]
        row_label = None
        for lbl in possible_labels:
            if lbl in cf.index:
                row_label = lbl
                break
        if row_label is None:
            # Fallback: use the first row
            row_label = cf.index[0]

        op_cf = cf.loc[row_label].sort_index()
        op_cf.name = "DAL_OperatingCF"
        op_cf.index = pd.DatetimeIndex(op_cf.index)
        # Resample to monthly, forward-fill
        monthly = op_cf.resample("MS").ffill()
        return monthly.to_frame()
    except Exception as exc:
        print(f"  Warning: Could not fetch DAL data – {exc}")
        return pd.DataFrame(columns=["DAL_OperatingCF"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataset(save: bool = True) -> pd.DataFrame:
    """Full pipeline: fetch, merge, compute pct changes, save.

    Parameters
    ----------
    save : bool
        If True, persist to Parquet.

    Returns
    -------
    pd.DataFrame
        Merged dataset with both level and pct-change columns.
    """
    print("[1/4] Fetching FRED macro data …")
    macro = fetch_fred_data()

    print("[2/4] Fetching corporate data …")
    dal = fetch_dal_cashflow()
    if not dal.empty:
        macro = macro.join(dal, how="left")
        macro["DAL_OperatingCF"] = macro["DAL_OperatingCF"].ffill()

    print("[3/4] Computing month-over-month percentage changes …")
    pct_cols = {}
    for col in macro.columns:
        pct_name = f"{col}_pct"
        pct_cols[pct_name] = macro[col].pct_change() * 100.0
    pct_df = pd.DataFrame(pct_cols, index=macro.index)
    combined = pd.concat([macro, pct_df], axis=1)
    combined = combined.dropna(how="all")

    if save:
        print(f"[4/4] Saving to {PARQUET_PATH} …")
        combined.to_parquet(PARQUET_PATH, engine="pyarrow")
    else:
        print("[4/4] Skipping save.")

    print(f"  Dataset shape: {combined.shape}")
    print(f"  Date range: {combined.index.min()} → {combined.index.max()}")
    return combined


def load_data(rebuild: bool = False) -> pd.DataFrame:
    """Load the merged dataset, building it first if necessary.

    Parameters
    ----------
    rebuild : bool
        Force a fresh download even if the Parquet file exists.

    Returns
    -------
    pd.DataFrame
    """
    if not rebuild and PARQUET_PATH.exists():
        print(f"Loading cached data from {PARQUET_PATH}")
        return pd.read_parquet(PARQUET_PATH)
    return build_dataset(save=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = build_dataset(save=True)
    print("\n— Preview —")
    print(df.tail(10).to_string())
