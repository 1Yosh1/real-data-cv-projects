"""
ECB Foreign Exchange Rate Ingestion Module.

Downloads and parses official ECB daily reference rates from the European Central Bank.
Provides forward-filled rates covering all calendar days for reliable FX lookups.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Currencies supported by this pipeline
SUPPORTED_CURRENCIES: list[str] = [
    "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "CNY", "SEK", "NOK", "DKK",
    "PLN", "CZK", "HUF", "TRY", "BRL", "MXN", "KRW", "SGD", "HKD", "INR",
    "ZAR", "NZD", "THB", "MYR", "PHP", "IDR", "ILS", "RON", "BGN", "HRK",
    "ISK", "RUB",
]

ECB_ZIP_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"
ECB_CSV_NAME = "eurofxref-hist.csv"


def download_ecb_rates(data_dir: str = "data/") -> Path:
    """Download the ECB historical daily reference-rate ZIP and extract the CSV.

    Parameters
    ----------
    data_dir : str
        Directory where the CSV will be stored.

    Returns
    -------
    Path
        Absolute path to the extracted CSV file.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_path = data_path / ECB_CSV_NAME

    if csv_path.exists():
        return csv_path

    response = requests.get(ECB_ZIP_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extract(ECB_CSV_NAME, path=str(data_path))

    return csv_path


def parse_ecb_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse the wide-format ECB CSV into a tidy long-format DataFrame.

    The ECB CSV has a header row with currency codes and each subsequent row
    starts with a date followed by rates.  Values marked "N/A" are treated as
    missing.  The file contains trailing commas, which Pandas reads as an
    unnamed extra column; those are dropped automatically.

    Parameters
    ----------
    csv_path : str | Path
        Path to the extracted ``eurofxref-hist.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (datetime64), ``currency_code`` (str),
        ``rate_to_eur`` (float64).
    """
    raw = pd.read_csv(csv_path, na_values=["N/A", ""])

    # Drop any unnamed trailing columns created by trailing commas
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]

    # The first column is the date column — its header is literally "Date"
    date_col = raw.columns[0]
    raw.rename(columns={date_col: "Date"}, inplace=True)
    raw["Date"] = pd.to_datetime(raw["Date"].str.strip(), format="%Y-%m-%d")

    # Strip whitespace from column names (ECB puts spaces after commas)
    raw.columns = [c.strip() for c in raw.columns]

    # Filter to only supported currencies that exist in the data
    available_currencies = [c for c in SUPPORTED_CURRENCIES if c in raw.columns]

    # Melt wide → long
    tidy = raw.melt(
        id_vars=["Date"],
        value_vars=available_currencies,
        var_name="currency_code",
        value_name="rate_to_eur",
    )
    tidy.rename(columns={"Date": "date"}, inplace=True)
    tidy.dropna(subset=["rate_to_eur"], inplace=True)
    tidy["rate_to_eur"] = tidy["rate_to_eur"].astype(float)
    tidy.sort_values(["currency_code", "date"], inplace=True)
    tidy.reset_index(drop=True, inplace=True)

    return tidy


def forward_fill_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill FX rates so every calendar day has a rate for each currency.

    Weekends and bank holidays are filled by carrying forward the last
    available business-day rate.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy FX rates with columns ``date``, ``currency_code``, ``rate_to_eur``.

    Returns
    -------
    pd.DataFrame
        Same schema with rows added for missing calendar days.
    """
    min_date = df["date"].min()
    max_date = df["date"].max()
    all_dates = pd.date_range(min_date, max_date, freq="D")
    currencies = df["currency_code"].unique()

    # Build a full Cartesian product of dates × currencies
    idx = pd.MultiIndex.from_product(
        [all_dates, currencies], names=["date", "currency_code"]
    )
    full = pd.DataFrame(index=idx).reset_index()

    merged = full.merge(df, on=["date", "currency_code"], how="left")
    merged.sort_values(["currency_code", "date"], inplace=True)

    # Forward-fill within each currency group
    merged["rate_to_eur"] = merged.groupby("currency_code")["rate_to_eur"].ffill()

    # Backward-fill the very first days if they fall before the first observation
    merged["rate_to_eur"] = merged.groupby("currency_code")["rate_to_eur"].bfill()

    merged.reset_index(drop=True, inplace=True)
    return merged


def load_fx_rates(data_dir: str = "data/") -> pd.DataFrame:
    """Main entry point: download ECB rates, parse, forward-fill, return clean DF.

    Parameters
    ----------
    data_dir : str
        Directory for data storage.

    Returns
    -------
    pd.DataFrame
        Forward-filled tidy FX rates with columns ``date``, ``currency_code``,
        ``rate_to_eur``.
    """
    csv_path = download_ecb_rates(data_dir=data_dir)
    tidy = parse_ecb_csv(csv_path)
    filled = forward_fill_rates(tidy)
    return filled
