"""
anomaly_detector.py
-------------------
Detect yield and pricing anomalies in airline route economics using
Isolation Forest and Z-score methods.

Flags routes with:
* predatory pricing (yield below cash operating cost per RPM)
* exorbitant yield outliers (> 3.5σ from stage-length median curve)
* multivariate anomalies via Isolation Forest

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _z_score_anomalies(
    df: pd.DataFrame,
    sigma_threshold: float = 3.5,
) -> pd.Series:
    """Flag routes whose yield is > *sigma_threshold* σ from the
    stage-length-binned median yield.

    Returns a boolean Series aligned with *df*.
    """
    # Bin by stage length (250-mile buckets) and compute median yield per bin
    bins = pd.cut(df["distance_sm"], bins=np.arange(0, 15_001, 250))
    median_yield = df.groupby(bins, observed=False)["yield_per_rpm"].transform("median")
    std_yield = df.groupby(bins, observed=False)["yield_per_rpm"].transform("std")

    # Avoid division by zero for bins with single observation
    std_yield = std_yield.replace(0, np.nan)

    z = (df["yield_per_rpm"] - median_yield).abs() / std_yield
    return z > sigma_threshold


def _predatory_pricing(df: pd.DataFrame) -> pd.Series:
    """Flag routes where yield per RPM falls below cash operating cost per RPM.

    Cash operating cost per RPM = total_cost / rpm.
    """
    cash_cost_per_rpm = np.where(
        df["rpm"] > 0,
        df["total_cost"] / df["rpm"],
        np.nan,
    )
    return pd.Series(
        df["yield_per_rpm"] < cash_cost_per_rpm,
        index=df.index,
    )


def _isolation_forest_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.03,
    seed: int = 42,
) -> pd.Series:
    """Fit an Isolation Forest on (stage_length, yield_per_mile, load_factor)
    and return a boolean anomaly mask.
    """
    features = df[["distance_sm", "yield_per_rpm", "load_factor"]].copy()
    # Drop rows with NaN for fitting, but keep alignment
    valid_mask = features.notna().all(axis=1)
    predictions = pd.Series(False, index=df.index)

    if valid_mask.sum() < 10:
        return predictions  # Not enough data

    iso = IsolationForest(
        contamination=contamination,
        random_state=seed,
        n_estimators=200,
        n_jobs=-1,
    )
    preds = iso.fit_predict(features.loc[valid_mask])
    predictions.loc[valid_mask] = preds == -1
    return predictions


def detect_anomalies(
    routes_df: pd.DataFrame,
    sigma_threshold: float = 3.5,
    contamination: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all anomaly detection methods and return the enriched DataFrame.

    Added columns
    -------------
    is_zscore_anomaly       : bool – yield Z-score above threshold
    is_predatory_pricing    : bool – yield below cash cost per RPM
    is_isolation_anomaly    : bool – Isolation Forest outlier
    is_anomaly              : bool – flagged by any method
    anomaly_type            : str  – human-readable label(s)
    """
    df = routes_df.copy()

    df["is_zscore_anomaly"] = _z_score_anomalies(df, sigma_threshold)
    df["is_predatory_pricing"] = _predatory_pricing(df)
    df["is_isolation_anomaly"] = _isolation_forest_anomalies(df, contamination, seed)

    df["is_anomaly"] = (
        df["is_zscore_anomaly"] | df["is_predatory_pricing"] | df["is_isolation_anomaly"]
    )

    # Build human-readable anomaly type label
    labels: list[str] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        if row["is_zscore_anomaly"]:
            parts.append("Yield Z-score outlier")
        if row["is_predatory_pricing"]:
            parts.append("Predatory pricing")
        if row["is_isolation_anomaly"]:
            parts.append("Isolation Forest outlier")
        labels.append("; ".join(parts) if parts else "")
    df["anomaly_type"] = labels

    return df
