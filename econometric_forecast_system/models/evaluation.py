"""
evaluation.py
=============
Forecast evaluation metrics and model comparison utilities.

Implements RMSE, MAE, MAPE, MDA, the Diebold-Mariano test for comparing
predictive accuracy, and a formatted comparison-table builder.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------

def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    actual, predicted : array-like

    Returns
    -------
    float
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    actual, predicted : array-like

    Returns
    -------
    float
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error with zero-safe denominator.

    Observations where ``actual == 0`` are excluded from the calculation
    to avoid division by zero.

    Parameters
    ----------
    actual, predicted : array-like

    Returns
    -------
    float
        Percentage (0–100+ scale).
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = a != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def compute_mda(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Directional Accuracy.

    Measures the fraction of time-steps where the predicted direction
    of change matches the actual direction of change.

    Parameters
    ----------
    actual, predicted : array-like

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)

    if len(a) < 2:
        return np.nan

    actual_dir = np.sign(np.diff(a))
    pred_dir = np.sign(np.diff(p))
    return float(np.mean(actual_dir == pred_dir))


# ---------------------------------------------------------------------------
# Diebold-Mariano test
# ---------------------------------------------------------------------------

def diebold_mariano_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int = 1,
) -> Dict[str, float]:
    """Diebold-Mariano test for comparing predictive accuracy.

    Tests H₀: the two forecasts have equal predictive accuracy
    against H₁: the forecasts differ in accuracy.

    Uses the squared-error loss differential:
        dₜ = e₁ₜ² − e₂ₜ²

    The DM statistic under the Harvey, Leybourne & Newbold (1997)
    small-sample correction is:

        DM = d̄ / √( (1/T) * V̂(d) )

    where V̂(d) accounts for h-step-ahead autocorrelation.

    Parameters
    ----------
    actual : array-like
        Observed values.
    pred1, pred2 : array-like
        Forecasts from model 1 and model 2.
    h : int
        Forecast horizon (used for HAC variance correction).

    Returns
    -------
    dict
        ``dm_stat``, ``p_value``, ``conclusion``.
    """
    a = np.asarray(actual, dtype=float)
    p1 = np.asarray(pred1, dtype=float)
    p2 = np.asarray(pred2, dtype=float)

    e1 = a - p1
    e2 = a - p2
    d = e1 ** 2 - e2 ** 2

    T = len(d)
    d_bar = np.mean(d)

    # Autocovariance for h-step-ahead
    gamma = []
    for k in range(h):
        cov_k = np.mean((d[k:] - d_bar) * (d[: T - k] - d_bar))
        gamma.append(cov_k)

    V_d = gamma[0] + 2 * sum(gamma[1:]) if len(gamma) > 1 else gamma[0]
    V_d = max(V_d, 1e-12)  # numerical safety

    dm_stat = d_bar / np.sqrt(V_d / T)

    # Harvey-Leybourne-Newbold small-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_stat_adj = dm_stat * correction

    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_adj), df=T - 1))

    if p_value < 0.05:
        if dm_stat_adj > 0:
            conclusion = "Model 2 significantly more accurate than Model 1"
        else:
            conclusion = "Model 1 significantly more accurate than Model 2"
    else:
        conclusion = "No significant difference in predictive accuracy"

    return {
        "dm_stat": round(float(dm_stat_adj), 4),
        "p_value": round(float(p_value), 4),
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def model_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Build a formatted comparison table from a results dictionary.

    Parameters
    ----------
    results_dict : dict
        ``{ model_name: { metric_name: value, … }, … }``

    Returns
    -------
    pd.DataFrame
        One row per model, columns for each metric.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index("Model")

    # Sort by test RMSE if available
    if "test_rmse" in df.columns:
        df = df.sort_values("test_rmse")

    return df
