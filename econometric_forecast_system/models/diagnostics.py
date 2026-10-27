"""
diagnostics.py
==============
Time-series diagnostic tools for econometric analysis.

Provides stationarity testing (ADF, KPSS), ACF/PACF visualisation,
Granger causality analysis, and automatic stationarity transformations.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss


# ---------------------------------------------------------------------------
# Stationarity testing
# ---------------------------------------------------------------------------

def check_stationarity(
    series: pd.Series,
    name: str = "",
    significance: float = 0.05,
) -> Dict[str, Any]:
    """Run Augmented Dickey-Fuller and KPSS tests on *series*.

    Parameters
    ----------
    series : pd.Series
        The time-series to test (NaNs are dropped automatically).
    name : str
        Human-readable label used in printed output.
    significance : float
        Significance level for hypothesis testing (default 5 %).

    Returns
    -------
    dict
        Keys: adf_stat, adf_pvalue, kpss_stat, kpss_pvalue,
              is_stationary_adf, is_stationary_kpss, conclusion.
    """
    clean = series.dropna().astype(float)
    label = name or series.name or "series"

    # ADF test  (H₀: unit root exists → non-stationary)
    adf_result = adfuller(clean, autolag="AIC")
    adf_stat, adf_p = adf_result[0], adf_result[1]
    is_stat_adf = adf_p < significance

    # KPSS test (H₀: series is stationary)
    kpss_result = kpss(clean, regression="c", nlags="auto")
    kpss_stat, kpss_p = kpss_result[0], kpss_result[1]
    is_stat_kpss = kpss_p >= significance

    # Joint conclusion
    if is_stat_adf and is_stat_kpss:
        conclusion = "Stationary"
    elif not is_stat_adf and not is_stat_kpss:
        conclusion = "Non-stationary"
    elif is_stat_adf and not is_stat_kpss:
        conclusion = "Trend-stationary (difference-stationary likely)"
    else:
        conclusion = "Inconclusive (ADF says non-stationary, KPSS says stationary)"

    result = {
        "adf_stat": round(adf_stat, 4),
        "adf_pvalue": round(adf_p, 4),
        "is_stationary_adf": is_stat_adf,
        "kpss_stat": round(kpss_stat, 4),
        "kpss_pvalue": round(kpss_p, 4),
        "is_stationary_kpss": is_stat_kpss,
        "conclusion": conclusion,
    }

    print(f"\n{'='*60}")
    print(f"  Stationarity diagnostics: {label}")
    print(f"{'='*60}")
    print(f"  ADF statistic : {result['adf_stat']}")
    print(f"  ADF p-value   : {result['adf_pvalue']}")
    print(f"  ADF stationary: {result['is_stationary_adf']}")
    print(f"  KPSS statistic: {result['kpss_stat']}")
    print(f"  KPSS p-value  : {result['kpss_pvalue']}")
    print(f"  KPSS stationary: {result['is_stationary_kpss']}")
    print(f"  Conclusion    : {result['conclusion']}")
    print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# ACF / PACF plots
# ---------------------------------------------------------------------------

def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    save_path: Optional[str] = None,
) -> str:
    """Create ACF and PACF plots side by side.

    Parameters
    ----------
    series : pd.Series
        The time-series to plot.
    lags : int
        Number of lags to include.
    save_path : str or None
        If None, a default path is generated from the series name.

    Returns
    -------
    str
        Path to the saved figure.
    """
    clean = series.dropna().astype(float)
    label = series.name or "series"

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(clean, lags=min(lags, len(clean) // 2 - 1), ax=axes[0], title=f"ACF – {label}")
    plot_pacf(clean, lags=min(lags, len(clean) // 2 - 1), ax=axes[1], title=f"PACF – {label}", method="ywm")
    plt.tight_layout()

    if save_path is None:
        save_path = f"output/acf_pacf_{label.replace(' ', '_')}.png"

    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "output", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ACF/PACF plot saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

def granger_causality_matrix(
    df: pd.DataFrame,
    target_col: str,
    maxlag: int = 4,
) -> pd.DataFrame:
    """Test which columns in *df* Granger-cause *target_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Panel of time-series columns (must be stationary).
    target_col : str
        Name of the dependent variable.
    maxlag : int
        Maximum number of lags to test.

    Returns
    -------
    pd.DataFrame
        Columns: variable, lag, f_stat, p_value, significant.
    """
    clean = df.dropna()
    predictors = [c for c in clean.columns if c != target_col]
    rows: List[Dict[str, Any]] = []

    for predictor in predictors:
        subset = clean[[target_col, predictor]].dropna()
        if len(subset) < maxlag + 10:
            continue
        try:
            gc_result = grangercausalitytests(
                subset[[target_col, predictor]],
                maxlag=maxlag,
                verbose=False,
            )
            for lag in range(1, maxlag + 1):
                f_stat = gc_result[lag][0]["ssr_ftest"][0]
                p_val = gc_result[lag][0]["ssr_ftest"][1]
                rows.append({
                    "variable": predictor,
                    "lag": lag,
                    "f_stat": round(f_stat, 4),
                    "p_value": round(p_val, 4),
                    "significant": p_val < 0.05,
                })
        except Exception as exc:
            print(f"  Granger test failed for {predictor}: {exc}")

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(["variable", "lag"]).reset_index(drop=True)
    return result_df


# ---------------------------------------------------------------------------
# Automatic stationarity transformation
# ---------------------------------------------------------------------------

def make_stationary(
    series: pd.Series,
    max_diffs: int = 2,
    significance: float = 0.05,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Apply differencing (and optionally log-transform) to achieve stationarity.

    Strategy
    --------
    1. If all values > 0 and the series is not stationary, try log-transform first.
    2. Apply successive differencing (up to *max_diffs*).
    3. Return the transformed series and metadata for inversion.

    Parameters
    ----------
    series : pd.Series
    max_diffs : int
    significance : float

    Returns
    -------
    tuple[pd.Series, dict]
        Transformed series and metadata dict with keys:
        ``log_applied``, ``n_diffs``, ``original_name``.
    """
    clean = series.dropna().astype(float)
    label = series.name or "series"
    meta: Dict[str, Any] = {
        "original_name": label,
        "log_applied": False,
        "n_diffs": 0,
    }

    current = clean.copy()

    # Attempt log-transform if values are strictly positive
    if (current > 0).all():
        log_candidate = np.log(current)
        adf_p = adfuller(log_candidate.dropna(), autolag="AIC")[1]
        if adf_p < significance:
            current = log_candidate
            current.name = f"log({label})"
            meta["log_applied"] = True
            print(f"  Log-transform applied to {label} → stationary (ADF p={adf_p:.4f})")
            return current.dropna(), meta

    # Successive differencing
    for d in range(1, max_diffs + 1):
        current = current.diff()
        adf_p = adfuller(current.dropna(), autolag="AIC")[1]
        meta["n_diffs"] = d
        if adf_p < significance:
            current.name = f"d{d}({label})"
            print(f"  Differencing order {d} applied to {label} → stationary (ADF p={adf_p:.4f})")
            return current.dropna(), meta

    # If still not stationary, try log + diff
    if (clean > 0).all() and not meta["log_applied"]:
        log_series = np.log(clean)
        for d in range(1, max_diffs + 1):
            log_series = log_series.diff()
            adf_p = adfuller(log_series.dropna(), autolag="AIC")[1]
            if adf_p < significance:
                meta["log_applied"] = True
                meta["n_diffs"] = d
                log_series.name = f"d{d}(log({label}))"
                print(f"  Log + diff({d}) applied to {label} → stationary (ADF p={adf_p:.4f})")
                return log_series.dropna(), meta

    current.name = f"d{meta['n_diffs']}({label})"
    print(f"  Warning: {label} may still be non-stationary after {max_diffs} diffs.")
    return current.dropna(), meta
