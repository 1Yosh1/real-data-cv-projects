"""
arima_model.py
==============
ARIMA and SARIMAX modelling utilities for time-series forecasting.

Includes grid-search order selection via AIC, SARIMAX estimation with
exogenous regressors, and forecast generation with confidence intervals.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Grid-search for best ARIMA(p,d,q)
# ---------------------------------------------------------------------------

def find_best_arima(
    series: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
) -> Tuple[Tuple[int, int, int], Any]:
    """Exhaustive grid-search over (p, d, q) to minimise AIC.

    Parameters
    ----------
    series : pd.Series
        Univariate time-series (should be stationary or close to it).
    max_p, max_d, max_q : int
        Upper bounds for the order search space.

    Returns
    -------
    tuple
        ``(best_order, fitted_model)`` where *best_order* is ``(p, d, q)``
        and *fitted_model* is a ``SARIMAXResultsWrapper``.
    """
    clean = series.dropna().astype(float)

    best_aic = np.inf
    best_order: Tuple[int, int, int] = (1, 0, 0)
    best_model: Any = None

    p_range = range(0, max_p + 1)
    d_range = range(0, max_d + 1)
    q_range = range(0, max_q + 1)

    total = (max_p + 1) * (max_d + 1) * (max_q + 1)
    tested = 0

    for p, d, q in itertools.product(p_range, d_range, q_range):
        if p == 0 and q == 0:
            continue  # trivial model
        tested += 1
        try:
            model = SARIMAX(
                clean,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
            if result.aic < best_aic:
                best_aic = result.aic
                best_order = (p, d, q)
                best_model = result
        except Exception:
            continue

    print(f"  ARIMA grid-search: tested {tested}/{total} combinations")
    print(f"  Best order: {best_order}, AIC: {best_aic:.2f}")
    return best_order, best_model


# ---------------------------------------------------------------------------
# SARIMAX with exogenous variables
# ---------------------------------------------------------------------------

def fit_sarimax(
    endog: pd.Series,
    exog: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
) -> Any:
    """Fit a SARIMAX model with exogenous regressors.

    Parameters
    ----------
    endog : pd.Series
        Endogenous (target) series.
    exog : pd.DataFrame
        Exogenous regressor matrix (same index as *endog*).
    order : tuple
        ARIMA (p, d, q) order.
    seasonal_order : tuple
        Seasonal (P, D, Q, s) order.

    Returns
    -------
    SARIMAXResultsWrapper
    """
    endog_clean = endog.dropna().astype(float)
    common_idx = endog_clean.index.intersection(exog.index)
    endog_clean = endog_clean.loc[common_idx]
    exog_clean = exog.loc[common_idx].astype(float).fillna(0)

    model = SARIMAX(
        endog_clean,
        exog=exog_clean,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=400)
    print(f"  SARIMAX fitted: AIC={result.aic:.2f}, BIC={result.bic:.2f}")
    return result


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def forecast_arima(
    model: Any,
    steps: int,
    exog_future: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Generate out-of-sample forecasts with confidence intervals.

    Parameters
    ----------
    model : SARIMAXResultsWrapper
        Fitted model.
    steps : int
        Number of periods to forecast.
    exog_future : pd.DataFrame or None
        Future exogenous values (required if model was fit with exog).
    alpha : float
        Significance level for CIs (default 5 % → 95 % intervals).

    Returns
    -------
    pd.DataFrame
        Columns: ``forecast``, ``lower_ci``, ``upper_ci``.
    """
    forecast_result = model.get_forecast(steps=steps, exog=exog_future, alpha=alpha)
    mean_forecast = forecast_result.predicted_mean
    ci = forecast_result.conf_int(alpha=alpha)

    result_df = pd.DataFrame(
        {
            "forecast": mean_forecast.values,
            "lower_ci": ci.iloc[:, 0].values,
            "upper_ci": ci.iloc[:, 1].values,
        },
        index=mean_forecast.index,
    )
    return result_df
