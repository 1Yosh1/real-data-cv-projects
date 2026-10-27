"""
regularized_linear.py
=====================
Regularised linear models for time-series forecasting.

Implements feature engineering (lag and rolling features) and trains Ridge
and ElasticNet regressors with temporal cross-validation.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Create lagged features for every column in *df*.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Name of the target column (included in lag creation).
    lags : list[int]
        Lag orders. Defaults to ``[1, 2, 3, 6, 12]``.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``<col>_lag<n>`` columns.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12]

    result = df.copy()
    for col in df.columns:
        for lag in lags:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
    return result


def create_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Create rolling-window statistics (mean, std) for all columns.

    Parameters
    ----------
    df : pd.DataFrame
    windows : list[int]
        Window sizes. Defaults to ``[3, 6, 12]``.

    Returns
    -------
    pd.DataFrame
        Original columns plus rolling mean/std columns.
    """
    if windows is None:
        windows = [3, 6, 12]

    result = df.copy()
    for col in df.columns:
        for w in windows:
            result[f"{col}_rmean{w}"] = df[col].rolling(window=w, min_periods=1).mean()
            result[f"{col}_rstd{w}"] = df[col].rolling(window=w, min_periods=1).std()
    return result


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def _prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features and handle any remaining NaNs."""
    X_tr = X_train.copy().fillna(0).replace([np.inf, -np.inf], 0)
    X_te = X_test.copy().fillna(0).replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    return X_tr_scaled, X_te_scaled, scaler


def _feature_importance(
    model: Any,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract and rank feature importance from a linear model."""
    coefs = model.coef_ if hasattr(model, "coef_") else np.zeros(len(feature_names))
    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    })
    importance = importance.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return importance


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, np.ndarray, pd.DataFrame]:
    """Train RidgeCV with temporal cross-validation.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Target vectors.

    Returns
    -------
    tuple
        ``(fitted_model, test_predictions, feature_importance_df)``
    """
    X_tr, X_te, scaler = _prepare_features(X_train, X_test)
    y_tr = y_train.values.ravel()

    alphas = np.logspace(-3, 3, 20)
    tscv = TimeSeriesSplit(n_splits=5)

    model = RidgeCV(alphas=alphas, cv=tscv, scoring="neg_mean_squared_error")
    model.fit(X_tr, y_tr)

    predictions = model.predict(X_te)
    importance = _feature_importance(model, list(X_train.columns))

    print(f"  Ridge — best alpha: {model.alpha_:.4f}")
    return model, predictions, importance


# ---------------------------------------------------------------------------
# ElasticNet regression
# ---------------------------------------------------------------------------

def train_elasticnet(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Any, np.ndarray, pd.DataFrame]:
    """Train ElasticNetCV with temporal cross-validation.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices.
    y_train, y_test : pd.Series
        Target vectors.

    Returns
    -------
    tuple
        ``(fitted_model, test_predictions, feature_importance_df)``
    """
    X_tr, X_te, scaler = _prepare_features(X_train, X_test)
    y_tr = y_train.values.ravel()

    tscv = TimeSeriesSplit(n_splits=5)
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=np.logspace(-4, 2, 30),
        cv=tscv,
        max_iter=10000,
        tol=1e-4,
    )
    model.fit(X_tr, y_tr)

    predictions = model.predict(X_te)
    importance = _feature_importance(model, list(X_train.columns))

    print(f"  ElasticNet — alpha: {model.alpha_:.6f}, l1_ratio: {model.l1_ratio_:.2f}")
    return model, predictions, importance
