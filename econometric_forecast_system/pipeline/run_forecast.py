"""
run_forecast.py
===============
End-to-end econometric forecasting pipeline.

Orchestrates data acquisition, diagnostics, model training (ARIMA, SARIMAX,
Ridge, ElasticNet), evaluation with Diebold-Mariano significance testing,
and generates an interactive Plotly HTML report.

Usage
-----
    python -m pipeline.run_forecast

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tabulate import tabulate

# Ensure the project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_fred_data import load_data
from models.arima_model import find_best_arima, fit_sarimax, forecast_arima
from models.diagnostics import check_stationarity, granger_causality_matrix, make_stationary, plot_acf_pacf
from models.evaluation import (
    compute_mae,
    compute_mape,
    compute_mda,
    compute_rmse,
    diebold_mariano_test,
    model_comparison_table,
)
from models.regularized_linear import (
    create_lag_features,
    create_rolling_features,
    train_elasticnet,
    train_ridge,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_COL = "CPI_pct"  # month-over-month CPI % change
EXOG_COLS = ["FedFunds_pct", "Treasury10Y_pct", "IndustrialProduction_pct", "M2MoneyStock_pct"]
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_PATH = OUTPUT_DIR / "forecast_report.html"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_load_data() -> pd.DataFrame:
    """Step 1 — Load or fetch the dataset."""
    print("\n" + "=" * 70)
    print("  STEP 1: Data Acquisition")
    print("=" * 70)
    df = load_data(rebuild=False)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    return df


def step_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """Step 2 — Run stationarity diagnostics on all pct-change series."""
    print("\n" + "=" * 70)
    print("  STEP 2: Stationarity Diagnostics")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    diag_results: Dict[str, Any] = {}
    pct_cols = [c for c in df.columns if c.endswith("_pct")]

    for col in pct_cols:
        series = df[col].dropna()
        if len(series) < 20:
            print(f"  Skipping {col}: too few observations ({len(series)})")
            continue
        result = check_stationarity(series, name=col)
        diag_results[col] = result
        try:
            plot_acf_pacf(series, lags=40, save_path=str(OUTPUT_DIR / f"acf_pacf_{col}.png"))
        except Exception as exc:
            print(f"  Could not plot ACF/PACF for {col}: {exc}")

    return diag_results


def step_granger(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2b — Granger causality analysis."""
    print("\n" + "=" * 70)
    print("  STEP 2b: Granger Causality Analysis")
    print("=" * 70)
    analysis_cols = [TARGET_COL] + [c for c in EXOG_COLS if c in df.columns]
    subset = df[analysis_cols].dropna()
    gc_matrix = granger_causality_matrix(subset, target_col=TARGET_COL, maxlag=4)
    if not gc_matrix.empty:
        sig = gc_matrix[gc_matrix["significant"]]
        print(f"\n  Significant Granger-causal relationships ({len(sig)} entries):")
        if not sig.empty:
            print(sig.to_string(index=False))
    return gc_matrix


def step_split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Step 3 — Temporal train/test split (80/20)."""
    print("\n" + "=" * 70)
    print("  STEP 3: Train / Test Split")
    print("=" * 70)
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    analysis = df[pct_cols].dropna()

    split_idx = int(len(analysis) * 0.8)
    train = analysis.iloc[:split_idx]
    test = analysis.iloc[split_idx:]

    y_train = train[TARGET_COL]
    y_test = test[TARGET_COL]

    print(f"  Train: {len(train)} obs  ({train.index.min()} → {train.index.max()})")
    print(f"  Test : {len(test)} obs   ({test.index.min()} → {test.index.max()})")
    return train, test, y_train, y_test


def step_arima(
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Step 4 — Fit baseline ARIMA on the target series."""
    print("\n" + "=" * 70)
    print("  STEP 4: Baseline ARIMA")
    print("=" * 70)
    best_order, fitted = find_best_arima(y_train, max_p=4, max_d=2, max_q=4)

    # In-sample predictions
    train_pred = fitted.fittedvalues.values

    # Out-of-sample forecast
    fc = forecast_arima(fitted, steps=len(y_test))
    test_pred = fc["forecast"].values
    return train_pred, test_pred, fitted


def step_sarimax(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Step 5 — Fit SARIMAX with exogenous macro variables."""
    print("\n" + "=" * 70)
    print("  STEP 5: SARIMAX with Exogenous Variables")
    print("=" * 70)
    exog_available = [c for c in EXOG_COLS if c in train.columns]
    if not exog_available:
        print("  No exogenous variables available — falling back to ARIMA")
        return step_arima(y_train, y_test)

    X_train_exog = train[exog_available]
    X_test_exog = test[exog_available]

    # Use a modest seasonal order to avoid convergence issues
    try:
        model = fit_sarimax(
            y_train,
            X_train_exog,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 12),
        )
    except Exception as exc:
        print(f"  SARIMAX failed with seasonal order: {exc}")
        print("  Retrying without seasonality …")
        model = fit_sarimax(
            y_train,
            X_train_exog,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
        )

    train_pred = model.fittedvalues.values

    fc = forecast_arima(model, steps=len(y_test), exog_future=X_test_exog)
    test_pred = fc["forecast"].values
    return train_pred, test_pred, model


def step_linear_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Step 6 — Create lag/rolling features, fit Ridge and ElasticNet."""
    print("\n" + "=" * 70)
    print("  STEP 6: Regularised Linear Models")
    print("=" * 70)

    # Combine train + test for consistent feature engineering
    full = pd.concat([train, test])
    full = create_lag_features(full, target_col=TARGET_COL)
    full = create_rolling_features(full)

    # Re-split
    train_fe = full.iloc[: len(train)]
    test_fe = full.iloc[len(train):]

    feature_cols = [c for c in train_fe.columns if c != TARGET_COL]
    X_train_fe = train_fe[feature_cols]
    X_test_fe = test_fe[feature_cols]

    # Ridge
    print("\n  Training Ridge …")
    ridge_model, ridge_pred, ridge_imp = train_ridge(X_train_fe, y_train, X_test_fe, y_test)

    # ElasticNet
    print("  Training ElasticNet …")
    enet_model, enet_pred, enet_imp = train_elasticnet(X_train_fe, y_train, X_test_fe, y_test)

    # In-sample predictions for train metrics
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_fe.fillna(0).replace([np.inf, -np.inf], 0))
    ridge_train_pred = ridge_model.predict(X_tr_s)
    enet_train_pred = enet_model.predict(X_tr_s)

    return {
        "ridge": {
            "model": ridge_model,
            "train_pred": ridge_train_pred,
            "test_pred": ridge_pred,
            "importance": ridge_imp,
        },
        "elasticnet": {
            "model": enet_model,
            "train_pred": enet_train_pred,
            "test_pred": enet_pred,
            "importance": enet_imp,
        },
    }


def step_evaluate(
    y_train: pd.Series,
    y_test: pd.Series,
    predictions: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Step 7 — Evaluate all models and build comparison table."""
    print("\n" + "=" * 70)
    print("  STEP 7: Model Evaluation")
    print("=" * 70)

    results: Dict[str, Dict[str, float]] = {}
    for name, preds in predictions.items():
        train_pred = preds["train_pred"]
        test_pred = preds["test_pred"]

        # Align lengths (ARIMA fitted values can differ in length)
        tr_actual = y_train.values[-len(train_pred):]
        te_actual = y_test.values[: len(test_pred)]
        train_pred = train_pred[: len(tr_actual)]
        test_pred = test_pred[: len(te_actual)]

        results[name] = {
            "train_rmse": compute_rmse(tr_actual, train_pred),
            "test_rmse": compute_rmse(te_actual, test_pred),
            "test_mae": compute_mae(te_actual, test_pred),
            "test_mape": compute_mape(te_actual, test_pred),
            "test_mda": compute_mda(te_actual, test_pred),
        }

    table = model_comparison_table(results)
    print("\n" + tabulate(table, headers="keys", tablefmt="github", floatfmt=".4f"))
    return table, results


def step_diebold_mariano(
    y_test: pd.Series,
    predictions: Dict[str, Dict[str, np.ndarray]],
    results: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """Step 8 — Diebold-Mariano test between best two models."""
    print("\n" + "=" * 70)
    print("  STEP 8: Diebold-Mariano Test")
    print("=" * 70)

    # Sort models by test_rmse
    ranked = sorted(results.items(), key=lambda x: x[1]["test_rmse"])
    if len(ranked) < 2:
        print("  Need at least 2 models for DM test.")
        return {}

    m1_name = ranked[0][0]
    m2_name = ranked[1][0]
    p1 = predictions[m1_name]["test_pred"]
    p2 = predictions[m2_name]["test_pred"]
    actual = y_test.values

    min_len = min(len(actual), len(p1), len(p2))
    dm = diebold_mariano_test(actual[:min_len], p1[:min_len], p2[:min_len], h=1)
    print(f"\n  Comparing: {m1_name} vs {m2_name}")
    print(f"  DM statistic: {dm['dm_stat']}")
    print(f"  p-value     : {dm['p_value']}")
    print(f"  Conclusion  : {dm['conclusion']}")
    return dm


def step_report(
    y_train: pd.Series,
    y_test: pd.Series,
    predictions: Dict[str, Dict[str, np.ndarray]],
    comparison_table: pd.DataFrame,
    linear_results: Dict[str, Any],
    dm_result: Dict[str, Any],
) -> None:
    """Step 9 — Generate interactive Plotly HTML report."""
    print("\n" + "=" * 70)
    print("  STEP 9: Generating HTML Report")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Actual vs Predicted (All Models)",
            "Forecast Cone — Best Model",
            "Feature Importance (Ridge)",
            "Residual Distribution",
            "Model Comparison",
            "Q-Q Plot of Residuals",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # --- Panel 1: Actual vs Predicted ---
    test_idx = y_test.index if hasattr(y_test, "index") else list(range(len(y_test)))

    fig.add_trace(
        go.Scatter(
            x=test_idx,
            y=y_test.values,
            name="Actual",
            mode="lines",
            line=dict(color="black", width=2),
        ),
        row=1, col=1,
    )

    colours = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    for i, (name, preds) in enumerate(predictions.items()):
        tp = preds["test_pred"][: len(y_test)]
        fig.add_trace(
            go.Scatter(
                x=test_idx[: len(tp)],
                y=tp,
                name=name,
                mode="lines",
                line=dict(color=colours[i % len(colours)], dash="dash"),
            ),
            row=1, col=1,
        )

    # --- Panel 2: Forecast cone for best model ---
    best_name = comparison_table.index[0]  # lowest test_rmse
    best_pred = predictions[best_name]["test_pred"][: len(y_test)]
    residuals = y_test.values[: len(best_pred)] - best_pred
    std_res = np.std(residuals)

    upper = best_pred + 1.96 * std_res
    lower = best_pred - 1.96 * std_res

    fig.add_trace(
        go.Scatter(x=test_idx[: len(best_pred)], y=y_test.values[: len(best_pred)],
                    name="Actual", mode="lines", line=dict(color="black"),
                    showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=test_idx[: len(best_pred)], y=best_pred,
                    name=f"{best_name} forecast", mode="lines",
                    line=dict(color="#2563eb"), showlegend=False),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=list(test_idx[: len(best_pred)]) + list(test_idx[: len(best_pred)][::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(37,99,235,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # --- Panel 3: Feature importance bar chart ---
    if "ridge" in linear_results:
        imp = linear_results["ridge"]["importance"].head(20)
        fig.add_trace(
            go.Bar(
                x=imp["abs_coefficient"].values,
                y=imp["feature"].values,
                orientation="h",
                marker_color="#2563eb",
                name="Ridge importance",
                showlegend=False,
            ),
            row=2, col=1,
        )

    # --- Panel 4: Residual histogram ---
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color="#16a34a",
            name="Residuals",
            showlegend=False,
        ),
        row=2, col=2,
    )

    # --- Panel 5: Comparison table as heatmap ---
    z_vals = comparison_table.values.astype(float)
    fig.add_trace(
        go.Heatmap(
            z=z_vals,
            x=list(comparison_table.columns),
            y=list(comparison_table.index),
            colorscale="RdYlGn_r",
            showscale=True,
            text=np.round(z_vals, 4).astype(str),
            texttemplate="%{text}",
            name="Metrics",
            showlegend=False,
        ),
        row=3, col=1,
    )

    # --- Panel 6: Q-Q plot ---
    sorted_res = np.sort(residuals)
    theoretical_q = stats_norm_ppf(sorted_res)
    fig.add_trace(
        go.Scatter(
            x=theoretical_q,
            y=sorted_res,
            mode="markers",
            marker=dict(color="#9333ea", size=4),
            name="Q-Q",
            showlegend=False,
        ),
        row=3, col=2,
    )
    qq_min = min(theoretical_q.min(), sorted_res.min())
    qq_max = max(theoretical_q.max(), sorted_res.max())
    fig.add_trace(
        go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_min, qq_max],
            mode="lines",
            line=dict(color="red", dash="dash"),
            showlegend=False,
        ),
        row=3, col=2,
    )

    # Layout
    fig.update_layout(
        title_text=(
            "Multivariate Econometric Forecasting Report<br>"
            "<sup>Author: Eyoas Gizachew Zewdie | University of Messina</sup>"
        ),
        height=1100,
        width=1400,
        template="plotly_white",
        font=dict(size=11),
    )

    fig.write_html(str(REPORT_PATH), include_plotlyjs="cdn")
    print(f"\n  Report saved → {REPORT_PATH}")


def stats_norm_ppf(sorted_data: np.ndarray) -> np.ndarray:
    """Compute theoretical quantiles for a Q-Q plot against the normal distribution."""
    from scipy import stats as sp_stats
    n = len(sorted_data)
    probs = (np.arange(1, n + 1) - 0.5) / n
    return sp_stats.norm.ppf(probs)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full forecasting pipeline."""
    print("\n" + "▓" * 70)
    print("  MULTIVARIATE ECONOMETRIC FORECASTING SYSTEM")
    print("  Author: Eyoas Gizachew Zewdie — University of Messina")
    print("▓" * 70)

    # 1. Data
    df = step_load_data()

    # 2. Diagnostics
    diag = step_diagnostics(df)
    gc = step_granger(df)

    # 3. Split
    train, test, y_train, y_test = step_split_data(df)

    # 4. ARIMA
    arima_train, arima_test, arima_model = step_arima(y_train, y_test)

    # 5. SARIMAX
    sarimax_train, sarimax_test, sarimax_model = step_sarimax(
        train, test, y_train, y_test
    )

    # 6. Linear models
    linear = step_linear_models(train, test, y_train, y_test)

    # Collect predictions
    all_predictions: Dict[str, Dict[str, np.ndarray]] = {
        "ARIMA": {"train_pred": arima_train, "test_pred": arima_test},
        "SARIMAX": {"train_pred": sarimax_train, "test_pred": sarimax_test},
        "Ridge": {
            "train_pred": linear["ridge"]["train_pred"],
            "test_pred": linear["ridge"]["test_pred"],
        },
        "ElasticNet": {
            "train_pred": linear["elasticnet"]["train_pred"],
            "test_pred": linear["elasticnet"]["test_pred"],
        },
    }

    # 7. Evaluate
    table, results = step_evaluate(y_train, y_test, all_predictions)

    # 8. Diebold-Mariano
    dm = step_diebold_mariano(y_test, all_predictions, results)

    # 9. HTML report
    step_report(y_train, y_test, all_predictions, table, linear, dm)

    print("\n" + "▓" * 70)
    print("  Pipeline complete.")
    print("▓" * 70 + "\n")


if __name__ == "__main__":
    main()
