# MacroFlow: Multivariate Econometric Volatility & Cash Forecasting System

**Author**: Eyoas Gizachew Zewdie  
*B.Sc. Data Analysis, University of Messina*  
*Contact / Portfolio Project*

---

## 1. Executive Summary

MacroFlow forecasts corporate operational cash flow volatility using real macroeconomic regressors from the Federal Reserve Economic Data (FRED) combined with corporate revenue series from Yahoo Finance (Delta Air Lines, ticker `DAL`). The system evaluates univariate baseline ARIMA against multivariate SARIMAX with exogenous macroeconomic variables and regularized shrinkage models (Ridge, ElasticNet).

---

## 2. Real Data Architecture

No dummy data. Data ingested directly from official endpoints:

| Indicator | Source | Endpoint / Series ID | Description | Frequency |
|---|---|---|---|---|
| **CPI** | St. Louis Fed (FRED) | `CPIAUCSL` | Consumer Price Index for All Urban Consumers | Monthly |
| **Fed Funds** | St. Louis Fed (FRED) | `FEDFUNDS` | Federal Funds Effective Rate | Monthly |
| **10Y Yield** | St. Louis Fed (FRED) | `DGS10` | 10-Year Treasury Constant Maturity Rate | Monthly |
| **Industrial Production** | St. Louis Fed (FRED) | `INDPRO` | Industrial Production Index | Monthly |
| **M2 Supply** | St. Louis Fed (FRED) | `M2SL` | M2 Money Supply | Monthly |
| **Operating Cash Flow** | Yahoo Finance | `DAL` | Delta Air Lines Operating Cash Flow | Quarterly (interpolated monthly) |

---

## 3. Econometric Methodology & Mathematical Rigor

### Stationarity Testing (ADF & KPSS)
- **Augmented Dickey-Fuller (ADF)**:
  $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \varepsilon_t$$
  $H_0: \gamma = 0$ (Unit root present / non-stationary).
- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)**:
  $y_t = \xi t + r_t + \varepsilon_t$ where $r_t = r_{t-1} + u_t$
  $H_0: \sigma_u^2 = 0$ (Trend/level stationary).
- Series differenced ($\Delta \ln Y_t$) until stationary across both tests.

### Granger Causality
Tests whether macro variable $X$ forecast-leads cash flow $Y$:
$$Y_t = c + \sum_{i=1}^p \alpha_i Y_{t-i} + \sum_{i=1}^p \beta_i X_{t-i} + \varepsilon_t$$
$F$-test on joint null $\beta_1 = \beta_2 = \dots = \beta_p = 0$.

### Forecasting Models
1. **Univariate ARIMA($p, d, q$)**: Grid search over order space minimizing Akaike Information Criterion:
   $$\text{AIC} = 2k - 2\ln(\hat{L})$$
2. **Multivariate SARIMAX($p,d,q)(P,D,Q)_s + \mathbf{\beta} \mathbf{X}_t$**: Macro regressors incorporated as exogenous drivers $\mathbf{X}_t$.
3. **Regularized Shrinkage (Ridge & ElasticNet)**:
   $$\min_{\mathbf{w}} \frac{1}{2n} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|_2^2 + \alpha \left( \rho \|\mathbf{w}\|_1 + \frac{1-\rho}{2} \|\mathbf{w}\|_2^2 \right)$$
   Fitted with expanding-window `TimeSeriesSplit(n_splits=5)`.

### Forecast Evaluation & Diebold-Mariano Test
- Metrics: RMSE, MAE, MAPE, Mean Directional Accuracy (MDA).
- **Diebold-Mariano Statistic**:
  $$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})/T}} \sim \mathcal{N}(0, 1)$$
  Evaluates statistical significance of loss differential between competing models.

---

## 4. Repository Structure

```
econometric_forecast_system/
├── data/
│   ├── fetch_fred_data.py       # Ingests CPI, FEDFUNDS, DGS10, INDPRO, M2SL, DAL cash flow
│   └── macro_corporate_dataset.parquet # Clean historical dataset
├── models/
│   ├── diagnostics.py          # ADF, KPSS, ACF/PACF, Granger causality matrix
│   ├── arima_model.py          # Auto-ARIMA & SARIMAX parameter search via AIC
│   ├── regularized_linear.py   # RidgeCV, ElasticNetCV with TimeSeriesSplit
│   └── evaluation.py           # RMSE, MAE, MAPE, MDA, Diebold-Mariano test
├── pipeline/
│   └── run_forecast.py         # End-to-end execution pipeline & Plotly HTML dashboard
├── output/
│   └── forecast_report.html    # Interactive visual report
├── requirements.txt
└── README.md
```

---

## 5. Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data ingestion
python data/fetch_fred_data.py

# 3. Execute econometric forecasting pipeline
python pipeline/run_forecast.py
```

Pipeline outputs tabular comparison to terminal and generates `output/forecast_report.html` with interactive charts.
