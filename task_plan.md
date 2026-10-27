# CV Portfolio Projects - Task Plan

## Status: COMPLETE

### Project 1: Aviation Route Profitability Engine
- Location: `/Users/eyoas/Documents/cv aviation /aviation_profitability_engine/`
- Status: READY.
- Files:
  - `data/download_real_data.py`: OpenFlights (airports, routes) + FRED jet fuel (`DJFUELUSGULF`) to Parquet.
  - `core/distance.py`: Vectorized Haversine (SM + NM), block time calculation.
  - `core/aircraft_specs.py`: B737-800, A320neo, ATR 72-600, B787-9 dataclasses.
  - `core/cost_model.py`: Fuel, crew, maintenance, landing fees, lease, CASM, RASM, breakeven load factor.
  - `core/anomaly_detector.py`: Isolation Forest + Z-score yield anomaly detection.
  - `app/charts.py`: Plotly Scattergeo map, P&L waterfall, fuel sensitivity curves.
  - `app/main.py`: Interactive Streamlit cockpit with departure selector, sliders, tabs.
  - `requirements.txt`: Streamlit, Plotly, DuckDB, Pandas, Scikit-Learn.
  - `README.md`: Mathematical derivations, portfolio credentials, architecture.

### Project 2: Econometric Forecasting System
- Location: `/Users/eyoas/Documents/cv aviation /econometric_forecast_system/`
- Status: READY.
- Files:
  - `data/fetch_fred_data.py`: FRED CSV ingestion (`CPIAUCSL`, `FEDFUNDS`, `DGS10`, `INDPRO`, `M2SL`) + `DAL` cash flow.
  - `models/diagnostics.py`: ADF test, KPSS test, ACF/PACF, Granger causality matrix.
  - `models/arima_model.py`: AIC grid search auto-ARIMA, multivariate SARIMAX.
  - `models/regularized_linear.py`: RidgeCV, ElasticNetCV, TimeSeriesSplit, lag/rolling features.
  - `models/evaluation.py`: RMSE, MAE, MAPE, MDA, Diebold-Mariano significance test.
  - `pipeline/run_forecast.py`: End-to-end evaluation pipeline, model comparison table, Plotly HTML report.
  - `requirements.txt`: Statsmodels, Scikit-Learn, Plotly, yfinance, Pandas, Scipy.
  - `README.md`: Econometric theory, ADF/KPSS formulas, Diebold-Mariano test details.

### Project 3: Multi-Currency ETL & Cost Variance
- Location: `/Users/eyoas/Documents/cv aviation /multicurrency_cost_variance/`
- Status: READY.
- Files:
  - `etl/fetch_ecb_fx.py`: Official ECB daily rates ZIP download & tidy unpivot with holiday forward-fill.
  - `etl/ledger_generator.py`: Realistic 100,000+ multi-currency procurement ledger generator.
  - `etl/etl_pipeline.py`: DuckDB ETL, dual currency normalization (EUR/USD), DLQ routing, >=99.8% reconciliation assertion.
  - `simulation/cost_drivers.py`: 15 operational cost drivers (PERT & Log-Normal distribution models).
  - `simulation/monte_carlo.py`: Latin Hypercube 10,000 iterations, 95%/99% VaR, Spearman rank sensitivity.
  - `cli/run_audit.py`: Rich terminal dashboard with ingestion table, risk summary, sensitivity bars.
  - `tests/conftest.py`, `tests/test_etl.py`, `tests/test_simulation.py`: Complete pytest test suite.
  - `requirements.txt`: DuckDB, Rich, Scipy, Pandas, Pytest.
  - `README.md`: Mathematical specification, risk architecture, audit guide.
