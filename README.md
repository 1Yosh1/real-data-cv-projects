# Real-Data CV Projects & Quantitative Engines

Portfolio assets for **Eyoas Gizachew Zewdie** (B.Sc. Data Analysis, University of Messina).  
100% verified, free, open-source real-world data sources (zero fake dummy placeholders).

---

## Projects Overview

| # | Project | Stack | Real Data Source | Core Capabilities |
|---|---|---|---|---|
| **1** | [**Aviation Route Profitability & Cost Engine**](aviation_profitability_engine/) | Python, DuckDB, Plotly, Streamlit | **OpenFlights** + **FRED Jet Fuel Spot** | CASM, RASM, breakeven load factor, Isolation Forest pricing anomaly detection, interactive Streamlit cockpit. |
| **2** | [**Multivariate Econometric Forecasting System**](econometric_forecast_system/) | Python, Statsmodels, Scikit-Learn, Plotly | **FRED Federal Reserve API** + **Yahoo Finance (DAL)** | ADF/KPSS stationarity, Granger causality, Auto-ARIMA, multivariate SARIMAX, Ridge/ElasticNet, Diebold-Mariano test. |
| **3** | [**Multi-Currency Transaction ETL & Cost Variance**](multicurrency_cost_variance/) | Python, DuckDB, Scipy, Rich, Pytest | **European Central Bank (ECB) FX Archives** | Dual-currency normalization (EUR/USD), $\ge 99.8\%$ reconciliation assertion, DLQ routing, 10,000-iteration Latin Hypercube Monte Carlo, 95%/99% VaR, Rich audit cockpit. |

---

## 1. Aviation Route Profitability Engine
- Stage length calculation via vectorized Haversine formula (Statute Miles & Nautical Miles).
- OEM performance models: Boeing 737-800, Airbus A320neo, ATR 72-600, Boeing 787-9.
- Cost engine: Fuel burn, crew, maintenance reserve, landing fees, monthly lease amortization.
- Pricing anomaly scanner: Isolation Forest ($\text{contamination}=0.03$) and $3.5\sigma$ yield filter.
- Streamlit visualizer with Plotly map and financial P&L waterfall chart.

## 2. Multivariate Econometric Forecasting System
- Macro drivers: CPI (`CPIAUCSL`), Fed Funds (`FEDFUNDS`), 10Y Treasury (`DGS10`), Industrial Production (`INDPRO`), M2 Supply (`M2SL`).
- Corporate benchmark: Delta Air Lines (`DAL`) operating cash flow.
- Rigorous statistical diagnostics: ADF, KPSS, ACF/PACF, Granger causality matrix.
- Models: Univariate ARIMA ($AIC$ minimization), Multivariate SARIMAX, RidgeCV, ElasticNetCV.
- Forecast validation: Walk-forward `TimeSeriesSplit`, Diebold-Mariano hypothesis test.

## 3. Multi-Currency ETL & Parametric Cost Variance Engine
- ECB reference rates: Historical daily exchange rates for 32 major currencies (1999–present).
- Enterprise ledger: 100,000+ multi-currency procurement records across 15 operational cost categories.
- DuckDB conversion engine with Dead Letter Queue (DLQ) and $\ge 99.8\%$ reconciliation test.
- Monte Carlo engine: Latin Hypercube sampling across 15 PERT and Log-Normal distributions.
- Risk metrics: Value at Risk (VaR 95%, VaR 99%) and Spearman rank tornado sensitivity.
