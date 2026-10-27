# Multi-Currency Financial Ledger & Parametric Cost Variance Engine

**Author**: Eyoas Gizachew Zewdie  
*B.Sc. Data Analysis, University of Messina*  
*University of Messina Data Intelligence Lab Experience*

---

## 1. Executive Summary

Production ETL and financial risk simulation engine. Ingests enterprise multi-currency ledgers, normalizes against official European Central Bank (ECB) reference exchange rates via DuckDB with $\ge 99.8\%$ reconciliation assertion, and runs 10,000-iteration Latin Hypercube Monte Carlo across 15 operational cost drivers to estimate Value at Risk (VaR 95%, VaR 99%) and driver sensitivity.

---

## 2. Real Data Architecture

- **ECB Official Daily Reference Rates**:
  - Endpoint: `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip`
  - Coverage: 32 major global currencies against EUR, 1999 to present daily history.
  - Forward-filled over weekends and market holidays.
- **Enterprise Multi-Currency Ledger**:
  - $100,000+$ realistic procurement transactions across 15 operational expense categories.
  - Currency matching vendor jurisdiction (EUR, USD, GBP, JPY, CHF, CAD, AUD, CNY, SEK, NOK).

---

## 3. Financial Risk & Monte Carlo Methodology

### 15 Operational Cost Drivers

| Driver | Category | Base Budget | Distribution | Parameters |
|---|---|---|---|---|
| Aviation Fuel | Direct | €45,000,000 | PERT | Min: 0.80, Mode: 1.00, Max: 1.45 |
| Direct Labor | Direct | €32,000,000 | PERT | Min: 0.92, Mode: 1.00, Max: 1.15 |
| Aircraft Maintenance | Direct | €28,000,000 | Log-Normal | $\mu=0.00, \sigma=0.12$ |
| Airport Handling | Direct | €18,000,000 | PERT | Min: 0.88, Mode: 1.00, Max: 1.20 |
| Air Navigation Fees | Direct | €12,000,000 | PERT | Min: 0.95, Mode: 1.00, Max: 1.08 |
| IT Infrastructure | Indirect | €8,000,000 | Log-Normal | $\mu=0.00, \sigma=0.15$ |
| Software Licenses | Indirect | €5,000,000 | PERT | Min: 0.90, Mode: 1.00, Max: 1.25 |
| Commercial Insurance | Indirect | €6,000,000 | PERT | Min: 0.85, Mode: 1.00, Max: 1.35 |
| Legal Compliance | Indirect | €3,000,000 | Log-Normal | $\mu=0.00, \sigma=0.20$ |
| Facility Leases | Indirect | €7,000,000 | PERT | Min: 0.98, Mode: 1.00, Max: 1.05 |
| Marketing | Indirect | €9,000,000 | PERT | Min: 0.70, Mode: 1.00, Max: 1.40 |
| Customer Care | Indirect | €6,000,000 | Log-Normal | $\mu=0.00, \sigma=0.10$ |
| Telecom | Indirect | €2,000,000 | PERT | Min: 0.90, Mode: 1.00, Max: 1.15 |
| Admin Staff | Indirect | €4,000,000 | PERT | Min: 0.95, Mode: 1.00, Max: 1.10 |
| Financing Fees | Indirect | €5,000,000 | Log-Normal | $\mu=0.00, \sigma=0.18$ |

### Simulation Engine
- **Sampling**: Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube(d=15)`) over uniform space.
- **PERT Transform**: Quantile mapped to generalized Beta distribution:
  $$\alpha = 1 + 4\frac{\text{mode}-\text{min}}{\text{max}-\text{min}}, \quad \beta = 1 + 4\frac{\text{max}-\text{mode}}{\text{max}-\text{min}}$$
- **Value at Risk (VaR)**:
  $$\text{VaR}_\alpha = Q_\alpha(\text{Total Budget})$$
  Evaluated at $\alpha = 0.95$ and $\alpha = 0.99$.
- **Sensitivity Ranking**: Spearman rank correlation $\rho_s(X_i, Y)$ between driver $X_i$ and total budget outcome $Y$.

---

## 4. Repository Layout

```
multicurrency_cost_variance/
├── etl/
│   ├── fetch_ecb_fx.py         # Ingests and melts ECB daily rates archive
│   ├── ledger_generator.py     # Generates multi-currency enterprise procurement ledger
│   └── etl_pipeline.py         # DuckDB ETL with DLQ and 99.8% reconciliation assertion
├── simulation/
│   ├── cost_drivers.py         # 15 operational cost driver specifications
│   └── monte_carlo.py          # Latin Hypercube 10k simulation, VaR 95/99, Spearman sensitivity
├── cli/
│   └── run_audit.py            # Rich terminal dashboard
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_etl.py             # ETL reconciliation and DLQ unit tests
│   └── test_simulation.py      # Monte Carlo, VaR ordering, and sensitivity unit tests
├── requirements.txt
└── README.md
```

---

## 5. Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run unit tests
pytest tests/ -v

# 3. Execute complete audit dashboard
python cli/run_audit.py
```
