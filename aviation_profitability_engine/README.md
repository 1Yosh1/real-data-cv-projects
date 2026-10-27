# ✈️ Aviation Route Profitability & Cost Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-grade data-engineering and analytics platform that models airline
route-level economics using real-world aviation data, OEM aircraft specifications,
and FRED fuel-price feeds.  Built as a portfolio project demonstrating end-to-end
data pipeline design, financial modelling, and interactive visualisation.

---

## 📋 Table of Contents

- [Features](#-features)
- [Data Sources](#-data-sources)
- [Architecture](#-architecture)
- [Formulas & Methodology](#-formulas--methodology)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Author](#-author)

---

## ✨ Features

| Capability | Description |
|---|---|
| **Real Data Ingestion** | Automated download of OpenFlights airports/routes and FRED jet-fuel prices. DuckDB-powered parsing into Parquet. |
| **Vectorised Distance Engine** | NumPy Haversine great-circle computation for statute & nautical miles plus block-time estimation. |
| **OEM Aircraft Library** | Frozen dataclass specs for B737-800, A320neo, ATR 72-600, and B787-9 with fuel burn, crew cost, MX reserves, and lease rates. |
| **Full Route P&L Model** | CASM, RASM, Breakeven Load Factor, Net Profit, and Operating Margin computed per route. |
| **Anomaly Detection** | Isolation Forest + Z-score flagging for predatory pricing, exorbitant yield outliers, and multivariate anomalies. |
| **Interactive Dashboard** | Streamlit app with Plotly globe map, P&L waterfall, fuel sensitivity analysis, and CASM histograms. |

---

## 📡 Data Sources

| Dataset | Source | URL |
|---|---|---|
| Airports | OpenFlights | https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat |
| Routes | OpenFlights | https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat |
| Jet Fuel Prices | FRED (DJFUELUSGULF) | https://fred.stlouisfed.org/graph/fredgraph.csv?id=DJFUELUSGULF |

---

## 🏗️ Architecture

```
aviation_profitability_engine/
├── data/
│   ├── download_real_data.py   # ETL pipeline: download → parse → Parquet
│   └── fuel_prices.parquet     # Generated at runtime
├── core/
│   ├── distance.py             # Vectorised Haversine & block-hour calculator
│   ├── aircraft_specs.py       # OEM specifications (dataclasses)
│   ├── cost_model.py           # CASM / RASM / Breakeven LF / Route P&L
│   └── anomaly_detector.py     # Isolation Forest & Z-score anomaly flags
├── app/
│   ├── main.py                 # Streamlit UI entry point
│   └── charts.py               # Plotly chart factories
├── requirements.txt
└── README.md
```

**Data flow:**

```
OpenFlights + FRED  ──►  download_real_data.py  ──►  Parquet files
                                                         │
                         ┌───────────────────────────────┘
                         ▼
               distance.py  →  cost_model.py  →  anomaly_detector.py
                                                         │
                         ┌───────────────────────────────┘
                         ▼
                    charts.py  →  main.py (Streamlit)
```

---

## 📐 Formulas & Methodology

### Haversine Distance

$$d = 2R \arcsin\!\left(\sqrt{\sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos\varphi_1 \cos\varphi_2 \sin^2\!\left(\frac{\Delta\lambda}{2}\right)}\right)$$

where $R_{SM} = 3{,}958.8$ mi and $R_{NM} = 3{,}440.065$ nm.

### Block Time

$$T_{block} = T_{taxi} + \frac{d}{V_{cruise} \times 1.15078}$$

where $T_{taxi} = 0.35$ h and $V_{cruise}$ is in KTAS.

### Cost Components

| Component | Formula |
|---|---|
| Fuel | $\text{Block Hours} \times \text{Burn (gal/hr)} \times \text{Price (\$/gal)}$ |
| Crew | $\text{Block Hours} \times \text{Crew Rate}$ |
| Maintenance | $\text{Block Hours} \times \text{MX Reserve Rate}$ |
| Landing Fees | $\text{MTOW (tonnes)} \times \$12/\text{tonne}$ |
| Lease | $\frac{\text{Monthly Lease}}{\text{Utilisation (350 hr/mo)}} \times \text{Block Hours}$ |

### Performance Metrics

| Metric | Formula |
|---|---|
| ASM | $\text{Seats} \times \text{Stage Length (SM)}$ |
| CASM | $\text{Total Cost} / \text{ASM}$ |
| RASM | $\text{Revenue} / \text{ASM}$ |
| Breakeven LF | $\text{CASM} / \text{Yield per RPM}$ |
| Operating Margin | $(\text{Revenue} - \text{Cost}) / \text{Revenue} \times 100$ |

### Yield Model

| Stage Length | Base Yield ($/RPM) |
|---|---|
| < 1 000 mi (short-haul) | $0.12 |
| 1 000 – 3 000 mi (medium) | $0.08 |
| > 3 000 mi (long-haul) | $0.06 |

Each route receives a ±15 % random variation.

### Anomaly Detection

- **Isolation Forest**: `contamination=0.03` on `(stage_length, yield_per_rpm, load_factor)`.
- **Z-score**: flag routes with yield > 3.5σ from stage-length-binned median.
- **Predatory pricing**: yield below cash operating cost per RPM.

---

## ⚙️ Setup & Installation

```bash
# 1. Clone or navigate to the project directory
cd aviation_profitability_engine

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download and prepare data
python -m data.download_real_data

# 5. Launch the dashboard
streamlit run app/main.py
```

---

## 🚀 Usage

1. **Select a departure airport** from the sidebar dropdown (searchable, IATA – City format).
2. **Choose an aircraft model** to apply its cost structure.
3. **Adjust fuel price** with the sensitivity slider ($1.50 – $4.50/gal).
4. **Override load factor** (optional) to test profitability scenarios.
5. Explore the four analysis tabs:
   - 🗺️ **Route Map** — colour-coded by operating margin.
   - 📊 **Route Economics** — sortable P&L table + CASM histogram + waterfall.
   - 🚨 **Anomaly Scanner** — flagged routes with anomaly type labels.
   - ⛽ **Fuel Sensitivity** — profit sensitivity to fuel-price changes + historical trend.

---

## 📸 Screenshots

> _Screenshots will be added after the first run of the dashboard._

| Route Map | P&L Waterfall | Fuel Sensitivity |
|---|---|---|
| _(placeholder)_ | _(placeholder)_ | _(placeholder)_ |

---

## 👤 Author

**Eyoas Gizachew Zewdie**
B.Sc. Data Analysis — University of Messina

---

> Built with Python · Streamlit · Plotly · DuckDB · scikit-learn
