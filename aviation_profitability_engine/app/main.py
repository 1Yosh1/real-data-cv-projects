"""
main.py
-------
Streamlit interactive dashboard for the Aviation Route Profitability Engine.

Run with:
    streamlit run app/main.py

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import sys
import pathlib

# ── Ensure project root is on sys.path so `core.*` imports resolve ──────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from core.aircraft_specs import FLEET_CATALOG, get_aircraft, list_aircraft_names
from core.distance import enrich_routes_with_distance
from core.cost_model import compute_route_economics
from core.anomaly_detector import detect_anomalies
from app.charts import (
    plot_route_map,
    plot_waterfall,
    plot_sensitivity,
    plot_casm_distribution,
)

DATA_DIR = PROJECT_ROOT / "data"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aviation Route Profitability Engine",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data loading (cached) ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading airport data …")
def load_airports() -> pd.DataFrame:
    """Load airports from Parquet (or trigger download if missing)."""
    parquet_path = DATA_DIR / "airports.parquet"
    if not parquet_path.exists():
        from data.download_real_data import run_pipeline
        run_pipeline(DATA_DIR)
    return pd.read_parquet(parquet_path)


@st.cache_data(show_spinner="Loading route data …")
def load_route_geo() -> pd.DataFrame:
    """Load route geography from Parquet (or trigger download)."""
    parquet_path = DATA_DIR / "route_geography.parquet"
    if not parquet_path.exists():
        from data.download_real_data import run_pipeline
        run_pipeline(DATA_DIR)
    return pd.read_parquet(parquet_path)


@st.cache_data(show_spinner="Loading fuel prices …")
def load_fuel_prices() -> pd.DataFrame:
    """Load FRED fuel price history."""
    parquet_path = DATA_DIR / "fuel_prices.parquet"
    if not parquet_path.exists():
        from data.download_real_data import run_pipeline
        run_pipeline(DATA_DIR)
    return pd.read_parquet(parquet_path)


def get_latest_fuel_price(fuel_df: pd.DataFrame) -> float:
    """Return the most recent fuel price observation, or a default."""
    if fuel_df.empty:
        return 2.50
    return float(fuel_df.sort_values("date").iloc[-1]["price_per_gallon"])


# ── Sidebar controls ───────────────────────────────────────────────────────

def sidebar_controls(
    airports_df: pd.DataFrame,
    route_geo_df: pd.DataFrame,
    latest_fuel: float,
) -> dict:
    """Render sidebar widgets and return a dict of user selections."""
    st.sidebar.header("✈️ Route Parameters")

    # Departure airport
    available_iatas = sorted(route_geo_df["source_iata"].unique())
    airport_labels = {}
    for iata in available_iatas:
        match = airports_df.loc[airports_df["iata"] == iata]
        if not match.empty:
            city = match.iloc[0]["city"]
            airport_labels[iata] = f"{iata} — {city}"
        else:
            airport_labels[iata] = iata

    label_list = [airport_labels[c] for c in available_iatas]
    default_idx = 0
    # Try to default to a major hub
    for preferred in ["ATL", "JFK", "LHR", "DXB", "LAX"]:
        if preferred in available_iatas:
            default_idx = available_iatas.index(preferred)
            break

    selected_label = st.sidebar.selectbox(
        "Departure Airport",
        label_list,
        index=default_idx,
    )
    selected_iata = selected_label.split(" — ")[0]

    # Aircraft
    aircraft_name = st.sidebar.selectbox(
        "Aircraft Model",
        list_aircraft_names(),
        index=0,
    )

    # Fuel price
    fuel_price = st.sidebar.slider(
        "Fuel Price ($/gal)",
        min_value=1.50,
        max_value=4.50,
        value=min(max(latest_fuel, 1.50), 4.50),
        step=0.05,
        format="$%.2f",
    )

    # Load factor override
    use_lf_override = st.sidebar.checkbox("Override Load Factor", value=False)
    lf_override = None
    if use_lf_override:
        lf_override = st.sidebar.slider(
            "Load Factor (%)",
            min_value=50,
            max_value=100,
            value=80,
            step=1,
        ) / 100.0

    return {
        "departure_iata": selected_iata,
        "aircraft_name": aircraft_name,
        "fuel_price": fuel_price,
        "load_factor_override": lf_override,
    }


# ── Computation pipeline ───────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing route economics …")
def run_economics(
    route_geo_json: str,
    aircraft_name: str,
    fuel_price: float,
    load_factor_override: float | None,
) -> pd.DataFrame:
    """End-to-end computation: distances → costs → anomalies.

    We accept a JSON string instead of a DataFrame so Streamlit can hash it.
    """
    route_geo_df = pd.read_json(route_geo_json, orient="records")
    aircraft = get_aircraft(aircraft_name)

    # Enrich with distance & block time
    enriched = enrich_routes_with_distance(
        route_geo_df,
        cruise_speed_ktas=aircraft.cruise_speed_ktas,
    )

    # Compute P&L
    econ = compute_route_economics(
        enriched,
        aircraft=aircraft,
        fuel_price=fuel_price,
        load_factor_override=load_factor_override,
    )

    # Stash fuel burn rate for sensitivity chart
    econ["_fuel_burn_gph"] = aircraft.fuel_burn_gph

    # Anomaly detection
    result = detect_anomalies(econ)

    return result


# ── Main app ────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Streamlit application."""
    st.title("✈️ Aviation Route Profitability & Cost Engine")
    st.caption("Analyse airline route economics with real-world data & OEM aircraft specs.")

    # Load data
    airports_df = load_airports()
    route_geo_df = load_route_geo()
    fuel_df = load_fuel_prices()
    latest_fuel = get_latest_fuel_price(fuel_df)

    # Sidebar
    params = sidebar_controls(airports_df, route_geo_df, latest_fuel)

    # Filter routes for selected departure
    dep_routes = route_geo_df[
        route_geo_df["source_iata"] == params["departure_iata"]
    ].reset_index(drop=True)

    if dep_routes.empty:
        st.warning(f"No routes found departing from **{params['departure_iata']}**.")
        return

    st.sidebar.metric("Routes from hub", len(dep_routes))

    # Run economics
    econ_df = run_economics(
        dep_routes.to_json(orient="records"),
        params["aircraft_name"],
        params["fuel_price"],
        params["load_factor_override"],
    )

    # ── Tabs ────────────────────────────────────────────────────────────
    tab_map, tab_econ, tab_anomaly, tab_fuel = st.tabs(
        ["🗺️ Route Map", "📊 Route Economics", "🚨 Anomaly Scanner", "⛽ Fuel Sensitivity"]
    )

    # Tab 1 — Route Map
    with tab_map:
        st.subheader("Global Flight Path Visualiser")
        map_fig = plot_route_map(econ_df)
        st.plotly_chart(map_fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        profitable = (econ_df["net_profit"] > 0).sum()
        col1.metric("Profitable Routes", f"{profitable} / {len(econ_df)}")
        col2.metric("Avg Margin", f"{econ_df['operating_margin_pct'].mean():.1f}%")
        col3.metric("Avg CASM", f"${econ_df['casm'].mean():.4f}")

    # Tab 2 — Route Economics
    with tab_econ:
        st.subheader("Route-Level Economics")

        display_cols = [
            "source_iata",
            "dest_iata",
            "distance_sm",
            "block_hours",
            "casm",
            "rasm",
            "operating_margin_pct",
            "net_profit",
            "load_factor",
            "breakeven_load_factor",
        ]
        display_df = econ_df[display_cols].copy()
        display_df.columns = [
            "Origin",
            "Destination",
            "Distance (SM)",
            "Block Hours",
            "CASM",
            "RASM",
            "Op. Margin %",
            "Net Profit ($)",
            "Load Factor",
            "Breakeven LF",
        ]
        display_df = display_df.sort_values("Op. Margin %", ascending=False)

        st.dataframe(
            display_df.style.format(
                {
                    "Distance (SM)": "{:,.0f}",
                    "Block Hours": "{:.2f}",
                    "CASM": "${:.4f}",
                    "RASM": "${:.4f}",
                    "Op. Margin %": "{:+.1f}%",
                    "Net Profit ($)": "${:,.0f}",
                    "Load Factor": "{:.1%}",
                    "Breakeven LF": "{:.1%}",
                }
            ),
            use_container_width=True,
            height=400,
        )

        # CASM distribution
        st.plotly_chart(plot_casm_distribution(econ_df), use_container_width=True)

        # Waterfall for selected route
        st.subheader("Route P&L Waterfall")
        route_options = [
            f"{row['source_iata']} → {row['dest_iata']}"
            for _, row in econ_df.iterrows()
        ]
        if route_options:
            selected_route = st.selectbox("Select a route", route_options, index=0)
            route_idx = route_options.index(selected_route)
            route_row = econ_df.iloc[route_idx]
            st.plotly_chart(plot_waterfall(route_row), use_container_width=True)

    # Tab 3 — Anomaly Scanner
    with tab_anomaly:
        st.subheader("Anomalous Routes")
        anomalies = econ_df[econ_df["is_anomaly"]].copy()
        if anomalies.empty:
            st.success("No anomalous routes detected for this configuration.")
        else:
            anom_cols = [
                "source_iata",
                "dest_iata",
                "distance_sm",
                "yield_per_rpm",
                "load_factor",
                "operating_margin_pct",
                "net_profit",
                "anomaly_type",
            ]
            anom_display = anomalies[anom_cols].copy()
            anom_display.columns = [
                "Origin",
                "Destination",
                "Distance (SM)",
                "Yield ($/RPM)",
                "Load Factor",
                "Op. Margin %",
                "Net Profit ($)",
                "Anomaly Type",
            ]
            st.dataframe(
                anom_display.style.format(
                    {
                        "Distance (SM)": "{:,.0f}",
                        "Yield ($/RPM)": "${:.4f}",
                        "Load Factor": "{:.1%}",
                        "Op. Margin %": "{:+.1f}%",
                        "Net Profit ($)": "${:,.0f}",
                    }
                ),
                use_container_width=True,
                height=350,
            )
            st.metric("Anomalous Routes", len(anomalies))

    # Tab 4 — Fuel Sensitivity
    with tab_fuel:
        st.subheader("Fuel-Price Sensitivity Analysis")
        fuel_route_options = [
            f"{row['source_iata']} → {row['dest_iata']}"
            for _, row in econ_df.iterrows()
        ]
        if fuel_route_options:
            selected_fuel_route = st.selectbox(
                "Select a route for sensitivity analysis",
                fuel_route_options,
                index=0,
                key="fuel_route",
            )
            fuel_route_idx = fuel_route_options.index(selected_fuel_route)
            fuel_route_row = econ_df.iloc[fuel_route_idx]
            fuel_range = np.arange(1.50, 4.55, 0.10)
            st.plotly_chart(
                plot_sensitivity(fuel_route_row, fuel_range),
                use_container_width=True,
            )

            # Summary stats
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Current Fuel Cost",
                f"${fuel_route_row['fuel_expense']:,.0f}",
            )
            pct_fuel = (
                fuel_route_row["fuel_expense"] / fuel_route_row["total_cost"] * 100
                if fuel_route_row["total_cost"] > 0
                else 0
            )
            col_b.metric("Fuel % of Total Cost", f"{pct_fuel:.1f}%")
            col_c.metric(
                "Current Profit",
                f"${fuel_route_row['net_profit']:,.0f}",
            )

        # Historical fuel trend
        st.subheader("Historical Jet Fuel Price (FRED)")
        if not fuel_df.empty:
            import plotly.express as px

            fuel_fig = px.line(
                fuel_df,
                x="date",
                y="price_per_gallon",
                title="U.S. Gulf Coast Jet Fuel Spot Price",
                labels={"date": "Date", "price_per_gallon": "$/gallon"},
            )
            fuel_fig.update_layout(height=350)
            st.plotly_chart(fuel_fig, use_container_width=True)


if __name__ == "__main__":
    main()
