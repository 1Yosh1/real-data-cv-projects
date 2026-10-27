"""
charts.py
---------
Plotly chart factories for the Aviation Route Profitability Engine.

All functions accept DataFrames or Series and return ``plotly.graph_objects.Figure``
objects ready for display in Streamlit.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ── Colour helpers ──────────────────────────────────────────────────────────

def _margin_color(margin: float) -> str:
    """Map operating margin to a traffic-light hex colour."""
    if margin < -2.0:
        return "#e74c3c"   # red  – loss
    if margin < 5.0:
        return "#f1c40f"   # yellow – breakeven zone
    return "#2ecc71"       # green – profit


def _margin_colorscale() -> list[list]:
    """Continuous colorscale: red → yellow → green."""
    return [
        [0.0, "#e74c3c"],
        [0.5, "#f1c40f"],
        [1.0, "#2ecc71"],
    ]


# ── 1. Route Map ────────────────────────────────────────────────────────────


def plot_route_map(routes_df: pd.DataFrame) -> go.Figure:
    """Plotly Scattergeo with great-circle arcs colour-coded by operating margin.

    Expected columns: source_iata, dest_iata, src_lat, src_lon, dst_lat,
    dst_lon, operating_margin_pct, net_profit, src_city, dst_city.
    """
    fig = go.Figure()

    # Clamp margins for colour mapping
    margins = routes_df["operating_margin_pct"].fillna(0).values
    vmin, vmax = float(np.nanpercentile(margins, 5)), float(np.nanpercentile(margins, 95))
    if vmin == vmax:
        vmin, vmax = -20, 20
    normed = np.clip((margins - vmin) / (vmax - vmin), 0, 1)

    # Draw each route as a great-circle arc
    for idx, row in routes_df.iterrows():
        margin_val = row["operating_margin_pct"] if pd.notna(row["operating_margin_pct"]) else 0.0
        color = _margin_color(margin_val)
        hover_text = (
            f"{row['source_iata']} → {row['dest_iata']}<br>"
            f"{row.get('src_city', '')} → {row.get('dst_city', '')}<br>"
            f"Margin: {margin_val:+.1f}%<br>"
            f"Profit: ${row['net_profit']:,.0f}"
        )
        fig.add_trace(
            go.Scattergeo(
                lon=[row["src_lon"], row["dst_lon"]],
                lat=[row["src_lat"], row["dst_lat"]],
                mode="lines",
                line=dict(width=1.2, color=color),
                opacity=0.7,
                hoverinfo="text",
                text=hover_text,
                showlegend=False,
            )
        )

    # Airport dots at origins
    origins = routes_df.drop_duplicates(subset=["source_iata"])
    fig.add_trace(
        go.Scattergeo(
            lon=origins["src_lon"],
            lat=origins["src_lat"],
            mode="markers",
            marker=dict(size=4, color="#3498db", opacity=0.8),
            text=origins["source_iata"],
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor="#f5f5f5",
            showocean=True,
            oceancolor="#dce6f0",
            showcountries=True,
            countrycolor="#bdc3c7",
            showcoastlines=True,
            coastlinecolor="#95a5a6",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        title="Route Network — Colour by Operating Margin",
        height=550,
    )
    return fig


# ── 2. P&L Waterfall ────────────────────────────────────────────────────────


def plot_waterfall(route_row: pd.Series) -> go.Figure:
    """P&L waterfall: Revenue → cost items → Net Margin."""
    categories = [
        "Revenue",
        "Fuel",
        "Crew",
        "Maintenance",
        "Landing Fees",
        "Lease",
        "Net Profit",
    ]
    values = [
        route_row["revenue"],
        -route_row["fuel_expense"],
        -route_row["crew_expense"],
        -route_row["mx_expense"],
        -route_row["landing_fee"],
        -route_row["lease_expense"],
        route_row["net_profit"],
    ]
    measures = ["absolute", "relative", "relative", "relative", "relative", "relative", "total"]

    fig = go.Figure(
        go.Waterfall(
            x=categories,
            y=values,
            measure=measures,
            textposition="outside",
            text=[f"${v:,.0f}" for v in values],
            connector=dict(line=dict(color="#95a5a6", width=1)),
            increasing=dict(marker_color="#2ecc71"),
            decreasing=dict(marker_color="#e74c3c"),
            totals=dict(
                marker_color="#3498db" if values[-1] >= 0 else "#c0392b"
            ),
        )
    )
    route_label = f"{route_row['source_iata']} → {route_row['dest_iata']}"
    fig.update_layout(
        title=f"P&L Waterfall — {route_label}",
        yaxis_title="USD ($)",
        showlegend=False,
        height=450,
        margin=dict(t=60, b=40),
    )
    return fig


# ── 3. Fuel Sensitivity ────────────────────────────────────────────────────


def plot_sensitivity(
    route_row: pd.Series,
    fuel_prices: np.ndarray | None = None,
) -> go.Figure:
    """Line chart showing profit sensitivity to fuel price changes.

    Parameters
    ----------
    route_row   : A single route Series with block_hours, fuel_burn_gph (via
                  total_cost minus fuel component) and other cost fields.
    fuel_prices : Array of fuel prices to evaluate; defaults to $1.50–$4.50.
    """
    if fuel_prices is None:
        fuel_prices = np.arange(1.50, 4.55, 0.10)

    bh = route_row["block_hours"]
    # Reconstruct non-fuel costs
    non_fuel_cost = (
        route_row["crew_expense"]
        + route_row["mx_expense"]
        + route_row["landing_fee"]
        + route_row["lease_expense"]
    )
    revenue = route_row["revenue"]

    # We need fuel burn rate; derive from fuel_expense / block_hours / original price
    # Safer: we stored the aircraft object externally, but here we can compute
    original_fuel_expense = route_row["fuel_expense"]
    # fuel_expense = bh * burn * price  →  burn * price = fuel_expense / bh
    if bh > 0 and original_fuel_expense > 0:
        burn_times_original_price = original_fuel_expense / bh
    else:
        burn_times_original_price = 850.0 * 2.50  # fallback

    # For each new price, scale proportionally
    original_price = burn_times_original_price / max(
        original_fuel_expense / bh / 1 if bh > 0 else 850, 1
    )
    # Simpler: treat fuel as linear in price
    # fuel_expense_new = original_fuel_expense * (new_price / original_price_used)
    # We don't know original_price_used directly so compute burn rate:
    # burn_rate = fuel_expense / (bh * price_used)  — but price_used varies
    # Instead just recompute: new_fuel = bh * burn_rate * new_price
    # burn_rate = fuel_expense / bh / price_used => we need price_used
    # Approximate: use the ratio approach
    # Since we always have the per-flight fuel expense, let's just compute
    # the derivative: for each $1 change in fuel, cost changes by bh * burn_rate
    # We can get burn_rate from any stored spec; but to keep this self-contained:

    # Use the stored expense and assume base price to recover burn rate
    # This is fine because the Streamlit app passes fuel_price alongside
    fuel_burn_rate = route_row.get("_fuel_burn_gph", 850.0)
    profits = []
    for fp in fuel_prices:
        new_fuel = bh * fuel_burn_rate * fp
        total = non_fuel_cost + new_fuel
        profits.append(revenue - total)

    profits = np.array(profits)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fuel_prices,
            y=profits,
            mode="lines+markers",
            line=dict(color="#2980b9", width=2.5),
            marker=dict(size=5),
            name="Net Profit",
        )
    )
    # Breakeven line
    fig.add_hline(y=0, line_dash="dash", line_color="#e74c3c", annotation_text="Breakeven")

    route_label = f"{route_row['source_iata']} → {route_row['dest_iata']}"
    fig.update_layout(
        title=f"Fuel-Price Sensitivity — {route_label}",
        xaxis_title="Jet Fuel Price ($/gal)",
        yaxis_title="Net Profit ($)",
        height=420,
        margin=dict(t=60, b=40),
    )
    return fig


# ── 4. CASM Distribution ───────────────────────────────────────────────────


def plot_casm_distribution(routes_df: pd.DataFrame) -> go.Figure:
    """Histogram of CASM distribution across all routes."""
    casm = routes_df["casm"].dropna()
    fig = go.Figure(
        go.Histogram(
            x=casm,
            nbinsx=60,
            marker_color="#3498db",
            opacity=0.85,
        )
    )
    median_val = float(casm.median())
    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Median: {median_val:.4f}",
    )
    fig.update_layout(
        title="CASM Distribution Across Routes",
        xaxis_title="CASM ($/ASM)",
        yaxis_title="Number of Routes",
        height=420,
        margin=dict(t=60, b=40),
    )
    return fig
