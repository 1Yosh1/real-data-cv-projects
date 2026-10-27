"""
cost_model.py
-------------
Full operating-cost and revenue model for airline route economics.

Implements CASM / RASM / Breakeven-Load-Factor analysis with
configurable fuel prices, load factors, and aircraft types.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.aircraft_specs import AircraftSpec

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_FUEL_PRICE: float = 2.50           # $/gal
DEFAULT_AIRPORT_FEE_PER_TONNE: float = 12.0  # $/tonne MTOW
DEFAULT_UTILIZATION_HRS_MONTH: float = 350.0  # block hours / month
RANDOM_SEED: int = 42


# ── Individual cost components ──────────────────────────────────────────────


def fuel_cost(
    block_hours: np.ndarray,
    fuel_burn_gph: float,
    fuel_price: float = DEFAULT_FUEL_PRICE,
) -> np.ndarray:
    """Fuel expense = Block Hours × Fuel Burn (gal/hr) × Fuel Price ($/gal)."""
    return np.asarray(block_hours) * fuel_burn_gph * fuel_price


def crew_cost(
    block_hours: np.ndarray,
    crew_rate: float,
) -> np.ndarray:
    """Flight & cabin crew expense = Block Hours × Crew Rate."""
    return np.asarray(block_hours) * crew_rate


def maintenance_cost(
    block_hours: np.ndarray,
    mx_rate: float,
) -> np.ndarray:
    """Maintenance reserve = Flight Hours × Reserve Rate."""
    return np.asarray(block_hours) * mx_rate


def landing_fees(
    mtow_tonnes: float,
    fee_per_tonne: float = DEFAULT_AIRPORT_FEE_PER_TONNE,
    n_routes: int = 1,
) -> np.ndarray:
    """Landing fees = MTOW (tonnes) × Airport Fee Factor.

    Applied once per flight (landing at destination).
    """
    return np.full(n_routes, mtow_tonnes * fee_per_tonne)


def lease_cost(
    block_hours: np.ndarray,
    monthly_lease: float,
    utilization: float = DEFAULT_UTILIZATION_HRS_MONTH,
) -> np.ndarray:
    """Ownership / lease cost allocated per block hour.

    Lease cost per block hour = Monthly Lease / Utilization Hours × Block Hours.
    """
    return np.asarray(block_hours) * (monthly_lease / utilization)


# ── Revenue model ───────────────────────────────────────────────────────────


def _yield_per_rpm(distance_sm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate realistic yield estimates ($/RPM) based on stage length.

    Base yield tiers:
        • Short-haul  (< 1 000 mi): $0.12/RPM
        • Medium-haul (1 000–3 000 mi): $0.08/RPM
        • Long-haul   (> 3 000 mi): $0.06/RPM

    A ±15 % random variation is applied per route.
    """
    dist = np.asarray(distance_sm, dtype=np.float64)
    base = np.where(dist < 1_000, 0.12, np.where(dist <= 3_000, 0.08, 0.06))
    variation = rng.uniform(0.85, 1.15, size=len(dist))
    return base * variation


def _simulated_load_factor(
    distance_sm: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Simulate load factors (65 %–92 %) correlated with distance band.

    Short-haul → mean ~82 %  |  Medium → mean ~79 %  |  Long → mean ~85 %
    """
    dist = np.asarray(distance_sm, dtype=np.float64)
    mean_lf = np.where(dist < 1_000, 0.82, np.where(dist <= 3_000, 0.79, 0.85))
    noise = rng.uniform(-0.07, 0.07, size=len(dist))
    lf = np.clip(mean_lf + noise, 0.65, 0.92)
    return lf


# ── Full route P&L engine ──────────────────────────────────────────────────


def compute_route_economics(
    routes_df: pd.DataFrame,
    aircraft: AircraftSpec,
    fuel_price: float = DEFAULT_FUEL_PRICE,
    load_factor_override: Optional[float] = None,
    airport_fee_per_tonne: float = DEFAULT_AIRPORT_FEE_PER_TONNE,
    utilization_hrs_month: float = DEFAULT_UTILIZATION_HRS_MONTH,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute full route-level P&L for every route in *routes_df*.

    Required input columns
    ----------------------
    distance_sm, block_hours

    Added columns
    -------------
    fuel_expense, crew_expense, mx_expense, landing_fee, lease_expense,
    total_cost, asm, yield_per_rpm, load_factor, rpm, revenue, rasm,
    casm, net_profit, operating_margin_pct, breakeven_load_factor
    """
    rng = np.random.default_rng(seed)
    df = routes_df.copy()
    n = len(df)

    bh = df["block_hours"].values
    dist = df["distance_sm"].values

    # ── Cost side ───────────────────────────────────────────────────────
    df["fuel_expense"] = fuel_cost(bh, aircraft.fuel_burn_gph, fuel_price)
    df["crew_expense"] = crew_cost(bh, aircraft.crew_cost_per_block_hr)
    df["mx_expense"] = maintenance_cost(bh, aircraft.mx_reserve_per_hr)
    df["landing_fee"] = landing_fees(aircraft.mtow_tonnes, airport_fee_per_tonne, n)
    df["lease_expense"] = lease_cost(bh, aircraft.monthly_lease_usd, utilization_hrs_month)

    df["total_cost"] = (
        df["fuel_expense"]
        + df["crew_expense"]
        + df["mx_expense"]
        + df["landing_fee"]
        + df["lease_expense"]
    )

    # ── Revenue side ────────────────────────────────────────────────────
    df["asm"] = aircraft.seats * dist  # Available Seat Miles

    df["yield_per_rpm"] = _yield_per_rpm(dist, rng)

    if load_factor_override is not None:
        df["load_factor"] = np.clip(load_factor_override, 0.0, 1.0)
    else:
        df["load_factor"] = _simulated_load_factor(dist, rng)

    df["rpm"] = df["asm"] * df["load_factor"]  # Revenue Passenger Miles
    df["revenue"] = df["rpm"] * df["yield_per_rpm"]

    # ── Performance metrics ─────────────────────────────────────────────
    df["casm"] = np.where(df["asm"] > 0, df["total_cost"] / df["asm"], np.nan)
    df["rasm"] = np.where(df["asm"] > 0, df["revenue"] / df["asm"], np.nan)

    df["net_profit"] = df["revenue"] - df["total_cost"]
    df["operating_margin_pct"] = np.where(
        df["revenue"] > 0,
        (df["net_profit"] / df["revenue"]) * 100.0,
        np.nan,
    )
    df["breakeven_load_factor"] = np.where(
        df["yield_per_rpm"] > 0,
        df["casm"] / df["yield_per_rpm"],
        np.nan,
    )

    # ── Aircraft label ──────────────────────────────────────────────────
    df["aircraft"] = aircraft.name

    return df
