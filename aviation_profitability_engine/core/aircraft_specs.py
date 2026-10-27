"""
aircraft_specs.py
-----------------
OEM-grade aircraft specifications stored as typed dataclasses.

Each aircraft record contains the performance, cost, and capacity
parameters needed by the route cost model.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class AircraftSpec:
    """Immutable specification sheet for a single aircraft type."""

    name: str
    seats: int
    fuel_burn_gph: float          # gallons per block hour
    crew_cost_per_block_hr: float # USD per block hour (flight + cabin crew)
    mx_reserve_per_hr: float      # maintenance reserve USD per flight hour
    cruise_speed_ktas: float      # true airspeed in knots
    mtow_tonnes: float            # maximum take-off weight (metric tonnes)
    monthly_lease_usd: float      # monthly dry-lease rate


# ── Fleet catalogue ─────────────────────────────────────────────────────────

B737_800 = AircraftSpec(
    name="Boeing 737-800",
    seats=162,
    fuel_burn_gph=850.0,
    crew_cost_per_block_hr=1_850.0,
    mx_reserve_per_hr=320.0,
    cruise_speed_ktas=453.0,
    mtow_tonnes=79.0,
    monthly_lease_usd=340_000.0,
)

A320NEO = AircraftSpec(
    name="Airbus A320neo",
    seats=180,
    fuel_burn_gph=680.0,
    crew_cost_per_block_hr=1_900.0,
    mx_reserve_per_hr=280.0,
    cruise_speed_ktas=447.0,
    mtow_tonnes=79.0,
    monthly_lease_usd=380_000.0,
)

ATR72_600 = AircraftSpec(
    name="ATR 72-600",
    seats=72,
    fuel_burn_gph=240.0,
    crew_cost_per_block_hr=1_100.0,
    mx_reserve_per_hr=220.0,
    cruise_speed_ktas=275.0,
    mtow_tonnes=23.0,
    monthly_lease_usd=140_000.0,
)

B787_9 = AircraftSpec(
    name="Boeing 787-9",
    seats=296,
    fuel_burn_gph=1_400.0,
    crew_cost_per_block_hr=2_800.0,
    mx_reserve_per_hr=450.0,
    cruise_speed_ktas=488.0,
    mtow_tonnes=254.0,
    monthly_lease_usd=1_050_000.0,
)

# ── Lookup helpers ──────────────────────────────────────────────────────────

FLEET_CATALOG: Dict[str, AircraftSpec] = {
    "Boeing 737-800": B737_800,
    "Airbus A320neo": A320NEO,
    "ATR 72-600": ATR72_600,
    "Boeing 787-9": B787_9,
}


def get_aircraft(name: str) -> AircraftSpec:
    """Return an AircraftSpec by its display name.

    Raises
    ------
    KeyError
        If *name* is not in the fleet catalogue.
    """
    if name not in FLEET_CATALOG:
        available = ", ".join(sorted(FLEET_CATALOG))
        raise KeyError(
            f"Unknown aircraft '{name}'. Available: {available}"
        )
    return FLEET_CATALOG[name]


def list_aircraft_names() -> list[str]:
    """Return a sorted list of available aircraft display names."""
    return sorted(FLEET_CATALOG.keys())
