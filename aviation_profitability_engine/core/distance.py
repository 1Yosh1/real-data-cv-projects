"""
distance.py
-----------
Vectorized Haversine distance and block-hour calculations for aviation
route analysis.

All heavy math uses NumPy broadcasting so entire route tables can be
processed in a single call without Python-level loops.

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ───────────────────────────────────────────────────────────────
EARTH_RADIUS_SM = 3_958.8  # statute miles
EARTH_RADIUS_NM = 3_440.065  # nautical miles
EARTH_RADIUS_KM = 6_371.0  # kilometres

DEFAULT_TAXI_TIME_HR = 0.35  # average ground taxi time (hours)


def haversine_sm(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Return great-circle distance in **statute miles** (vectorized).

    Parameters
    ----------
    lat1, lon1 : array-like   Origin coordinates in decimal degrees.
    lat2, lon2 : array-like   Destination coordinates in decimal degrees.

    Returns
    -------
    np.ndarray  Distance in statute miles.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_SM * c


def haversine_nm(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Return great-circle distance in **nautical miles** (vectorized)."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=np.float64)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_NM * c


def block_time(
    distance_sm: np.ndarray,
    cruise_speed_ktas: float,
    taxi_time_hr: float = DEFAULT_TAXI_TIME_HR,
) -> np.ndarray:
    """Compute block time (hours) = taxi time + cruise time.

    Parameters
    ----------
    distance_sm      : Stage length in statute miles.
    cruise_speed_ktas: True airspeed in knots (1 kt ≈ 1.15078 sm/h).
    taxi_time_hr     : Fixed taxi-out + taxi-in time (default 0.35 h).

    Returns
    -------
    np.ndarray  Block time in hours.
    """
    cruise_speed_smph = cruise_speed_ktas * 1.15078  # convert knots → statute mph
    cruise_time = np.asarray(distance_sm, dtype=np.float64) / cruise_speed_smph
    return taxi_time_hr + cruise_time


def enrich_routes_with_distance(
    routes_df: pd.DataFrame,
    cruise_speed_ktas: float,
    taxi_time_hr: float = DEFAULT_TAXI_TIME_HR,
) -> pd.DataFrame:
    """Add distance and block-time columns to a route-geography DataFrame.

    Expected input columns: src_lat, src_lon, dst_lat, dst_lon.

    Added columns
    -------------
    distance_sm   : stage length (statute miles)
    distance_nm   : stage length (nautical miles)
    block_hours   : total block time (hours)
    """
    df = routes_df.copy()
    df["distance_sm"] = haversine_sm(
        df["src_lat"].values,
        df["src_lon"].values,
        df["dst_lat"].values,
        df["dst_lon"].values,
    )
    df["distance_nm"] = haversine_nm(
        df["src_lat"].values,
        df["src_lon"].values,
        df["dst_lat"].values,
        df["dst_lon"].values,
    )
    df["block_hours"] = block_time(
        df["distance_sm"].values,
        cruise_speed_ktas=cruise_speed_ktas,
        taxi_time_hr=taxi_time_hr,
    )
    # Drop trivial zero-distance routes (same airport)
    df = df[df["distance_sm"] > 10.0].reset_index(drop=True)
    return df
