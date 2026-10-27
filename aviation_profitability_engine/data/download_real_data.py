"""
download_real_data.py
---------------------
Downloads aviation datasets from OpenFlights and FRED, then parses and
cleans them into Parquet tables using DuckDB for downstream analytics.

Data sources
~~~~~~~~~~~~
* OpenFlights airports.dat  – airport coordinates, IATA codes, countries
* OpenFlights routes.dat    – airline route pairs (source → destination)
* FRED DJFUELUSGULF         – U.S. Gulf Coast jet fuel spot price ($/gal)

Author: Eyoas Gizachew Zewdie
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import duckdb
import pandas as pd
import requests

# ── URLs ────────────────────────────────────────────────────────────────────
AIRPORTS_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
)
ROUTES_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"
)
FUEL_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DJFUELUSGULF"

# ── Column schemas ──────────────────────────────────────────────────────────
AIRPORT_COLUMNS = [
    "airport_id",
    "name",
    "city",
    "country",
    "iata",
    "icao",
    "latitude",
    "longitude",
    "altitude",
    "timezone_offset",
    "dst",
    "tz_database",
    "type",
    "source",
]

ROUTE_COLUMNS = [
    "airline",
    "airline_id",
    "source_airport",
    "source_airport_id",
    "dest_airport",
    "dest_airport_id",
    "codeshare",
    "stops",
    "equipment",
]

DATA_DIR = pathlib.Path(__file__).resolve().parent


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Download *url* to *dest* if the file does not already exist."""
    if dest.exists():
        print(f"  ✓ cached  {dest.name}")
        return dest
    print(f"  ↓ downloading {dest.name} …")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  ✓ saved   {dest.name} ({len(resp.content):,} bytes)")
    return dest


def download_airports(data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    """Fetch airports.dat and return its local path."""
    return _download(AIRPORTS_URL, data_dir / "airports.dat")


def download_routes(data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    """Fetch routes.dat and return its local path."""
    return _download(ROUTES_URL, data_dir / "routes.dat")


def download_fuel(data_dir: pathlib.Path = DATA_DIR) -> pathlib.Path:
    """Fetch FRED jet fuel CSV and return its local path."""
    return _download(FUEL_URL, data_dir / "fuel_prices.csv")


# ── Parsing helpers ─────────────────────────────────────────────────────────


def parse_airports(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read airports.dat (no header) → cleaned DataFrame with valid IATA codes."""
    con = duckdb.connect()
    query = f"""
    SELECT *
    FROM read_csv(
        '{csv_path}',
        header = false,
        columns = {{
            'airport_id':       'INTEGER',
            'name':             'VARCHAR',
            'city':             'VARCHAR',
            'country':          'VARCHAR',
            'iata':             'VARCHAR',
            'icao':             'VARCHAR',
            'latitude':         'DOUBLE',
            'longitude':        'DOUBLE',
            'altitude':         'INTEGER',
            'timezone_offset':  'VARCHAR',
            'dst':              'VARCHAR',
            'tz_database':      'VARCHAR',
            'type':             'VARCHAR',
            'source':           'VARCHAR'
        }}
    )
    WHERE iata IS NOT NULL
      AND iata != ''
      AND iata != '\\N'
      AND latitude  IS NOT NULL
      AND longitude IS NOT NULL
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df


def parse_routes(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read routes.dat → cleaned DataFrame with only direct (0-stop) routes."""
    con = duckdb.connect()
    query = f"""
    SELECT *
    FROM read_csv(
        '{csv_path}',
        header = false,
        columns = {{
            'airline':            'VARCHAR',
            'airline_id':         'VARCHAR',
            'source_airport':     'VARCHAR',
            'source_airport_id':  'VARCHAR',
            'dest_airport':       'VARCHAR',
            'dest_airport_id':    'VARCHAR',
            'codeshare':          'VARCHAR',
            'stops':              'INTEGER',
            'equipment':          'VARCHAR'
        }}
    )
    WHERE source_airport IS NOT NULL
      AND dest_airport   IS NOT NULL
      AND source_airport != '\\N'
      AND dest_airport   != '\\N'
      AND stops = 0
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df


def parse_fuel(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read FRED fuel CSV → DataFrame[date, price_per_gallon]."""
    df = pd.read_csv(csv_path, skiprows=0, parse_dates=["DATE"])
    df.columns = ["date", "price_per_gallon"]
    df["price_per_gallon"] = pd.to_numeric(df["price_per_gallon"], errors="coerce")
    df.dropna(subset=["price_per_gallon"], inplace=True)
    return df


def build_route_geography(
    airports_df: pd.DataFrame,
    routes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join routes with airport coordinates to produce a geo-enriched route table.

    Returns a DataFrame with columns:
        source_iata, dest_iata, src_lat, src_lon, dst_lat, dst_lon,
        src_city, dst_city, src_country, dst_country, src_elevation, dst_elevation
    """
    con = duckdb.connect()
    con.register("airports", airports_df)
    con.register("routes", routes_df)
    query = """
    SELECT DISTINCT
        r.source_airport  AS source_iata,
        r.dest_airport    AS dest_iata,
        a1.latitude       AS src_lat,
        a1.longitude      AS src_lon,
        a2.latitude       AS dst_lat,
        a2.longitude      AS dst_lon,
        a1.city           AS src_city,
        a2.city           AS dst_city,
        a1.country        AS src_country,
        a2.country        AS dst_country,
        a1.altitude       AS src_elevation,
        a2.altitude       AS dst_elevation
    FROM routes r
    JOIN airports a1 ON r.source_airport = a1.iata
    JOIN airports a2 ON r.dest_airport   = a2.iata
    WHERE a1.latitude  IS NOT NULL
      AND a1.longitude IS NOT NULL
      AND a2.latitude  IS NOT NULL
      AND a2.longitude IS NOT NULL
    """
    result = con.execute(query).fetchdf()
    con.close()
    return result


def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Persist a DataFrame as a Parquet file."""
    df.to_parquet(path, index=False)
    print(f"  ✓ parquet  {path.name}  ({len(df):,} rows)")


def run_pipeline(data_dir: Optional[pathlib.Path] = None) -> dict[str, pd.DataFrame]:
    """Execute the full download → parse → save pipeline.

    Returns
    -------
    dict with keys 'airports', 'routes', 'route_geo', 'fuel'
    """
    data_dir = pathlib.Path(data_dir) if data_dir else DATA_DIR

    print("━━━  Aviation Data Pipeline  ━━━")

    # 1. Download
    airports_csv = download_airports(data_dir)
    routes_csv = download_routes(data_dir)
    fuel_csv = download_fuel(data_dir)

    # 2. Parse
    airports_df = parse_airports(airports_csv)
    routes_df = parse_routes(routes_csv)
    fuel_df = parse_fuel(fuel_csv)

    # 3. Join routes with geography
    route_geo_df = build_route_geography(airports_df, routes_df)

    # 4. Save Parquet
    save_parquet(airports_df, data_dir / "airports.parquet")
    save_parquet(route_geo_df, data_dir / "route_geography.parquet")
    save_parquet(fuel_df, data_dir / "fuel_prices.parquet")

    print(f"\n  Airports  : {len(airports_df):>7,}")
    print(f"  Routes    : {len(routes_df):>7,}")
    print(f"  Route-Geo : {len(route_geo_df):>7,}")
    print(f"  Fuel obs  : {len(fuel_df):>7,}")
    print("━━━  Pipeline complete  ━━━\n")

    return {
        "airports": airports_df,
        "routes": routes_df,
        "route_geo": route_geo_df,
        "fuel": fuel_df,
    }


if __name__ == "__main__":
    run_pipeline()
