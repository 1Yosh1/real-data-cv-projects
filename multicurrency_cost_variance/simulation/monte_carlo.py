"""
Monte Carlo Simulation Module.

Runs a 10,000-iteration parametric simulation over 15 cost drivers using
Latin Hypercube Sampling, then computes Value-at-Risk, sensitivity analysis,
and tornado chart data for executive risk reporting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import qmc, spearmanr

from simulation.cost_drivers import CostDriver, _pert_beta_params


def _map_uniform_to_driver(u: np.ndarray, driver: CostDriver) -> np.ndarray:
    """Map uniform [0,1] samples to a driver's cost distribution.

    Parameters
    ----------
    u : np.ndarray
        Uniform samples in [0, 1].
    driver : CostDriver
        The cost driver whose distribution is used.

    Returns
    -------
    np.ndarray
        Simulated annual costs (EUR).
    """
    if driver.distribution_type == "pert":
        a, b, c = driver.min_factor, driver.mode_factor, driver.max_factor
        alpha, beta_p = _pert_beta_params(a, b, c)
        factors = a + sp_stats.beta.ppf(u, alpha, beta_p) * (c - a)
    else:
        # Log-normal with μ=0, σ=driver.std_dev
        factors = sp_stats.lognorm.ppf(u, s=driver.std_dev, scale=1.0)

    return driver.base_budget * factors


def run_monte_carlo(
    drivers: list[CostDriver],
    n_iterations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Execute a Latin Hypercube Monte Carlo simulation.

    Parameters
    ----------
    drivers : list[CostDriver]
        The 15 cost drivers.
    n_iterations : int
        Number of simulation iterations.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_iterations, len(drivers) + 1)``.  One column per driver
        (named by driver name) plus a ``total`` column.
    """
    d = len(drivers)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    uniform_samples = sampler.random(n=n_iterations)  # shape: (n_iterations, d)

    results: dict[str, np.ndarray] = {}
    for i, drv in enumerate(drivers):
        results[drv.name] = _map_uniform_to_driver(uniform_samples[:, i], drv)

    df = pd.DataFrame(results)
    df["total"] = df.sum(axis=1)
    return df


def compute_var(
    simulations: pd.DataFrame,
    confidence_levels: list[float] | None = None,
) -> dict[str, Any]:
    """Compute Value-at-Risk metrics from simulation results.

    Parameters
    ----------
    simulations : pd.DataFrame
        Output of ``run_monte_carlo`` containing a ``total`` column.
    confidence_levels : list[float], optional
        Percentile levels (default ``[0.95, 0.99]``).

    Returns
    -------
    dict
        Keys: ``base_budget``, ``expected_budget``, ``std_dev``,
        ``var_95``, ``var_99``, ``max_cost``, ``min_cost``.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    totals = simulations["total"]
    result: dict[str, Any] = {
        "base_budget": float(
            simulations.drop(columns="total").iloc[0].sum()
            if len(simulations) > 0
            else 0
        ),
        "expected_budget": float(totals.mean()),
        "std_dev": float(totals.std()),
        "min_cost": float(totals.min()),
        "max_cost": float(totals.max()),
    }

    # Compute base budget as sum of mode/expected outcomes per driver
    # (use the median of the simulation for each driver as proxy)
    base_cols = [c for c in simulations.columns if c != "total"]
    result["base_budget"] = float(simulations[base_cols].median().sum())

    for level in confidence_levels:
        key = f"var_{int(level * 100)}"
        result[key] = float(np.percentile(totals, level * 100))

    return result


def sensitivity_analysis(
    simulations: pd.DataFrame,
    drivers: list[CostDriver],
) -> pd.DataFrame:
    """Rank drivers by Spearman correlation with total budget variance.

    Parameters
    ----------
    simulations : pd.DataFrame
        Output of ``run_monte_carlo``.
    drivers : list[CostDriver]
        List of cost drivers (used for metadata).

    Returns
    -------
    pd.DataFrame
        Columns: ``driver``, ``category``, ``base_budget``,
        ``spearman_corr``, ``abs_corr``.  Sorted descending by absolute
        correlation.
    """
    total = simulations["total"].values
    rows: list[dict[str, Any]] = []

    for drv in drivers:
        col_data = simulations[drv.name].values
        corr, _ = spearmanr(col_data, total)
        rows.append(
            {
                "driver": drv.name,
                "category": drv.category,
                "base_budget": drv.base_budget,
                "spearman_corr": round(corr, 4),
                "abs_corr": round(abs(corr), 4),
            }
        )

    result = pd.DataFrame(rows)
    result.sort_values("abs_corr", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def tornado_data(
    simulations: pd.DataFrame,
    drivers: list[CostDriver],
) -> pd.DataFrame:
    """Compute tornado chart data: P5–P95 range of total budget per driver.

    For each driver the 5th and 95th percentile of that driver's own cost
    are computed, and then the resulting total budget is estimated by
    holding all *other* drivers at their median and varying only the
    target driver.

    Parameters
    ----------
    simulations : pd.DataFrame
        Output of ``run_monte_carlo``.
    drivers : list[CostDriver]
        List of cost drivers.

    Returns
    -------
    pd.DataFrame
        Columns: ``driver``, ``low_total``, ``high_total``, ``range``,
        ``base_total``.  Sorted descending by ``range``.
    """
    driver_names = [d.name for d in drivers]
    medians = simulations[driver_names].median()
    base_total = float(medians.sum())

    rows: list[dict[str, Any]] = []
    for drv in drivers:
        p5 = float(np.percentile(simulations[drv.name], 5))
        p95 = float(np.percentile(simulations[drv.name], 95))
        # Total when this driver is at its P5 (all others at median)
        low_total = base_total - float(medians[drv.name]) + p5
        high_total = base_total - float(medians[drv.name]) + p95
        rows.append(
            {
                "driver": drv.name,
                "low_total": round(low_total, 2),
                "high_total": round(high_total, 2),
                "range": round(high_total - low_total, 2),
                "base_total": round(base_total, 2),
            }
        )

    result = pd.DataFrame(rows)
    result.sort_values("range", ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result
