from __future__ import annotations

import numpy as np
import pytest

from simulation.cost_drivers import CostDriver, get_all_drivers
from simulation.monte_carlo import (
    compute_var,
    run_monte_carlo,
    sensitivity_analysis,
    tornado_data,
)


def test_get_all_drivers():
    drivers = get_all_drivers()
    assert len(drivers) == 15
    direct = [d for d in drivers if d.category == "Direct"]
    indirect = [d for d in drivers if d.category == "Indirect"]
    assert len(direct) == 5
    assert len(indirect) == 10


def test_monte_carlo_execution(sample_drivers: list[CostDriver]):
    n_iter = 500
    sim_df = run_monte_carlo(sample_drivers, n_iterations=n_iter, seed=42)
    assert len(sim_df) == n_iter
    assert "total" in sim_df.columns
    for d in sample_drivers:
        assert d.name in sim_df.columns


def test_compute_var_ordering(sample_drivers: list[CostDriver]):
    sim_df = run_monte_carlo(sample_drivers, n_iterations=1000, seed=42)
    var_res = compute_var(sim_df, confidence_levels=[0.95, 0.99])
    assert var_res["var_99"] > var_res["var_95"]
    assert var_res["var_95"] > var_res["expected_budget"]


def test_sensitivity_ranking(sample_drivers: list[CostDriver]):
    sim_df = run_monte_carlo(sample_drivers, n_iterations=500, seed=42)
    sens_df = sensitivity_analysis(sim_df, sample_drivers)
    assert len(sens_df) == 15
    assert "spearman_corr" in sens_df.columns
    corrs = sens_df["spearman_corr"].values
    assert not np.isnan(corrs).any()


def test_tornado_data(sample_drivers: list[CostDriver]):
    sim_df = run_monte_carlo(sample_drivers, n_iterations=500, seed=42)
    tornado_df = tornado_data(sim_df, sample_drivers)
    assert len(tornado_df) == 15
    assert "range" in tornado_df.columns
    assert (tornado_df["range"] >= 0).all()
