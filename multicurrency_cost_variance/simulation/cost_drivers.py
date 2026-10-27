"""
Cost Drivers Module.

Defines 15 operational cost drivers for aviation enterprise budgeting.
Each driver specifies a probability distribution (PERT or Log-Normal) for
Monte Carlo simulation of annual cost variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats as sp_stats


@dataclass(frozen=True)
class CostDriver:
    """Single operational cost driver with stochastic parameters.

    Attributes
    ----------
    name : str
        Human-readable driver name.
    category : str
        ``"Direct"`` or ``"Indirect"``.
    base_budget : float
        Annual budgeted cost in EUR.
    distribution_type : str
        ``"pert"`` or ``"lognormal"``.
    min_factor : float
        Minimum multiplicative factor (PERT only).
    mode_factor : float
        Most-likely multiplicative factor (PERT only).
    max_factor : float
        Maximum multiplicative factor (PERT only).
    std_dev : float
        Standard deviation σ for log-normal distribution.
    """

    name: str
    category: Literal["Direct", "Indirect"]
    base_budget: float
    distribution_type: Literal["pert", "lognormal"]
    min_factor: float = 0.0
    mode_factor: float = 1.0
    max_factor: float = 0.0
    std_dev: float = 0.0


def get_all_drivers() -> list[CostDriver]:
    """Return the canonical list of 15 aviation cost drivers.

    Returns
    -------
    list[CostDriver]
        Five direct and ten indirect cost drivers.
    """
    return [
        # ── Direct (5) ──────────────────────────────────────────────
        CostDriver(
            name="Aviation Fuel",
            category="Direct",
            base_budget=45_000_000.0,
            distribution_type="pert",
            min_factor=0.80,
            mode_factor=1.00,
            max_factor=1.45,
        ),
        CostDriver(
            name="Direct Labor",
            category="Direct",
            base_budget=32_000_000.0,
            distribution_type="pert",
            min_factor=0.92,
            mode_factor=1.00,
            max_factor=1.15,
        ),
        CostDriver(
            name="Aircraft Maintenance",
            category="Direct",
            base_budget=28_000_000.0,
            distribution_type="lognormal",
            std_dev=0.12,
        ),
        CostDriver(
            name="Airport Handling",
            category="Direct",
            base_budget=18_000_000.0,
            distribution_type="pert",
            min_factor=0.88,
            mode_factor=1.00,
            max_factor=1.20,
        ),
        CostDriver(
            name="Air Navigation Fees",
            category="Direct",
            base_budget=12_000_000.0,
            distribution_type="pert",
            min_factor=0.95,
            mode_factor=1.00,
            max_factor=1.08,
        ),
        # ── Indirect (10) ───────────────────────────────────────────
        CostDriver(
            name="IT Infrastructure",
            category="Indirect",
            base_budget=8_000_000.0,
            distribution_type="lognormal",
            std_dev=0.15,
        ),
        CostDriver(
            name="Software Licenses",
            category="Indirect",
            base_budget=5_000_000.0,
            distribution_type="pert",
            min_factor=0.90,
            mode_factor=1.00,
            max_factor=1.25,
        ),
        CostDriver(
            name="Commercial Insurance",
            category="Indirect",
            base_budget=6_000_000.0,
            distribution_type="pert",
            min_factor=0.85,
            mode_factor=1.00,
            max_factor=1.35,
        ),
        CostDriver(
            name="Legal Compliance",
            category="Indirect",
            base_budget=3_000_000.0,
            distribution_type="lognormal",
            std_dev=0.20,
        ),
        CostDriver(
            name="Facility Leases",
            category="Indirect",
            base_budget=7_000_000.0,
            distribution_type="pert",
            min_factor=0.98,
            mode_factor=1.00,
            max_factor=1.05,
        ),
        CostDriver(
            name="Marketing",
            category="Indirect",
            base_budget=9_000_000.0,
            distribution_type="pert",
            min_factor=0.70,
            mode_factor=1.00,
            max_factor=1.40,
        ),
        CostDriver(
            name="Customer Care",
            category="Indirect",
            base_budget=6_000_000.0,
            distribution_type="lognormal",
            std_dev=0.10,
        ),
        CostDriver(
            name="Telecom",
            category="Indirect",
            base_budget=2_000_000.0,
            distribution_type="pert",
            min_factor=0.90,
            mode_factor=1.00,
            max_factor=1.15,
        ),
        CostDriver(
            name="Admin Staff",
            category="Indirect",
            base_budget=4_000_000.0,
            distribution_type="pert",
            min_factor=0.95,
            mode_factor=1.00,
            max_factor=1.10,
        ),
        CostDriver(
            name="Financing Fees",
            category="Indirect",
            base_budget=5_000_000.0,
            distribution_type="lognormal",
            std_dev=0.18,
        ),
    ]


def _pert_beta_params(
    a: float, b: float, c: float, lam: float = 4.0
) -> tuple[float, float]:
    """Compute Beta distribution α and β from PERT min/mode/max.

    Parameters
    ----------
    a : float
        Minimum value.
    b : float
        Most-likely (mode) value.
    c : float
        Maximum value.
    lam : float
        PERT shape parameter (default 4).

    Returns
    -------
    alpha, beta : float
        Shape parameters for ``scipy.stats.beta``.
    """
    mu = (a + lam * b + c) / (lam + 2)
    # Guard against division by zero when a == c
    if c == a:
        return 1.0, 1.0
    alpha = ((mu - a) * (2 * b - a - c)) / ((b - mu) * (c - a))
    # If alpha is non-positive (degenerate case), fall back to symmetric
    if alpha <= 0:
        alpha = 1.0
    beta_param = alpha * (c - mu) / (mu - a) if (mu - a) != 0 else 1.0
    if beta_param <= 0:
        beta_param = 1.0
    return alpha, beta_param


def sample_cost(driver: CostDriver, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample *n* cost realisations from a driver's distribution.

    Parameters
    ----------
    driver : CostDriver
        The cost driver to sample from.
    n : int
        Number of samples.
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of length *n* with simulated annual costs (EUR).
    """
    if rng is None:
        rng = np.random.default_rng()

    if driver.distribution_type == "pert":
        a, b, c = driver.min_factor, driver.mode_factor, driver.max_factor
        alpha, beta_p = _pert_beta_params(a, b, c)
        # Scale Beta(0,1) → [a, c]
        raw = sp_stats.beta.ppf(rng.random(n), alpha, beta_p)
        factors = a + raw * (c - a)
    else:
        # Log-normal: factor = exp(N(0, σ))
        factors = rng.lognormal(mean=0.0, sigma=driver.std_dev, size=n)

    return driver.base_budget * factors
