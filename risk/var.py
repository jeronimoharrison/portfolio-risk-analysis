"""Value at Risk calculations: Historical, Parametric, and Monte Carlo."""

import numpy as np
import pandas as pd
from scipy import stats

from config import MC_SIMULATIONS, MC_HORIZON_DAYS, CONFIDENCE_LEVELS


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical VaR — percentile of actual return distribution.

    Returns a positive number representing the loss threshold.
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Parametric (Gaussian) VaR using mean and std of returns."""
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = MC_SIMULATIONS,
    horizon: int = MC_HORIZON_DAYS,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR using Cholesky decomposition for correlated simulation.

    Parameters
    ----------
    returns : pd.DataFrame
        Per-asset daily returns.
    weights : pd.Series
        Portfolio weights indexed by ticker.
    confidence : float
        Confidence level (e.g. 0.95).
    n_simulations : int
        Number of simulation paths.
    horizon : int
        Holding period in days.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Positive number representing the VaR threshold.
    """
    rng = np.random.default_rng(seed)
    aligned = returns[weights.index].dropna()
    mean = aligned.mean().values
    cov = aligned.cov().values

    # Cholesky decomposition
    L = np.linalg.cholesky(cov)

    # Simulate correlated returns
    n_assets = len(weights)
    Z = rng.standard_normal((n_simulations, horizon, n_assets))

    portfolio_sim_returns = np.zeros(n_simulations)
    w = weights.values

    for sim in range(n_simulations):
        cum = 1.0
        for day in range(horizon):
            correlated = mean + L @ Z[sim, day]
            port_ret = np.dot(w, correlated)
            cum *= (1 + port_ret)
        portfolio_sim_returns[sim] = cum - 1

    return -np.percentile(portfolio_sim_returns, (1 - confidence) * 100)


def compute_all_var(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    confidence_levels: list = None,
) -> pd.DataFrame:
    """Compute VaR using all three methods at multiple confidence levels.

    Returns
    -------
    pd.DataFrame
        Columns: Method, rows: confidence levels, values: VaR.
    """
    if confidence_levels is None:
        confidence_levels = CONFIDENCE_LEVELS

    results = []
    for conf in confidence_levels:
        results.append({
            "Confidence": f"{conf:.0%}",
            "Historical VaR": historical_var(portfolio_returns, conf),
            "Parametric VaR": parametric_var(portfolio_returns, conf),
            "Monte Carlo VaR": monte_carlo_var(asset_returns, weights, conf),
        })

    return pd.DataFrame(results).set_index("Confidence")


def per_asset_var(
    asset_returns: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute Historical and Parametric VaR per asset."""
    records = []
    for col in asset_returns.columns:
        series = asset_returns[col].dropna()
        records.append({
            "Ticker": col,
            "Historical VaR": historical_var(series, confidence),
            "Parametric VaR": parametric_var(series, confidence),
        })
    return pd.DataFrame(records).set_index("Ticker")
