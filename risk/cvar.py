"""Conditional Value at Risk (Expected Shortfall)."""

import numpy as np
import pandas as pd

from config import CONFIDENCE_LEVELS


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """CVaR / Expected Shortfall — average loss beyond VaR threshold.

    Returns a positive number representing the expected loss
    in the worst (1 - confidence) fraction of scenarios.
    """
    clean = returns.dropna()
    cutoff = np.percentile(clean, (1 - confidence) * 100)
    tail = clean[clean <= cutoff]
    return -tail.mean() if len(tail) > 0 else 0.0


def compute_all_cvar(
    portfolio_returns: pd.Series,
    confidence_levels: list = None,
) -> pd.DataFrame:
    """Compute CVaR at multiple confidence levels.

    Returns
    -------
    pd.DataFrame
        Index: confidence level, columns: CVaR.
    """
    if confidence_levels is None:
        confidence_levels = CONFIDENCE_LEVELS

    results = []
    for conf in confidence_levels:
        results.append({
            "Confidence": f"{conf:.0%}",
            "CVaR": conditional_var(portfolio_returns, conf),
        })

    return pd.DataFrame(results).set_index("Confidence")


def per_asset_cvar(
    asset_returns: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute CVaR per individual asset."""
    records = []
    for col in asset_returns.columns:
        records.append({
            "Ticker": col,
            "CVaR": conditional_var(asset_returns[col], confidence),
        })
    return pd.DataFrame(records).set_index("Ticker")
