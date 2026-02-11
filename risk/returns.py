"""Return calculations for portfolio analysis."""

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple (arithmetic) returns from price series."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate logarithmic returns from price series."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Calculate weighted portfolio returns.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Per-asset returns (columns = tickers).
    weights : pd.Series
        Weights indexed by ticker.

    Returns
    -------
    pd.Series
        Portfolio-level returns.
    """
    aligned = asset_returns[weights.index].fillna(0)
    return aligned.dot(weights)


def cumulative_returns(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculate cumulative returns from a return series."""
    return (1 + returns).cumprod() - 1
