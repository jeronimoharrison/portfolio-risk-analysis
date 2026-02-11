"""Correlation and covariance matrix analysis."""

import pandas as pd
import numpy as np


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation matrix."""
    return returns.corr()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise covariance matrix."""
    return returns.cov()


def asset_class_correlation(
    asset_returns: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Compute correlation between asset classes (weighted composite returns).

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Per-asset daily returns.
    portfolio : pd.DataFrame
        Portfolio with Ticker, Asset Class, Weight columns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix between asset classes.
    """
    ac_returns = {}
    for ac, group in portfolio.groupby("Asset Class"):
        tickers = group["Ticker"].tolist()
        w = group["Weight"].values
        total_w = w.sum()
        if total_w > 0:
            # Normalize weights within class
            w_norm = w / total_w
            ac_returns[ac] = asset_returns[tickers].fillna(0).dot(w_norm)

    ac_df = pd.DataFrame(ac_returns)
    return ac_df.corr()


def intra_class_avg_correlation(
    asset_returns: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.Series:
    """Average pairwise correlation within each asset class."""
    results = {}
    for ac, group in portfolio.groupby("Asset Class"):
        tickers = group["Ticker"].tolist()
        if len(tickers) < 2:
            results[ac] = np.nan
            continue
        corr = asset_returns[tickers].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        results[ac] = corr.where(mask).stack().mean()

    return pd.Series(results, name="Avg Intra-Class Correlation")
