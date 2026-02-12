"""Portfolio risk and performance metrics."""

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility from daily returns."""
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def annualized_return(returns: pd.Series) -> float:
    """Annualized return from daily returns."""
    total = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    ann_ret = annualized_return(returns)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 0 else 0.0
    if downside_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / downside_vol


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak. Returns a positive number."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return -drawdowns.min() if len(drawdowns) > 0 else 0.0


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative - running_max) / running_max


def beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Beta of asset relative to benchmark."""
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    cov_matrix = aligned.cov()
    var_benchmark = cov_matrix.iloc[1, 1]
    if var_benchmark == 0:
        return np.nan
    return cov_matrix.iloc[0, 1] / var_benchmark


def tracking_error(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Annualized tracking error (std of excess returns)."""
    excess = (portfolio_returns - benchmark_returns).dropna()
    return excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def composite_benchmark_returns(
    benchmark_returns: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.Series:
    """Compute weighted composite benchmark returns at portfolio level.

    Each holding's benchmark is weighted by the holding's portfolio weight,
    producing a single blended benchmark return series.
    """
    bench_ret = pd.Series(0.0, index=benchmark_returns.index)
    for _, row in portfolio.iterrows():
        bench_ticker = row["Benchmark"]
        weight = row["Weight"]
        if bench_ticker in benchmark_returns.columns:
            bench_ret = bench_ret + weight * benchmark_returns[bench_ticker].fillna(0)
    return bench_ret


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Annualized information ratio (active return / tracking error)."""
    active_ret = annualized_return(portfolio_returns) - annualized_return(benchmark_returns)
    te = tracking_error(portfolio_returns, benchmark_returns)
    if te == 0:
        return 0.0
    return active_ret / te


def risk_contribution(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Marginal risk contribution per asset.

    Returns the fraction of total portfolio variance attributable
    to each asset.
    """
    aligned = asset_returns[weights.index].dropna()
    cov = aligned.cov().values
    w = weights.values

    port_var = w @ cov @ w
    if port_var == 0:
        return pd.Series(0.0, index=weights.index)

    marginal = cov @ w
    contrib = w * marginal / port_var
    return pd.Series(contrib, index=weights.index)


def compute_asset_metrics(
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-asset metrics: vol, Sharpe, Sortino, max DD, Beta, TE.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Daily returns per asset (columns = tickers).
    benchmark_returns : pd.DataFrame
        Daily returns for benchmarks (columns = benchmark tickers).
    portfolio : pd.DataFrame
        Portfolio dataframe with Ticker, Benchmark, Weight columns.

    Returns
    -------
    pd.DataFrame
        Per-asset metrics.
    """
    records = []
    for _, row in portfolio.iterrows():
        ticker = row["Ticker"]
        bench = row["Benchmark"]
        asset_ret = asset_returns[ticker].dropna()
        bench_ret = benchmark_returns[bench].dropna() if bench in benchmark_returns.columns else pd.Series(dtype=float)

        records.append({
            "Ticker": ticker,
            "Name": row.get("Name", ""),
            "Asset Class": row["Asset Class"],
            "Weight": row["Weight"],
            "Ann. Return": annualized_return(asset_ret),
            "Ann. Volatility": annualized_volatility(asset_ret),
            "Sharpe": sharpe_ratio(asset_ret),
            "Sortino": sortino_ratio(asset_ret),
            "Max Drawdown": max_drawdown(asset_ret),
            "Beta": beta(asset_ret, bench_ret) if len(bench_ret) > 0 else np.nan,
            "Tracking Error": tracking_error(asset_ret, bench_ret) if len(bench_ret) > 0 else np.nan,
        })

    return pd.DataFrame(records)


def compute_asset_class_metrics(
    asset_returns: pd.DataFrame,
    portfolio: pd.DataFrame,
    weights: pd.Series,
) -> pd.DataFrame:
    """Compute metrics aggregated by asset class."""
    rc = risk_contribution(asset_returns, weights)

    groups = portfolio.groupby("Asset Class")
    records = []
    for ac, group in groups:
        tickers = group["Ticker"].tolist()
        ac_weights = group["Weight"].values
        ac_returns = asset_returns[tickers].fillna(0).dot(ac_weights)

        records.append({
            "Asset Class": ac,
            "Weight": group["Weight"].sum(),
            "Ann. Return": annualized_return(ac_returns),
            "Ann. Volatility": annualized_volatility(ac_returns),
            "Sharpe": sharpe_ratio(ac_returns),
            "Max Drawdown": max_drawdown(ac_returns),
            "Risk Contribution": rc[tickers].sum(),
        })

    return pd.DataFrame(records)
