"""Generate realistic mock market data for testing without Bloomberg."""

import numpy as np
import pandas as pd
from datetime import date

# Realistic parameters per asset: (annual_return, annual_vol, starting_price)
MOCK_PROFILES = {
    # Equities
    "AAPL US Equity":   (0.25, 0.30, 175.0),
    "MSFT US Equity":   (0.22, 0.28, 370.0),
    "AMZN US Equity":   (0.20, 0.35, 145.0),
    "GOOGL US Equity":  (0.18, 0.28, 140.0),
    "NVDA US Equity":   (0.40, 0.50, 480.0),
    "META US Equity":   (0.28, 0.38, 350.0),
    "TSLA US Equity":   (0.15, 0.55, 245.0),
    "JPM US Equity":    (0.12, 0.22, 170.0),
    "V US Equity":      (0.15, 0.20, 275.0),
    "JNJ US Equity":    (0.06, 0.15, 155.0),
    "SPY US Equity":    (0.12, 0.16, 470.0),
    # Fixed Income
    "TLT US Equity":    (0.02, 0.15, 95.0),
    "AGG US Equity":    (0.03, 0.05, 100.0),
    "LQD US Equity":    (0.04, 0.08, 110.0),
    "HYG US Equity":    (0.05, 0.07, 75.0),
    "SHY US Equity":    (0.04, 0.02, 82.0),
    # Commodities
    "GLD US Equity":    (0.08, 0.15, 190.0),
    "SLV US Equity":    (0.06, 0.25, 22.0),
    "USO US Equity":    (0.05, 0.30, 72.0),
    "DBA US Equity":    (0.03, 0.12, 25.0),
    # FX
    "EURUSD Curncy":    (0.01, 0.08, 1.09),
    "GBPUSD Curncy":    (0.01, 0.09, 1.27),
    "USDJPY Curncy":    (0.02, 0.10, 148.0),
    # Alternatives
    "VNQ US Equity":    (0.08, 0.20, 82.0),
    "ARKK US Equity":   (0.10, 0.45, 45.0),
    # Benchmarks / Indices
    "SPX Index":        (0.11, 0.15, 4800.0),
    "LBUSTRUU Index":   (0.03, 0.05, 2200.0),
    "BCOMTR Index":     (0.04, 0.14, 240.0),
    "DXY Curncy":       (-0.01, 0.07, 103.0),
    "NDX Index":        (0.16, 0.20, 16800.0),
}

# Correlation groups — assets in the same group are more correlated
_CORR_GROUPS = {
    "equity":  ["AAPL US Equity", "MSFT US Equity", "AMZN US Equity", "GOOGL US Equity",
                "NVDA US Equity", "META US Equity", "TSLA US Equity", "JPM US Equity",
                "V US Equity", "JNJ US Equity", "SPY US Equity", "SPX Index", "NDX Index"],
    "fi":      ["TLT US Equity", "AGG US Equity", "LQD US Equity", "HYG US Equity",
                "SHY US Equity", "LBUSTRUU Index"],
    "cmdty":   ["GLD US Equity", "SLV US Equity", "USO US Equity", "DBA US Equity",
                "BCOMTR Index"],
    "fx":      ["EURUSD Curncy", "GBPUSD Curncy", "USDJPY Curncy", "DXY Curncy"],
    "alt":     ["VNQ US Equity", "ARKK US Equity"],
}


def _get_group(ticker: str) -> str:
    for group, members in _CORR_GROUPS.items():
        if ticker in members:
            return group
    return "other"


def generate_mock_prices(
    tickers: list,
    start_date: date,
    end_date: date,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic correlated mock price data.

    Uses geometric Brownian motion with intra-group correlations.
    """
    rng = np.random.default_rng(seed)

    # Build business day index
    dates = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(dates)
    n_assets = len(tickers)

    # Build correlation matrix
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            gi = _get_group(tickers[i])
            gj = _get_group(tickers[j])
            if gi == gj:
                # High intra-group correlation
                c = rng.uniform(0.5, 0.85)
            elif {gi, gj} == {"equity", "alt"}:
                c = rng.uniform(0.3, 0.6)
            elif {gi, gj} == {"equity", "fi"}:
                # Equity-bond negative correlation
                c = rng.uniform(-0.4, -0.1)
            elif {gi, gj} == {"equity", "cmdty"}:
                c = rng.uniform(0.0, 0.3)
            elif {gi, gj} == {"fx", "equity"}:
                c = rng.uniform(-0.2, 0.2)
            else:
                c = rng.uniform(-0.15, 0.15)
            corr[i, j] = c
            corr[j, i] = c

    # Ensure positive semi-definite
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # Re-normalize to correlation matrix
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)

    L = np.linalg.cholesky(corr)

    # Generate correlated daily returns via GBM
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)

    for i, ticker in enumerate(tickers):
        profile = MOCK_PROFILES.get(ticker)
        if profile:
            ann_ret, ann_vol, p0 = profile
        else:
            # Fallback for unknown tickers
            ann_ret, ann_vol, p0 = 0.08, 0.20, 100.0

        daily_mu = ann_ret / 252
        daily_sigma = ann_vol / np.sqrt(252)

        Z = rng.standard_normal(n_days)
        correlated_Z = np.zeros(n_days)
        for t in range(n_days):
            z_all = rng.standard_normal(n_assets)
            correlated = L @ z_all
            correlated_Z[t] = correlated[i]

        # GBM: S(t) = S(0) * exp(sum of (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        log_returns = (daily_mu - 0.5 * daily_sigma**2) + daily_sigma * correlated_Z
        cum_log_returns = np.cumsum(log_returns)
        prices[ticker] = p0 * np.exp(cum_log_returns)

    prices.index.name = "Date"
    return prices


def mock_fetch_portfolio_data(
    portfolio: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple:
    """Drop-in replacement for bloomberg.fetch_portfolio_data using mock data."""
    asset_tickers = portfolio["Ticker"].tolist()
    benchmark_tickers = portfolio["Benchmark"].unique().tolist()
    all_tickers = list(set(asset_tickers + benchmark_tickers))

    all_prices = generate_mock_prices(all_tickers, start_date, end_date)

    asset_prices = all_prices[asset_tickers]
    benchmark_prices = all_prices[benchmark_tickers]

    return asset_prices, benchmark_prices
