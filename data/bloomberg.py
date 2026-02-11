"""Bloomberg API data fetching via blpapi."""

import pandas as pd
from datetime import date
from typing import List, Optional

try:
    import blpapi
except ImportError:
    blpapi = None

from config import BLOOMBERG_HOST, BLOOMBERG_PORT


def _start_session(host: str = BLOOMBERG_HOST, port: int = BLOOMBERG_PORT):
    """Start a Bloomberg API session."""
    if blpapi is None:
        raise ImportError(
            "blpapi is not installed. Install it with: pip install blpapi "
            "(requires Bloomberg Terminal or B-PIPE)"
        )
    options = blpapi.SessionOptions()
    options.setServerHost(host)
    options.setServerPort(port)
    session = blpapi.Session(options)
    if not session.start():
        raise ConnectionError("Failed to start Bloomberg session")
    if not session.openService("//blp/refdata"):
        session.stop()
        raise ConnectionError("Failed to open //blp/refdata service")
    return session


def fetch_historical_prices(
    tickers: List[str],
    start_date: date,
    end_date: date,
    field: str = "PX_LAST",
    host: str = BLOOMBERG_HOST,
    port: int = BLOOMBERG_PORT,
) -> pd.DataFrame:
    """Fetch historical daily prices from Bloomberg for multiple tickers.

    Parameters
    ----------
    tickers : list of str
        Bloomberg tickers (e.g. ['AAPL US Equity', 'SPX Index']).
    start_date, end_date : date
        Date range for historical data.
    field : str
        Bloomberg field to fetch (default PX_LAST).

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per ticker.
    """
    session = _start_session(host, port)
    try:
        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("HistoricalDataRequest")

        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        request.getElement("fields").appendValue(field)
        request.set("startDate", start_date.strftime("%Y%m%d"))
        request.set("endDate", end_date.strftime("%Y%m%d"))
        request.set("periodicitySelection", "DAILY")
        request.set("nonTradingDayFillOption", "ACTIVE_DAYS_ONLY")
        request.set("nonTradingDayFillMethod", "PREVIOUS_VALUE")

        session.sendRequest(request)

        prices = {}
        while True:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_data = msg.getElement("securityData")
                    ticker_name = sec_data.getElementAsString("security")
                    field_data = sec_data.getElement("fieldData")

                    dates = []
                    values = []
                    for i in range(field_data.numValues()):
                        point = field_data.getValueAsElement(i)
                        dt = point.getElementAsDatetime("date")
                        val = point.getElementAsFloat(field)
                        dates.append(pd.Timestamp(dt.year, dt.month, dt.day))
                        values.append(val)

                    prices[ticker_name] = pd.Series(values, index=dates)

            if event.eventType() == blpapi.Event.RESPONSE:
                break

        df = pd.DataFrame(prices)
        df.index.name = "Date"
        df = df.sort_index()
        return df

    finally:
        session.stop()


def fetch_security_names(
    tickers: List[str],
    host: str = BLOOMBERG_HOST,
    port: int = BLOOMBERG_PORT,
) -> dict:
    """Fetch security names from Bloomberg.

    Returns
    -------
    dict
        Mapping of ticker -> security name.
    """
    session = _start_session(host, port)
    try:
        refdata = session.getService("//blp/refdata")
        request = refdata.createRequest("ReferenceDataRequest")

        for ticker in tickers:
            request.getElement("securities").appendValue(ticker)
        request.getElement("fields").appendValue("NAME")

        session.sendRequest(request)

        names = {}
        while True:
            event = session.nextEvent(5000)
            for msg in event:
                if msg.hasElement("securityData"):
                    sec_array = msg.getElement("securityData")
                    for i in range(sec_array.numValues()):
                        sec = sec_array.getValueAsElement(i)
                        ticker_name = sec.getElementAsString("security")
                        fields = sec.getElement("fieldData")
                        if fields.hasElement("NAME"):
                            names[ticker_name] = fields.getElementAsString("NAME")
            if event.eventType() == blpapi.Event.RESPONSE:
                break

        return names

    finally:
        session.stop()


def fetch_portfolio_data(
    portfolio: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple:
    """Fetch all market data needed for portfolio analysis.

    Tries Bloomberg first. If blpapi is not installed or the connection
    fails, automatically falls back to realistic mock data so the tool
    can be tested without a Bloomberg Terminal.

    Parameters
    ----------
    portfolio : pd.DataFrame
        Portfolio holdings with Ticker and Benchmark columns.
    start_date, end_date : date
        Date range.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (asset_prices, benchmark_prices) - both indexed by date.
    """
    # Try Bloomberg first
    if blpapi is not None:
        try:
            asset_tickers = portfolio["Ticker"].tolist()
            benchmark_tickers = portfolio["Benchmark"].unique().tolist()

            all_tickers = list(set(asset_tickers + benchmark_tickers))
            all_prices = fetch_historical_prices(all_tickers, start_date, end_date)

            asset_prices = all_prices[asset_tickers].dropna(how="all")
            benchmark_prices = all_prices[benchmark_tickers].dropna(how="all")

            # Fill missing names in portfolio
            missing_names = portfolio["Name"].isna() | (portfolio["Name"].astype(str).str.strip() == "")
            if missing_names.any():
                tickers_need_names = portfolio.loc[missing_names, "Ticker"].tolist()
                names = fetch_security_names(tickers_need_names)
                for idx in portfolio.index[missing_names]:
                    ticker = portfolio.at[idx, "Ticker"]
                    if ticker in names:
                        portfolio.at[idx, "Name"] = names[ticker]

            return asset_prices, benchmark_prices
        except Exception:
            pass  # Fall through to mock data

    # Fallback to mock data
    import warnings
    warnings.warn(
        "Bloomberg not available — using simulated mock data for testing.",
        stacklevel=2,
    )
    from data.mock_data import mock_fetch_portfolio_data
    return mock_fetch_portfolio_data(portfolio, start_date, end_date)
