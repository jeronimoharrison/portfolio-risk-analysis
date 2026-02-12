# Portfolio Risk Analysis Tool

A comprehensive multi-asset portfolio risk analysis tool that computes a full suite of risk metrics, generates styled Excel reports, and serves an interactive web dashboard with a Bloomberg Terminal-inspired dark theme.

## Features

- **Multi-asset support**: Equities, Fixed Income, Commodities, FX, Alternatives
- **Bloomberg integration**: Pulls live market data via `blpapi` with automatic fallback to simulated data
- **Risk metrics**: VaR (Historical, Parametric, Monte Carlo), CVaR, volatility, Sharpe ratio, Sortino ratio, max drawdown, beta, tracking error, risk contribution
- **Excel reports**: 5-sheet styled report with summary, asset class breakdown, VaR analysis, correlation heatmap, and holdings detail
- **Web dashboard**: Interactive Dash dashboard with KPI cards, 7 chart types, sortable tables, and Excel export

## Dashboard

![Dashboard](https://raw.githubusercontent.com/jeronimoharrison/portfolio-risk-analysis/master/dashboard_with_data.png)

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

> **Note**: `blpapi` requires a Bloomberg Terminal or B-PIPE connection. Without it, the tool automatically uses realistic simulated data.

### Generate a portfolio template

```bash
python main.py --template
```

### Run analysis with Excel report

```bash
python main.py --portfolio templates/portfolio_template.xlsx --excel
```

### Launch the web dashboard

```bash
python main.py --dashboard
```

Then open http://127.0.0.1:8050, upload your portfolio `.xlsx` file, and click **RUN ANALYSIS**.

### Run both

```bash
python main.py --portfolio portfolio.xlsx --start 2023-01-01 --end 2025-01-01 --excel --dashboard
```

## Portfolio Input Format

Create an `.xlsx` file with a sheet named **Holdings**:

| Ticker | Name | Asset Class | Weight | Benchmark |
|---|---|---|---|---|
| AAPL US Equity | Apple Inc | Equity | 0.15 | SPX Index |
| MSFT US Equity | Microsoft Corp | Equity | 0.10 | SPX Index |
| TLT US Equity | iShares 20+ Yr Treasury | Fixed Income | 0.20 | LBUSTRUU Index |
| GLD US Equity | SPDR Gold Shares | Commodity | 0.10 | BCOMTR Index |
| EURUSD Curncy | EUR/USD | FX | 0.05 | DXY Curncy |
| SPY US Equity | SPDR S&P 500 | Alternative | 0.40 | SPX Index |

- **Weight** must sum to 1.0
- **Asset Class**: Equity, Fixed Income, Commodity, FX, or Alternative
- **Benchmark**: Bloomberg ticker used for beta and tracking error calculations

## Project Structure

```
├── main.py                     # CLI entry point
├── config.py                   # Default settings
├── data/
│   ├── bloomberg.py            # Bloomberg API with mock fallback
│   ├── mock_data.py            # GBM-based simulated market data
│   └── portfolio_loader.py     # Excel loader and validation
├── risk/
│   ├── returns.py              # Return calculations
│   ├── var.py                  # VaR (historical, parametric, Monte Carlo)
│   ├── cvar.py                 # Conditional VaR / Expected Shortfall
│   ├── metrics.py              # Volatility, Sharpe, Sortino, drawdown, beta
│   └── correlation.py          # Correlation and covariance matrices
├── output/
│   ├── excel_report.py         # 5-sheet styled Excel report
│   └── dashboard.py            # Dash web dashboard
├── templates/
│   └── portfolio_template.xlsx # Sample portfolio
└── requirements.txt
```

## Dependencies

- `pandas`, `numpy`, `scipy` — Data and statistics
- `dash`, `plotly` — Web dashboard and charts
- `openpyxl` — Excel read/write
- `blpapi` — Bloomberg API (optional)

## CLI Options

```
usage: main.py [-h] [--portfolio PORTFOLIO] [--start START] [--end END]
               [--excel] [--dashboard] [--output OUTPUT] [--template]

  --portfolio, -p    Path to portfolio Excel file (.xlsx)
  --start, -s        Start date YYYY-MM-DD (default: 2 years ago)
  --end, -e          End date YYYY-MM-DD (default: today)
  --excel            Generate Excel risk report
  --dashboard        Launch web dashboard on localhost:8050
  --output, -o       Output path for Excel report
  --template         Generate a sample portfolio template
```
