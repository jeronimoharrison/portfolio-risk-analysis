"""Portfolio Risk Analysis Tool — CLI entry point."""

import argparse
import sys
from datetime import date, timedelta

from config import DEFAULT_START_DATE, DEFAULT_END_DATE, DASHBOARD_HOST, DASHBOARD_PORT


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Risk Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --portfolio portfolio.xlsx --excel --dashboard
  python main.py --portfolio portfolio.xlsx --start 2023-01-01 --end 2025-01-01 --excel
  python main.py --template
        """,
    )
    parser.add_argument(
        "--portfolio", "-p",
        help="Path to portfolio Excel file (.xlsx)",
    )
    parser.add_argument(
        "--start", "-s",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
        help=f"Start date (YYYY-MM-DD, default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end", "-e",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
        help=f"End date (YYYY-MM-DD, default: {DEFAULT_END_DATE})",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Generate Excel risk report",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help=f"Launch Dash web dashboard on {DASHBOARD_HOST}:{DASHBOARD_PORT}",
    )
    parser.add_argument(
        "--output", "-o",
        default="portfolio_risk_report.xlsx",
        help="Output path for Excel report (default: portfolio_risk_report.xlsx)",
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Generate a portfolio template Excel file and exit",
    )

    args = parser.parse_args()

    # Generate template mode
    if args.template:
        from data.portfolio_loader import generate_template
        path = generate_template()
        print(f"Portfolio template generated: {path}")
        return

    # Dashboard-only mode (no portfolio needed upfront — upload in browser)
    if args.dashboard and not args.portfolio and not args.excel:
        print(f"Launching dashboard at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        print("Upload your portfolio file in the browser.")
        from output.dashboard import run_dashboard
        run_dashboard(host=DASHBOARD_HOST, port=DASHBOARD_PORT)
        return

    # Require portfolio for analysis
    if not args.portfolio:
        parser.error("--portfolio is required for analysis (or use --dashboard alone, or --template)")

    # Load portfolio
    from data.portfolio_loader import load_portfolio
    print(f"Loading portfolio from {args.portfolio}...")
    portfolio = load_portfolio(args.portfolio)
    print(f"  {len(portfolio)} holdings loaded")

    # Fetch Bloomberg data
    from data.bloomberg import fetch_portfolio_data
    print(f"Fetching Bloomberg data ({args.start} to {args.end})...")
    asset_prices, benchmark_prices = fetch_portfolio_data(portfolio, args.start, args.end)
    print(f"  {len(asset_prices)} trading days retrieved")

    # Compute returns
    from risk.returns import simple_returns, portfolio_returns
    import pandas as pd

    asset_ret = simple_returns(asset_prices)
    bench_ret = simple_returns(benchmark_prices)
    weights = pd.Series(portfolio["Weight"].values, index=portfolio["Ticker"].values)
    port_ret = portfolio_returns(asset_ret, weights)

    # Composite benchmark
    from risk.metrics import (
        annualized_return, annualized_volatility, sharpe_ratio, max_drawdown,
        composite_benchmark_returns, tracking_error, information_ratio,
    )
    from risk.var import historical_var
    from risk.returns import cumulative_returns

    comp_bench = composite_benchmark_returns(bench_ret, portfolio)

    print("\n-- Portfolio Summary --")
    print(f"  Total Return:      {cumulative_returns(port_ret).iloc[-1]:+.2%}")
    print(f"  Ann. Return:       {annualized_return(port_ret):+.2%}")
    print(f"  Benchmark Return:  {annualized_return(comp_bench):+.2%}")
    print(f"  Active Return:     {annualized_return(port_ret) - annualized_return(comp_bench):+.2%}")
    print(f"  Ann. Volatility:   {annualized_volatility(port_ret):.2%}")
    print(f"  Tracking Error:    {tracking_error(port_ret, comp_bench):.2%}")
    print(f"  Information Ratio: {information_ratio(port_ret, comp_bench):.2f}")
    print(f"  Sharpe Ratio:      {sharpe_ratio(port_ret):.2f}")
    print(f"  Max Drawdown:      {max_drawdown(port_ret):.2%}")
    print(f"  VaR (95%):         {historical_var(port_ret, 0.95):.2%}")
    print(f"  VaR (99%):         {historical_var(port_ret, 0.99):.2%}")

    # Excel report
    if args.excel:
        from output.excel_report import generate_report
        print(f"\nGenerating Excel report: {args.output}...")
        path = generate_report(portfolio, asset_ret, bench_ret, weights, args.output)
        print(f"  Report saved: {path}")

    # Dashboard
    if args.dashboard:
        print(f"\nLaunching dashboard at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        from output.dashboard import run_dashboard
        run_dashboard(host=DASHBOARD_HOST, port=DASHBOARD_PORT)


if __name__ == "__main__":
    main()
