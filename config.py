"""Default configuration for portfolio risk analysis."""

from datetime import date, timedelta

# Date range defaults
DEFAULT_END_DATE = date.today()
DEFAULT_START_DATE = DEFAULT_END_DATE - timedelta(days=2 * 365)

# Risk parameters
CONFIDENCE_LEVELS = [0.95, 0.99]
RISK_FREE_RATE = 0.05  # annualized
TRADING_DAYS_PER_YEAR = 252

# Monte Carlo
MC_SIMULATIONS = 10_000
MC_HORIZON_DAYS = 1

# Bloomberg
BLOOMBERG_HOST = "localhost"
BLOOMBERG_PORT = 8194

# Valid asset classes
VALID_ASSET_CLASSES = {"Equity", "Fixed Income", "Commodity", "FX", "Alternative"}

# Dashboard
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
