"""Load and validate portfolio holdings from Excel."""

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from pathlib import Path

from config import VALID_ASSET_CLASSES

REQUIRED_COLUMNS = {"Ticker", "Asset Class", "Weight", "Benchmark"}


def load_portfolio(filepath: str) -> pd.DataFrame:
    """Load portfolio from Excel file and validate.

    Parameters
    ----------
    filepath : str
        Path to .xlsx file with a 'Holdings' sheet.

    Returns
    -------
    pd.DataFrame
        Validated portfolio dataframe.

    Raises
    ------
    ValueError
        If validation fails.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {filepath}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError("Portfolio file must be .xlsx format")

    df = pd.read_excel(filepath, sheet_name="Holdings", engine="openpyxl")
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Validate portfolio dataframe."""
    # Check required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check no empty tickers
    if df["Ticker"].isna().any() or (df["Ticker"].astype(str).str.strip() == "").any():
        raise ValueError("All rows must have a Ticker")

    # Check duplicate tickers
    dupes = df["Ticker"].duplicated()
    if dupes.any():
        raise ValueError(f"Duplicate tickers: {df.loc[dupes, 'Ticker'].tolist()}")

    # Check asset classes
    invalid = set(df["Asset Class"].dropna()) - VALID_ASSET_CLASSES
    if invalid:
        raise ValueError(
            f"Invalid asset classes: {invalid}. Valid: {VALID_ASSET_CLASSES}"
        )

    # Check weights sum to 1.0
    total_weight = df["Weight"].sum()
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {total_weight:.6f}"
        )

    # Check no negative weights
    if (df["Weight"] < 0).any():
        raise ValueError("Weights must be non-negative")

    # Check benchmarks not empty
    if df["Benchmark"].isna().any():
        raise ValueError("All holdings must have a Benchmark ticker")


def generate_template(output_path: str = "templates/portfolio_template.xlsx") -> str:
    """Generate an example portfolio template Excel file."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Holdings"

    headers = ["Ticker", "Name", "Asset Class", "Weight", "Benchmark"]
    sample_data = [
        ["AAPL US Equity", "Apple Inc", "Equity", 0.15, "SPX Index"],
        ["MSFT US Equity", "Microsoft Corp", "Equity", 0.10, "SPX Index"],
        ["TLT US Equity", "iShares 20+ Yr Treasury", "Fixed Income", 0.20, "LBUSTRUU Index"],
        ["GLD US Equity", "SPDR Gold Shares", "Commodity", 0.10, "BCOMTR Index"],
        ["EURUSD Curncy", "EUR/USD", "FX", 0.05, "DXY Curncy"],
        ["SPY US Equity", "SPDR S&P 500", "Alternative", 0.40, "SPX Index"],
    ]

    # Style
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1B3A5C", end_color="1B3A5C", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin"),
    )

    # Write headers
    for col_idx, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    # Write sample data
    for row_idx, row_data in enumerate(sample_data, 2):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.border = thin_border
            if col_idx == 4:  # Weight column
                cell.number_format = "0.00%"

    # Auto-fit column widths
    for col_idx, header in enumerate(headers, 1):
        max_len = len(header)
        for row in range(2, len(sample_data) + 2):
            val = str(ws.cell(row=row, column=col_idx).value or "")
            max_len = max(max_len, len(val))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max_len + 4

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path
