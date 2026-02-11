"""Dash web dashboard with Bloomberg Terminal-inspired dark theme."""

import io
import base64
from datetime import date, timedelta

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from data.portfolio_loader import load_portfolio
from data.bloomberg import fetch_portfolio_data
from risk.returns import simple_returns, portfolio_returns, cumulative_returns
from risk.var import historical_var, parametric_var, monte_carlo_var, compute_all_var
from risk.cvar import conditional_var, compute_all_cvar
from risk.metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, drawdown_series, risk_contribution, compute_asset_class_metrics,
    compute_asset_metrics,
)
from risk.correlation import correlation_matrix, asset_class_correlation
from output.excel_report import generate_report

# ── Theme colors ──
BG_COLOR = "#0A0E17"
CARD_BG = "#141B2D"
PANEL_BG = "#1A2035"
TEXT_COLOR = "#E0E0E0"
GREEN = "#00C853"
ORANGE = "#FF9100"
RED = "#FF1744"
BLUE = "#2979FF"
HEADER_GREEN = "#00E676"

# Dark theme layout defaults applied to every figure
_DARK_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family="Consolas, monospace"),
    xaxis=dict(gridcolor="#2A3050", zerolinecolor="#2A3050"),
    yaxis=dict(gridcolor="#2A3050", zerolinecolor="#2A3050"),
    colorway=[GREEN, BLUE, ORANGE, RED, "#AB47BC", "#26C6DA", "#FFEE58"],
)


def _make_fig(**extra_layout) -> go.Figure:
    """Create a new Figure with the dark theme applied."""
    fig = go.Figure()
    fig.update_layout(**_DARK_LAYOUT, **extra_layout)
    return fig


def _kpi_card(title: str, value: str, color: str = TEXT_COLOR) -> html.Div:
    return html.Div(
        [
            html.P(title, style={"color": "#888", "margin": "0", "fontSize": "12px",
                                  "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.H3(value, style={"color": color, "margin": "4px 0 0 0", "fontSize": "24px"}),
        ],
        style={
            "backgroundColor": CARD_BG,
            "padding": "16px 24px",
            "borderRadius": "6px",
            "border": "1px solid #2A3050",
            "flex": "1",
            "minWidth": "160px",
        },
    )


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        title="Portfolio Risk Dashboard",
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div(
        style={"backgroundColor": BG_COLOR, "minHeight": "100vh", "padding": "20px",
               "fontFamily": "Consolas, monospace", "color": TEXT_COLOR},
        children=[
            # ── Header ──
            html.Div(
                [
                    html.H1("PORTFOLIO RISK ANALYTICS",
                            style={"color": HEADER_GREEN, "margin": "0", "letterSpacing": "3px"}),
                    html.P("Bloomberg-powered risk analysis",
                           style={"color": "#666", "margin": "4px 0 0 0"}),
                ],
                style={"marginBottom": "20px"},
            ),

            # ── Controls Row ──
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "20px",
                       "flexWrap": "wrap", "alignItems": "flex-end"},
                children=[
                    # File upload
                    html.Div([
                        html.Label("Portfolio File", style={"color": "#888", "fontSize": "12px"}),
                        dcc.Upload(
                            id="upload-portfolio",
                            children=html.Div(["Drag & Drop or ", html.A("Select .xlsx", style={"color": GREEN})]),
                            style={
                                "borderWidth": "1px", "borderStyle": "dashed", "borderColor": "#444",
                                "borderRadius": "6px", "padding": "12px", "textAlign": "center",
                                "backgroundColor": PANEL_BG, "cursor": "pointer", "minWidth": "250px",
                            },
                        ),
                    ]),
                    # Date range
                    html.Div([
                        html.Label("Start Date", style={"color": "#888", "fontSize": "12px"}),
                        dcc.DatePickerSingle(
                            id="start-date",
                            date=date.today() - timedelta(days=730),
                            display_format="YYYY-MM-DD",
                            style={"backgroundColor": PANEL_BG},
                        ),
                    ]),
                    html.Div([
                        html.Label("End Date", style={"color": "#888", "fontSize": "12px"}),
                        dcc.DatePickerSingle(
                            id="end-date",
                            date=date.today(),
                            display_format="YYYY-MM-DD",
                            style={"backgroundColor": PANEL_BG},
                        ),
                    ]),
                    # Confidence level
                    html.Div([
                        html.Label("Confidence", style={"color": "#888", "fontSize": "12px"}),
                        dcc.Dropdown(
                            id="confidence-level",
                            options=[
                                {"label": "95%", "value": 0.95},
                                {"label": "99%", "value": 0.99},
                            ],
                            value=0.95,
                            clearable=False,
                            style={"width": "100px", "backgroundColor": PANEL_BG, "color": "#000"},
                        ),
                    ]),
                    # Run button
                    html.Button(
                        "RUN ANALYSIS",
                        id="run-btn",
                        style={
                            "backgroundColor": GREEN, "color": "#000", "border": "none",
                            "padding": "12px 24px", "borderRadius": "6px", "cursor": "pointer",
                            "fontWeight": "bold", "fontFamily": "Consolas, monospace",
                            "letterSpacing": "1px",
                        },
                    ),
                    # Export button
                    html.Button(
                        "EXPORT EXCEL",
                        id="export-btn",
                        style={
                            "backgroundColor": ORANGE, "color": "#000", "border": "none",
                            "padding": "12px 24px", "borderRadius": "6px", "cursor": "pointer",
                            "fontWeight": "bold", "fontFamily": "Consolas, monospace",
                        },
                    ),
                    dcc.Download(id="download-report"),
                ],
            ),

            # ── Status ──
            html.Div(id="status-msg", style={"color": ORANGE, "marginBottom": "12px"}),

            # ── Store for computed data ──
            dcc.Store(id="analysis-store"),

            # ── KPI Cards ──
            html.Div(
                id="kpi-row",
                style={"display": "flex", "gap": "16px", "marginBottom": "20px", "flexWrap": "wrap"},
            ),

            # ── Charts Grid ──
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px",
                       "marginBottom": "20px"},
                children=[
                    # Cumulative returns
                    html.Div(
                        dcc.Graph(id="chart-cumret", figure=_empty_fig(), config={"displayModeBar": False}),
                        style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "8px"},
                    ),
                    # Allocation pie
                    html.Div(
                        dcc.Graph(id="chart-allocation", figure=_empty_fig(), config={"displayModeBar": False}),
                        style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "8px"},
                    ),
                    # Rolling volatility
                    html.Div(
                        dcc.Graph(id="chart-rolling-vol", figure=_empty_fig(), config={"displayModeBar": False}),
                        style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "8px"},
                    ),
                    # Correlation heatmap
                    html.Div(
                        dcc.Graph(id="chart-corr", figure=_empty_fig(), config={"displayModeBar": False}),
                        style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "8px"},
                    ),
                    # VaR distribution
                    html.Div(
                        dcc.Graph(id="chart-var-dist", figure=_empty_fig(), config={"displayModeBar": False}),
                        style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "8px"},
                    ),
                    # Drawdown
                    html.Div(
                        dcc.Graph(id="chart-drawdown", figure=_empty_fig(), config={"displayModeBar": False}),
                        style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "8px"},
                    ),
                    # Risk contribution
                    html.Div(
                        dcc.Graph(id="chart-risk-contrib", figure=_empty_fig(), config={"displayModeBar": False}),
                        style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "8px",
                               "gridColumn": "span 2"},
                    ),
                ],
            ),

            # ── Holdings Table ──
            html.Div(
                id="holdings-table-container",
                style={"backgroundColor": CARD_BG, "borderRadius": "6px", "padding": "16px"},
            ),
        ],
    )

    _register_callbacks(app)
    return app


def _register_callbacks(app: dash.Dash):
    """Register all Dash callbacks."""

    @app.callback(
        [
            Output("analysis-store", "data"),
            Output("status-msg", "children"),
            Output("kpi-row", "children"),
            Output("chart-cumret", "figure"),
            Output("chart-allocation", "figure"),
            Output("chart-rolling-vol", "figure"),
            Output("chart-corr", "figure"),
            Output("chart-var-dist", "figure"),
            Output("chart-drawdown", "figure"),
            Output("chart-risk-contrib", "figure"),
            Output("holdings-table-container", "children"),
        ],
        Input("run-btn", "n_clicks"),
        [
            State("upload-portfolio", "contents"),
            State("upload-portfolio", "filename"),
            State("start-date", "date"),
            State("end-date", "date"),
            State("confidence-level", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, contents, filename, start_date, end_date, confidence):
        if not contents:
            return (
                None, "Please upload a portfolio file.", [],
                _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(),
                _empty_fig(), _empty_fig(), _empty_fig(), "",
            )

        try:
            # Decode uploaded file
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            # Save temp file and load
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            tmp.write(decoded)
            tmp.close()

            portfolio = load_portfolio(tmp.name)
            os.unlink(tmp.name)

            start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

            # Fetch data (Bloomberg or mock fallback)
            asset_prices, benchmark_prices = fetch_portfolio_data(portfolio, start, end)

            # Compute returns
            asset_ret = simple_returns(asset_prices)
            bench_ret = simple_returns(benchmark_prices)
            weights = pd.Series(
                portfolio["Weight"].values, index=portfolio["Ticker"].values
            )
            port_ret = portfolio_returns(asset_ret, weights)

            # ── KPI Cards ──
            total_ret = cumulative_returns(port_ret).iloc[-1]
            ann_vol = annualized_volatility(port_ret)
            sharpe_val = sharpe_ratio(port_ret)
            var_95 = historical_var(port_ret, confidence)
            mdd = max_drawdown(port_ret)

            def _fmt_pct(v):
                return f"{v:+.2%}" if v != 0 else "0.00%"

            kpi_cards = [
                _kpi_card("Total Return", _fmt_pct(total_ret),
                          GREEN if total_ret >= 0 else RED),
                _kpi_card("Ann. Volatility", f"{ann_vol:.2%}", ORANGE),
                _kpi_card("Sharpe Ratio", f"{sharpe_val:.2f}",
                          GREEN if sharpe_val > 1 else ORANGE if sharpe_val > 0 else RED),
                _kpi_card(f"VaR ({confidence:.0%})", f"{var_95:.2%}", RED),
                _kpi_card("Max Drawdown", f"{mdd:.2%}", RED),
            ]

            # ── Charts ──
            # 1. Cumulative returns
            cum_port = cumulative_returns(port_ret)
            fig_cumret = _make_fig(
                title="Cumulative Returns", height=350,
                yaxis_tickformat=".0%", margin=dict(t=40, b=30, l=50, r=20),
                legend=dict(orientation="h", y=-0.15),
            )
            fig_cumret.add_trace(go.Scatter(
                x=cum_port.index, y=cum_port.values,
                name="Portfolio", line=dict(color=GREEN, width=2),
            ))
            for bench_ticker in portfolio["Benchmark"].unique():
                if bench_ticker in bench_ret.columns:
                    cum_bench = cumulative_returns(bench_ret[bench_ticker])
                    fig_cumret.add_trace(go.Scatter(
                        x=cum_bench.index, y=cum_bench.values,
                        name=bench_ticker, line=dict(width=1, dash="dash"),
                    ))

            # 2. Allocation donut
            ac_weights = portfolio.groupby("Asset Class")["Weight"].sum()
            fig_alloc = _make_fig(
                title="Asset Class Allocation", height=350,
                margin=dict(t=40, b=30, l=20, r=20),
                showlegend=False,
            )
            fig_alloc.add_trace(go.Pie(
                labels=ac_weights.index, values=ac_weights.values,
                hole=0.5, textinfo="label+percent",
                marker=dict(colors=[GREEN, BLUE, ORANGE, RED, "#AB47BC", "#26C6DA"]),
            ))

            # 3. Rolling volatility (63-day)
            fig_rvol = _make_fig(
                title="Rolling 63-Day Volatility", height=350,
                yaxis_tickformat=".0%", margin=dict(t=40, b=30, l=50, r=20),
                legend=dict(orientation="h", y=-0.15),
            )
            window = 63
            port_rolling = port_ret.rolling(window).std() * np.sqrt(252)
            fig_rvol.add_trace(go.Scatter(
                x=port_rolling.index, y=port_rolling.values,
                name="Portfolio", line=dict(color=GREEN, width=2),
            ))
            for ac, group in portfolio.groupby("Asset Class"):
                tickers = group["Ticker"].tolist()
                w = group["Weight"].values
                ac_ret = asset_ret[tickers].fillna(0).dot(w)
                ac_rolling = ac_ret.rolling(window).std() * np.sqrt(252)
                fig_rvol.add_trace(go.Scatter(
                    x=ac_rolling.index, y=ac_rolling.values,
                    name=ac, line=dict(width=1),
                ))

            # 4. Correlation heatmap
            corr = correlation_matrix(asset_ret)
            short_labels = [t.split()[0] for t in corr.columns]
            fig_corr = _make_fig(
                title="Correlation Matrix", height=350,
                margin=dict(t=40, b=30, l=80, r=20),
            )
            fig_corr.add_trace(go.Heatmap(
                z=corr.values, x=short_labels, y=short_labels,
                colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2), texttemplate="%{text}",
            ))

            # 5. VaR distribution histogram
            fig_var = _make_fig(
                title="Return Distribution & VaR", height=350,
                xaxis_tickformat=".1%", margin=dict(t=40, b=30, l=50, r=20),
                showlegend=False,
            )
            fig_var.add_trace(go.Histogram(
                x=port_ret.values, nbinsx=80,
                marker_color=BLUE, opacity=0.7, name="Daily Returns",
            ))
            var_val = historical_var(port_ret, confidence)
            fig_var.add_vline(
                x=-var_val, line_dash="dash", line_color=RED,
                annotation_text=f"VaR {confidence:.0%}: {var_val:.2%}",
                annotation_font_color=RED,
            )
            cvar_val = conditional_var(port_ret, confidence)
            fig_var.add_vline(
                x=-cvar_val, line_dash="dot", line_color=ORANGE,
                annotation_text=f"CVaR: {cvar_val:.2%}",
                annotation_font_color=ORANGE,
            )

            # 6. Drawdown chart
            dd = drawdown_series(port_ret)
            fig_dd = _make_fig(
                title="Portfolio Drawdown", height=350,
                yaxis_tickformat=".0%", margin=dict(t=40, b=30, l=50, r=20),
            )
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values, fill="tozeroy",
                line=dict(color=RED, width=1), fillcolor="rgba(255,23,68,0.3)",
                name="Drawdown",
            ))

            # 7. Risk contribution by asset class
            rc = risk_contribution(asset_ret, weights)
            rc_by_ac = {}
            for ac, group in portfolio.groupby("Asset Class"):
                tickers = group["Ticker"].tolist()
                rc_by_ac[ac] = rc[tickers].sum()
            rc_series = pd.Series(rc_by_ac)

            fig_rc = _make_fig(
                title="Risk Contribution by Asset Class", height=300,
                yaxis_tickformat=".0%", margin=dict(t=40, b=30, l=50, r=20),
            )
            fig_rc.add_trace(go.Bar(
                x=rc_series.index, y=rc_series.values,
                marker_color=[GREEN, BLUE, ORANGE, RED, "#AB47BC", "#26C6DA"][:len(rc_series)],
                text=[f"{v:.1%}" for v in rc_series.values],
                textposition="auto",
            ))

            # ── Holdings Table ──
            detail = compute_asset_metrics(asset_ret, bench_ret, portfolio)
            fmt_cols = ["Ann. Return", "Ann. Volatility", "Max Drawdown", "Weight", "Tracking Error"]
            for c in fmt_cols:
                if c in detail.columns:
                    detail[c] = detail[c].apply(lambda v: f"{v:.2%}" if pd.notna(v) else "")
            for c in ["Sharpe", "Sortino", "Beta"]:
                if c in detail.columns:
                    detail[c] = detail[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "")

            table = dash_table.DataTable(
                data=detail.to_dict("records"),
                columns=[{"name": c, "id": c} for c in detail.columns],
                sort_action="native",
                style_header={
                    "backgroundColor": "#1B2A3D", "color": HEADER_GREEN,
                    "fontWeight": "bold", "border": "1px solid #2A3050",
                    "fontFamily": "Consolas, monospace",
                },
                style_cell={
                    "backgroundColor": CARD_BG, "color": TEXT_COLOR,
                    "border": "1px solid #2A3050", "textAlign": "right",
                    "fontFamily": "Consolas, monospace", "fontSize": "13px",
                    "padding": "8px",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Ticker"}, "textAlign": "left"},
                    {"if": {"column_id": "Name"}, "textAlign": "left"},
                    {"if": {"column_id": "Asset Class"}, "textAlign": "left"},
                ],
            )

            holdings_section = html.Div([
                html.H3("Holdings Detail", style={"color": HEADER_GREEN, "marginBottom": "12px"}),
                table,
            ])

            store_data = {
                "portfolio_file": filename,
                "start_date": str(start),
                "end_date": str(end),
            }

            return (
                store_data, f"Analysis complete - {len(portfolio)} holdings analyzed.",
                kpi_cards, fig_cumret, fig_alloc, fig_rvol, fig_corr,
                fig_var, fig_dd, fig_rc, holdings_section,
            )

        except Exception as e:
            empty = _empty_fig()
            return (
                None, f"Error: {str(e)}", [],
                empty, empty, empty, empty, empty, empty, empty, "",
            )

    @app.callback(
        Output("download-report", "data"),
        Input("export-btn", "n_clicks"),
        [
            State("upload-portfolio", "contents"),
            State("upload-portfolio", "filename"),
            State("start-date", "date"),
            State("end-date", "date"),
        ],
        prevent_initial_call=True,
    )
    def export_excel(n_clicks, contents, filename, start_date, end_date):
        if not contents:
            return None

        try:
            content_type, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)

            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            tmp.write(decoded)
            tmp.close()

            portfolio = load_portfolio(tmp.name)
            os.unlink(tmp.name)

            start = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date
            end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date

            asset_prices, benchmark_prices = fetch_portfolio_data(portfolio, start, end)
            asset_ret = simple_returns(asset_prices)
            bench_ret = simple_returns(benchmark_prices)
            weights = pd.Series(portfolio["Weight"].values, index=portfolio["Ticker"].values)

            output_file = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            output_file.close()

            generate_report(portfolio, asset_ret, bench_ret, weights, output_file.name)

            return dcc.send_file(output_file.name, filename="portfolio_risk_report.xlsx")

        except Exception:
            return None


def _empty_fig() -> go.Figure:
    """Return an empty figure with dark theme."""
    fig = _make_fig(
        height=350, margin=dict(t=40, b=30, l=50, r=20),
        annotations=[dict(
            text="Upload portfolio and run analysis",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="#666", size=14),
        )],
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    """Launch the dashboard server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)
