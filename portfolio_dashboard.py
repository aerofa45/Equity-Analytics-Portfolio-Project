import os
print("RUNNING DASHBOARD FROM:", os.path.abspath(__file__))

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
import streamlit as st


DB_PATH = "portfolio.db"
BENCHMARK = "SPY"


def load_price_data_from_db(tickers, start, end):
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" for _ in tickers)

    query = f"""
        SELECT 
            Date,
            ticker,
            adj_close
        FROM prices
        WHERE ticker IN ({placeholders})
          AND Date(Date) BETWEEN Date(?) AND Date(?)
        ORDER BY Date, ticker;
    """

    params = list(tickers) + [start.isoformat(), end.isoformat()]
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["Date"])
    conn.close()

    if df.empty:
        return pd.DataFrame()

    prices = df.pivot(index="Date", columns="ticker", values="adj_close")
    prices = prices.sort_index().dropna(how="all")
    return prices



def load_weights_from_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT ticker, weight FROM portfolio_weights", conn)
    conn.close()
    return df.set_index("ticker")["weight"]


def compute_portfolio_metrics(price_df, weights, rf_rate_annual=0.02):
    daily_returns = price_df.pct_change().dropna()
    if daily_returns.empty:
        return None

    weights = np.array(weights)
    weights = weights / weights.sum()

    port_daily_returns = (daily_returns * weights).sum(axis=1)
    cum_portfolio = (1 + port_daily_returns).cumprod()
    cum_assets = (1 + daily_returns).cumprod()

    n_days = len(port_daily_returns)
    total_return_port = cum_portfolio.iloc[-1] - 1
    ann_return_port = (1 + total_return_port) ** (252 / n_days) - 1
    ann_vol_port = port_daily_returns.std() * np.sqrt(252)
    sharpe_port = (ann_return_port - rf_rate_annual) / ann_vol_port if ann_vol_port > 0 else np.nan

    running_max = cum_portfolio.cummax()
    drawdown = cum_portfolio / running_max - 1
    max_drawdown = drawdown.min()

    asset_stats = []
    for col in daily_returns.columns:
        r = daily_returns[col]
        cum = cum_assets[col]
        total_ret = cum.iloc[-1] - 1
        ann_ret = (1 + total_ret) ** (252 / n_days) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = (ann_ret - rf_rate_annual) / ann_vol if ann_vol > 0 else np.nan
        asset_stats.append(
            {
                "Ticker": col,
                "Total Return": total_ret,
                "Annualized Return": ann_ret,
                "Annualized Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
            }
        )
    asset_stats_df = pd.DataFrame(asset_stats).set_index("Ticker")
    corr = daily_returns.corr()

    metrics = {
        "total_return_port": total_return_port,
        "ann_return_port": ann_return_port,
        "ann_vol_port": ann_vol_port,
        "sharpe_port": sharpe_port,
        "max_drawdown": max_drawdown,
    }

    return {
        "daily_returns": daily_returns,
        "port_daily_returns": port_daily_returns,
        "cum_portfolio": cum_portfolio,
        "cum_assets": cum_assets,
        "metrics": metrics,
        "asset_stats": asset_stats_df,
        "corr": corr,
    }


def format_pct(x, decimals=1):
    if pd.isna(x):
        return "—"
    return f"{x * 100:.{decimals}f}%"


st.set_page_config(page_title="Equity Portfolio Dashboard", layout="wide")

st.title("Equity Portfolio Analytics Dashboard")
st.write("Created By: Aman")
st.caption("Data source: SQLite database `portfolio.db` populated via the Jupyter notebook.")

st.sidebar.header("Controls")
end_date = st.sidebar.date_input("End date", value=date.today())
start_date = st.sidebar.date_input("Start date", value=end_date - timedelta(days=365))
rf_rate = st.sidebar.slider("Risk-free rate (annual, %)", 0.0, 5.0, 2.0, 0.25) / 100.0

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")

try:
    weights = load_weights_from_db()
except Exception as e:
    st.error("Could not load weights from database. Run the notebook first.")
    st.stop()

selected_tickers = st.sidebar.multiselect(
    "Select tickers",
    options=list(weights.index),
    default=list(weights.index),
)

if not selected_tickers:
    st.warning("Select at least one ticker.")
    st.stop()

price_df = load_price_data_from_db(selected_tickers, start_date, end_date)

if price_df.empty:
    st.error("No price data for this date range. Run the notebook to refresh data.")
    st.stop()

weights_selected = weights.loc[selected_tickers].values
results = compute_portfolio_metrics(price_df, weights_selected, rf_rate_annual=rf_rate)
if results is None:
    st.error("Unable to compute metrics.")
    st.stop()

daily_returns = results["daily_returns"]
port_daily_returns = results["port_daily_returns"]
cum_portfolio = results["cum_portfolio"]
cum_assets = results["cum_assets"]
metrics = results["metrics"]
asset_stats = results["asset_stats"]
corr = results["corr"]

benchmark_series = None
if BENCHMARK in price_df.columns:
    b = price_df[BENCHMARK].pct_change().dropna()
    bench_cum = (1 + b).cumprod()
    benchmark_series = bench_cum.reindex(cum_portfolio.index).dropna()
    bench_total = benchmark_series.iloc[-1] - 1
    bench_ann = (1 + bench_total) ** (252 / len(benchmark_series)) - 1
else:
    bench_ann = np.nan

outperformance = metrics["ann_return_port"] - bench_ann if not math.isnan(bench_ann) else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Return", format_pct(metrics["total_return_port"]))
c2.metric("Annualized Return", format_pct(metrics["ann_return_port"]))
c3.metric("Annualized Volatility", format_pct(metrics["ann_vol_port"]))
c4.metric("Sharpe Ratio", f"{metrics['sharpe_port']:.2f}" if not pd.isna(metrics["sharpe_port"]) else "—")
c5.metric("Max Drawdown", format_pct(metrics["max_drawdown"]))

if not math.isnan(bench_ann):
    st.caption(
        f"Benchmark ({BENCHMARK}) annualized return: {format_pct(bench_ann)}, "
        f"portfolio outperformance: {format_pct(outperformance)}."
    )

st.subheader("Performance Overview")
perf_df = pd.DataFrame({"Portfolio": cum_portfolio})
if benchmark_series is not None:
    perf_df["Benchmark"] = benchmark_series

perf_long = perf_df.reset_index().melt(id_vars="Date", var_name="Series", value_name="Cumulative Value")
fig_perf = px.line(perf_long, x="Date", y="Cumulative Value", color="Series", title="Cumulative Performance")
st.plotly_chart(fig_perf, use_container_width=True)

asset_stats_display = asset_stats.copy()
asset_stats_display["Weight"] = weights.loc[asset_stats_display.index]
asset_stats_display = asset_stats_display.sort_values("Annualized Return", ascending=False)

col_ret, col_vol = st.columns(2)
fig_ret = px.bar(asset_stats_display.reset_index(), x="Ticker", y="Annualized Return",
                 hover_data=["Total Return", "Annualized Volatility", "Sharpe Ratio", "Weight"],
                 title="Annualized Return by Asset")
fig_vol = px.bar(asset_stats_display.reset_index(), x="Ticker", y="Annualized Volatility",
                 hover_data=["Annualized Return", "Sharpe Ratio", "Weight"],
                 title="Annualized Volatility by Asset")
col_ret.plotly_chart(fig_ret, use_container_width=True)
col_vol.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Risk–Return & Diversification")
scatter_df = asset_stats_display.reset_index()
scatter_df["Weight (%)"] = scatter_df["Weight"] * 100
fig_scatter = px.scatter(scatter_df, x="Annualized Volatility", y="Annualized Return",
                         size="Weight (%)", color="Ticker",
                         hover_data=["Sharpe Ratio", "Weight (%)"],
                         title="Risk–Return Profile")
fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                     title="Correlation of Daily Returns",
                     color_continuous_scale="RdBu", zmin=-1, zmax=1)
col_sc, col_ch = st.columns(2)
col_sc.plotly_chart(fig_scatter, use_container_width=True)
col_ch.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Table & Insights")
st.dataframe(
    asset_stats_display.style.format(
        {
            "Total Return": "{:.2%}",
            "Annualized Return": "{:.2%}",
            "Annualized Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
            "Weight": "{:.2%}",
        }
    ),
    use_container_width=True,
)

best_asset = asset_stats_display["Annualized Return"].idxmax()
worst_asset = asset_stats_display["Annualized Return"].idxmin()
best_ret = asset_stats_display.loc[best_asset, "Annualized Return"]
worst_ret = asset_stats_display.loc[worst_asset, "Annualized Return"]

vol_asset = asset_stats_display["Annualized Volatility"].idxmax()
vol_val = asset_stats_display.loc[vol_asset, "Annualized Volatility"]

corr_matrix = corr.copy()
np.fill_diagonal(corr_matrix.values, np.nan)
max_corr_val = corr_matrix.max().max()
max_corr_pair = corr_matrix.stack().idxmax()
min_corr_val = corr_matrix.min().min()
min_corr_pair = corr_matrix.stack().idxmin()

st.markdown("### Key Insights")
st.write(f"- Best performer: **{best_asset}** with annualized return {format_pct(best_ret)}.")
st.write(f"- Weakest performer: **{worst_asset}** with annualized return {format_pct(worst_ret)}.")
st.write(f"- Most volatile asset: **{vol_asset}** with volatility {format_pct(vol_val)}.")
st.write(
    f"- Most positively correlated pair: **{max_corr_pair[0]} & {max_corr_pair[1]}** "
    f"(ρ ≈ {max_corr_val:.2f}) – low diversification."
)
st.write(
    f"- Most diversifying pair: **{min_corr_pair[0]} & {min_corr_pair[1]}** "
    f"(ρ ≈ {min_corr_val:.2f})."
)
if not math.isnan(outperformance):
    if outperformance > 0:
        st.write(f"- Portfolio **outperforms** benchmark by {format_pct(outperformance)} annually.")
    else:
        st.write(f"- Portfolio **underperforms** benchmark by {format_pct(-outperformance)} annually.")

st.caption("Run `portfolio_end_to_end.ipynb` regularly to refresh data in the database.")
