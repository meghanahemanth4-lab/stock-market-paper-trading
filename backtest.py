
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="StockSense AI | Backtesting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 StockSense AI")
st.header("Historical Strategy Backtesting V2")
st.write(
    "Compare a moving-average strategy with buy-and-hold "
    "using historical market data."
)

# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------

symbol = st.text_input("Stock ticker", value="TCS.NS").strip().upper()

period = st.selectbox(
    "Historical period",
    ["1y", "2y", "5y"],
    index=1
)

initial_capital = st.number_input(
    "Initial capital (₹)",
    min_value=1000.0,
    value=100000.0,
    step=10000.0
)

transaction_cost_pct = st.number_input(
    "Transaction cost per position change (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.10,
    step=0.05
)

run_backtest = st.button("Run Backtest", type="primary")


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

def calculate_metrics(values, daily_returns, initial_value):
    final_value = float(values.iloc[-1])

    total_return = (final_value / initial_value - 1) * 100

    returns = daily_returns.dropna()

    if len(returns) > 1:
        annualized_volatility = (
            returns.std(ddof=1) * np.sqrt(252) * 100
        )
    else:
        annualized_volatility = np.nan

    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = (
            returns.mean() / returns.std(ddof=1)
        ) * np.sqrt(252)
    else:
        sharpe = np.nan

    peak = values.cummax()
    drawdown = values / peak - 1
    max_drawdown = drawdown.min() * 100

    return {
        "Final Value": final_value,
        "Total Return (%)": total_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown (%)": max_drawdown,
    }


# --------------------------------------------------
# RUN SIMULATION
# --------------------------------------------------

if run_backtest:

    if not symbol:
        st.error("Enter a valid stock ticker.")
        st.stop()

    try:
        with st.spinner("Downloading historical data..."):
            raw = yf.Ticker(symbol).history(
                period=period,
                interval="1d",
                auto_adjust=True
            )

        if raw.empty:
            st.error("No data found. Check the ticker symbol.")
            st.stop()

        raw = raw.dropna(
            subset=["Open", "High", "Low", "Close"]
        ).copy()

        # Keep the full price history for the benchmark.
        if len(raw) < 60:
            st.error("Not enough data for a 20/50-day strategy.")
            st.stop()

        # --------------------------------------------------
        # INDICATORS AND SIGNALS
        # --------------------------------------------------

        raw["SMA20"] = raw["Close"].rolling(20).mean()
        raw["SMA50"] = raw["Close"].rolling(50).mean()

        raw["Signal"] = (
            raw["SMA20"] > raw["SMA50"]
        ).astype(float)

        # A signal from yesterday determines today's exposure.
        raw["Position"] = raw["Signal"].shift(1).fillna(0.0)

        # --------------------------------------------------
        # RETURNS AND TRANSACTION COSTS
        # --------------------------------------------------

        raw["Market_Return"] = raw["Close"].pct_change()

        # The first return is unavailable because there is
        # no previous closing price in the downloaded data.
        raw = raw.dropna(subset=["Market_Return"]).copy()

        # Buy-and-hold starts on the same evaluation date.
        raw["BuyHold_Return"] = raw["Market_Return"]

        # A change from cash to stock or stock to cash
        # incurs one position-change cost.
        raw["Turnover"] = (
            raw["Position"].diff().abs()
        )

        if len(raw):
            raw.iloc[0, raw.columns.get_loc("Turnover")] = abs(
                raw["Position"].iloc[0]
            )

        cost_rate = transaction_cost_pct / 100.0

        raw["Strategy_Return"] = (
            raw["Position"] * raw["Market_Return"]
            - raw["Turnover"].fillna(0) * cost_rate
        )

        # Prevent returns below -100% from invalidating
        # compounded portfolio values.
        raw["Strategy_Return"] = raw["Strategy_Return"].clip(
            lower=-1.0
        )

        raw["Strategy_Value"] = (
            initial_capital
            * (1 + raw["Strategy_Return"]).cumprod()
        )

        raw["BuyHold_Value"] = (
            initial_capital
            * (1 + raw["BuyHold_Return"]).cumprod()
        )

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        strategy_metrics = calculate_metrics(
            raw["Strategy_Value"],
            raw["Strategy_Return"],
            initial_capital
        )

        benchmark_metrics = calculate_metrics(
            raw["BuyHold_Value"],
            raw["BuyHold_Return"],
            initial_capital
        )

        st.subheader("Performance Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Strategy Final Value",
            f"₹{strategy_metrics['Final Value']:,.2f}"
        )

        c2.metric(
            "Strategy Return",
            f"{strategy_metrics['Total Return (%)']:.2f}%"
        )

        c3.metric(
            "Buy-and-Hold Return",
            f"{benchmark_metrics['Total Return (%)']:.2f}%"
        )

        c4.metric(
            "Evaluation Sessions",
            f"{len(raw):,}"
        )

        # --------------------------------------------------
        # METRICS COMPARISON
        # --------------------------------------------------

        comparison = pd.DataFrame({
            "Metric": [
                "Final Portfolio Value",
                "Total Return (%)",
                "Annualized Volatility (%)",
                "Sharpe Ratio",
                "Maximum Drawdown (%)"
            ],
            "SMA Strategy": [
                strategy_metrics["Final Value"],
                strategy_metrics["Total Return (%)"],
                strategy_metrics["Annualized Volatility (%)"],
                strategy_metrics["Sharpe Ratio"],
                strategy_metrics["Maximum Drawdown (%)"]
            ],
            "Buy and Hold": [
                benchmark_metrics["Final Value"],
                benchmark_metrics["Total Return (%)"],
                benchmark_metrics["Annualized Volatility (%)"],
                benchmark_metrics["Sharpe Ratio"],
                benchmark_metrics["Maximum Drawdown (%)"]
            ]
        })

        st.subheader("Strategy vs Benchmark")

        st.dataframe(
            comparison.round(2),
            hide_index=True,
            use_container_width=True
        )

        # --------------------------------------------------
        # PORTFOLIO CHART
        # --------------------------------------------------

        st.subheader("Portfolio Value Over Time")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=raw.index,
                y=raw["Strategy_Value"],
                name="SMA Strategy",
                mode="lines"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=raw.index,
                y=raw["BuyHold_Value"],
                name="Buy and Hold",
                mode="lines"
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            hovermode="x unified",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------
        # DRAWDOWN CHART
        # --------------------------------------------------

        st.subheader("Portfolio Drawdown")

        drawdown_fig = go.Figure()

        for column, label in [
            ("Strategy_Value", "SMA Strategy"),
            ("BuyHold_Value", "Buy and Hold")
        ]:
            drawdown = (
                raw[column] / raw[column].cummax() - 1
            ) * 100

            drawdown_fig.add_trace(
                go.Scatter(
                    x=raw.index,
                    y=drawdown,
                    name=label,
                    mode="lines"
                )
            )

        drawdown_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=350,
            hovermode="x unified"
        )

        st.plotly_chart(
            drawdown_fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # HISTORICAL DETAILS
        # --------------------------------------------------

        st.subheader("Recent Backtest Data")

        columns = [
            "Close",
            "SMA20",
            "SMA50",
            "Signal",
            "Position",
            "Market_Return",
            "Turnover",
            "Strategy_Return",
            "Strategy_Value",
            "BuyHold_Value"
        ]

        st.dataframe(
            raw[columns].tail(30).round(4),
            use_container_width=True
        )

        st.download_button(
            "Download Backtest CSV",
            data=raw.to_csv().encode("utf-8"),
            file_name=f"{symbol.replace('.', '_')}_backtest_v2.csv",
            mime="text/csv"
        )

        st.info(
            "The strategy uses previous-session signals and "
            "simplified proportional transaction costs. "
            "This is a long-or-cash simulation, not a broker "
            "execution simulator."
        )

        st.caption(
            "Historical results are not forecasts. This prototype "
            "does not fully model taxes, slippage, bid-ask spreads, "
            "corporate-action timing, or actual order fills."
        )

    except Exception as error:
        st.error(
            "The backtest failed. Check your internet connection "
            "and ticker symbol."
        )
        st.exception(error)
