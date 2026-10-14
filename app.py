
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. LOAD STOCK DATA
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(symbol, period):
    stock = yf.Ticker(symbol)

    data = stock.history(
        period=period,
        interval="1d",
        auto_adjust=True
    )

    if data.empty:
        return pd.DataFrame()

    # Remove rows without essential price information.
    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    ).copy()

    if data.empty:
        return pd.DataFrame()

    # Simple Moving Averages
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    # Exponential Moving Average
    data["EMA20"] = data["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    # Relative Strength Index
    data["RSI"] = ta.momentum.RSIIndicator(
        close=data["Close"],
        window=14
    ).rsi()

    # MACD
    macd = ta.trend.MACD(
        close=data["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Histogram"] = macd.macd_diff()

    # Bollinger Bands
    bands = ta.volatility.BollingerBands(
        close=data["Close"],
        window=20,
        window_dev=2
    )

    data["BB_Upper"] = bands.bollinger_hband()
    data["BB_Middle"] = bands.bollinger_mavg()
    data["BB_Lower"] = bands.bollinger_lband()

    # Daily percentage return
    data["Daily_Return"] = data["Close"].pct_change() * 100

    # Rolling volatility based on daily returns
    data["Volatility20"] = (
        data["Daily_Return"].rolling(20).std()
    )

    return data


# =========================================================
# 4. APPLICATION HEADER
# =========================================================

st.title("📈 StockSense AI")

st.markdown(
    """
    **Intelligent Stock Market Analysis & Paper Trading Platform**

    Explore market prices, technical indicators, and
    rule-based trading signals using historical market data.
    """
)

st.caption(
    "Educational project · Market data may be delayed. "
    "Signals are analytical indicators, not financial advice."
)


# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Dashboard Settings")

popular_stocks = {
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA"
}

selected_name = st.sidebar.selectbox(
    "Choose a stock",
    list(popular_stocks.keys())
)

custom_symbol = st.sidebar.text_input(
    "Or enter a ticker symbol",
    placeholder="e.g. WIPRO.NS"
).strip().upper()

ticker_symbol = custom_symbol or popular_stocks[selected_name]

period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Historical period",
    list(period_options.keys()),
    index=3
)

period = period_options[selected_period]

if st.sidebar.button("🔄 Refresh Market Data", use_container_width=True):
    load_stock_data.clear()
    st.rerun()


# =========================================================
# 6. FETCH DATA
# =========================================================

try:
    with st.spinner(f"Loading market data for {ticker_symbol}..."):
        data = load_stock_data(ticker_symbol, period)

except Exception as error:
    st.error(
        "Could not retrieve market data. "
        "Check your internet connection and ticker symbol."
    )
    st.caption(f"Technical details: {error}")
    st.stop()


if data.empty:
    st.error(
        f"No market data found for {ticker_symbol}. "
        "Check the ticker symbol and try again."
    )
    st.info(
        "Examples: TCS.NS, INFY.NS, RELIANCE.NS, AAPL, MSFT"
    )
    st.stop()


# =========================================================
# 7. CALCULATE MARKET METRICS
# =========================================================

latest = data.iloc[-1]

current_price = float(latest["Close"])

if len(data) >= 2:
    previous_close = float(data["Close"].iloc[-2])
    price_change = current_price - previous_close

    percentage_change = (
        price_change / previous_close * 100
        if previous_close != 0
        else 0
    )
else:
    previous_close = None
    price_change = None
    percentage_change = None

period_high = float(data["High"].max())
period_low = float(data["Low"].min())

latest_volume = int(latest["Volume"])

last_date = data.index[-1].strftime("%d %b %Y")


# =========================================================
# 8. STOCK OVERVIEW
# =========================================================

st.subheader(f"📊 Market Overview — {ticker_symbol}")

st.caption(
    f"Latest available session: {last_date} · "
    f"Selected period: {selected_period}"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest Close",
        f"₹{current_price:,.2f}"
    )

with col2:
    if percentage_change is not None:
        st.metric(
            "Change vs Previous Session",
            f"₹{price_change:,.2f}",
            f"{percentage_change:+.2f}%"
        )
    else:
        st.metric("Session Change", "N/A")

with col3:
    st.metric(
        f"High ({selected_period})",
        f"₹{period_high:,.2f}"
    )

with col4:
    st.metric(
        f"Low ({selected_period})",
        f"₹{period_low:,.2f}"
    )

col5, col6 = st.columns(2)

with col5:
    st.metric(
        "Latest Reported Volume",
        f"{latest_volume:,}"
    )

with col6:
    st.metric(
        "Data Points",
        f"{len(data):,}"
    )


st.divider()


# =========================================================
# 9. PRICE CHART AND MOVING AVERAGES
# =========================================================

st.subheader("🕯️ Price Chart & Moving Averages")

chart = go.Figure()

chart.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlesticks"
    )
)

chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["SMA20"],
        mode="lines",
        name="SMA 20",
        line=dict(width=1.5)
    )
)

chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["SMA50"],
        mode="lines",
        name="SMA 50",
        line=dict(width=1.5)
    )
)

# Show SMA200 only when enough history exists.
if data["SMA200"].notna().any():
    chart.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA200"],
            mode="lines",
            name="SMA 200",
            line=dict(width=1.5)
        )
    )

chart.update_layout(
    title=f"{ticker_symbol} Historical Price",
    xaxis_title="Date",
    yaxis_title="Adjusted Price",
    height=550,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(chart, use_container_width=True)

st.caption(
    "SMA lines are calculated from adjusted closing prices. "
    "Longer moving averages require more historical observations."
)


# =========================================================
# 10. TECHNICAL INDICATOR SUMMARY
# =========================================================

st.subheader("🧭 Technical Indicator Summary")

rsi_value = latest["RSI"]
macd_value = latest["MACD"]
macd_signal_value = latest["MACD_Signal"]
sma20_value = latest["SMA20"]
sma50_value = latest["SMA50"]

col1, col2, col3 = st.columns(3)

with col1:
    if pd.notna(sma20_value):
        st.metric("SMA 20", f"₹{sma20_value:,.2f}")
    else:
        st.metric("SMA 20", "Insufficient data")

with col2:
    if pd.notna(sma50_value):
        st.metric("SMA 50", f"₹{sma50_value:,.2f}")
    else:
        st.metric("SMA 50", "Insufficient data")

with col3:
    if pd.notna(rsi_value):
        st.metric("RSI (14)", f"{rsi_value:.2f}")
    else:
        st.metric("RSI (14)", "Insufficient data")


# =========================================================
# 11. RSI CHART
# =========================================================

st.subheader("📉 Relative Strength Index (RSI)")

rsi_chart = go.Figure()

rsi_chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["RSI"],
        mode="lines",
        name="RSI (14)"
    )
)

rsi_chart.add_hline(
    y=70,
    line_dash="dash",
    annotation_text="70"
)

rsi_chart.add_hline(
    y=30,
    line_dash="dash",
    annotation_text="30"
)

rsi_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI",
    yaxis=dict(range=[0, 100]),
    height=350,
    hovermode="x unified"
)

st.plotly_chart(rsi_chart, use_container_width=True)

if pd.notna(rsi_value):
    if rsi_value >= 70:
        st.info(
            "RSI is at or above 70. This is commonly "
            "described as an overbought reading."
        )
    elif rsi_value <= 30:
        st.info(
            "RSI is at or below 30. This is commonly "
            "described as an oversold reading."
        )
    else:
        st.write(
            f"RSI is {rsi_value:.2f}, between the "
            "commonly monitored 30 and 70 levels."
        )


# =========================================================
# 12. MACD CHART
# =========================================================

st.subheader("📈 MACD — Momentum Indicator")

macd_chart = go.Figure()

macd_chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MACD"],
        mode="lines",
        name="MACD"
    )
)

macd_chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["MACD_Signal"],
        mode="lines",
        name="Signal Line"
    )
)

macd_chart.add_trace(
    go.Bar(
        x=data.index,
        y=data["MACD_Histogram"],
        name="Histogram",
        opacity=0.5
    )
)

macd_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="MACD",
    height=400,
    hovermode="x unified"
)

st.plotly_chart(macd_chart, use_container_width=True)


# =========================================================
# 13. BOLLINGER BANDS
# =========================================================

st.subheader("📏 Bollinger Bands")

bb_chart = go.Figure()

bb_chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["BB_Upper"],
        mode="lines",
        name="Upper Band",
        line=dict(dash="dash")
    )
)

bb_chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["BB_Lower"],
        mode="lines",
        name="Lower Band",
        line=dict(dash="dash"),
        fill="tonexty",
        opacity=0.2
    )
)

bb_chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["BB_Middle"],
        mode="lines",
        name="Middle Band"
    )
)

bb_chart.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        mode="lines",
        name="Close Price"
    )
)

bb_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=400,
    hovermode="x unified"
)

st.plotly_chart(bb_chart, use_container_width=True)


# =========================================================
# 14. TRADING VOLUME
# =========================================================

st.subheader("📊 Trading Volume")

volume_chart = go.Figure()

volume_chart.add_trace(
    go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume"
    )
)

volume_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="Shares Traded",
    height=300,
    hovermode="x unified"
)

st.plotly_chart(volume_chart, use_container_width=True)


# =========================================================
# 15. RULE-BASED SIGNAL (NOT MACHINE LEARNING)
# =========================================================

st.subheader("🧠 Technical Signal — Experimental")

# This is a transparent indicator rule, not an ML prediction.
required_values = [
    latest["SMA20"],
    latest["SMA50"],
    latest["RSI"],
    latest["MACD"],
    latest["MACD_Signal"]
]

if any(pd.isna(value) for value in required_values):
    st.warning(
        "There is not enough historical data to calculate "
        "all indicators. Select a longer period."
    )

else:
    bullish_conditions = 0
    bearish_conditions = 0

    if latest["SMA20"] > latest["SMA50"]:
        bullish_conditions += 1
    elif latest["SMA20"] < latest["SMA50"]:
        bearish_conditions += 1

    if latest["MACD"] > latest["MACD_Signal"]:
        bullish_conditions += 1
    elif latest["MACD"] < latest["MACD_Signal"]:
        bearish_conditions += 1

    if 50 < latest["RSI"] < 70:
        bullish_conditions += 1
    elif 30 < latest["RSI"] < 50:
        bearish_conditions += 1

    if bullish_conditions >= 2:
        signal = "Bullish technical conditions"
    elif bearish_conditions >= 2:
        signal = "Bearish technical conditions"
    else:
        signal = "Mixed / neutral conditions"

    st.info(f"Indicator summary: **{signal}**")

    st.write(
        f"Bullish conditions: {bullish_conditions} of 3"
    )

    st.write(
        f"Bearish conditions: {bearish_conditions} of 3"
    )

    st.caption(
        "This is a simple rule-based summary, not a trained "
        "AI model or a recommendation to trade. It does not "
        "account for transaction costs, slippage, or your risk profile."
    )


# =========================================================
# 16. HISTORICAL DATA TABLE
# =========================================================

st.subheader("📋 Historical Data & Indicators")

display_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA20",
    "SMA50",
    "RSI",
    "MACD",
    "MACD_Signal",
    "Daily_Return",
    "Volatility20"
]

st.dataframe(
    data[display_columns].tail(30).round(2),
    use_container_width=True
)


# =========================================================
# 17. FOOTER
# =========================================================

st.divider()

st.caption(
    "StockSense AI | Built with Python, Streamlit, "
    "yfinance, Pandas, TA and Plotly."
)

st.caption(
    "For educational and research purposes only. "
    "Historical patterns do not guarantee future performance."
)
