
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import streamlit as st
import yfinance as yf


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="StockSense AI | Paper Trading",
    page_icon="📈",
    layout="wide",
)

DB_PATH = Path(__file__).resolve().parent / "paper_trading.db"
STARTING_CASH = 100_000.0


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def connect_db():
    return sqlite3.connect(DB_PATH, timeout=10)


def initialise_database():
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                total REAL NOT NULL
            )
        """)

        conn.execute(
            "INSERT OR IGNORE INTO account (id, cash) VALUES (1, ?)",
            (STARTING_CASH,),
        )


def get_cash():
    with connect_db() as conn:
        row = conn.execute(
            "SELECT cash FROM account WHERE id = 1"
        ).fetchone()

    return float(row[0])


def get_trades():
    with connect_db() as conn:
        return pd.read_sql_query(
            "SELECT * FROM trades ORDER BY id ASC",
            conn,
        )


# --------------------------------------------------
# FIFO ACCOUNTING
# --------------------------------------------------

def calculate_accounting(trades):
    """
    Rebuild holdings and realized P&L from the transaction ledger.

    FIFO means the earliest purchased shares are matched first
    against subsequent sell orders.
    """

    lots_by_symbol = defaultdict(deque)
    realized_pnl = 0.0
    completed_sales = []

    if trades.empty:
        return pd.DataFrame(), 0.0, pd.DataFrame()

    for _, trade in trades.sort_values("id").iterrows():
        symbol = str(trade["symbol"])
        side = str(trade["side"]).upper()
        quantity = int(trade["quantity"])
        price = float(trade["price"])

        if side == "BUY":
            lots_by_symbol[symbol].append({
                "quantity": quantity,
                "price": price,
            })

        elif side == "SELL":
            remaining_to_sell = quantity
            sale_cost_basis = 0.0

            while remaining_to_sell > 0 and lots_by_symbol[symbol]:
                lot = lots_by_symbol[symbol][0]

                matched_quantity = min(
                    remaining_to_sell,
                    lot["quantity"],
                )

                sale_cost_basis += matched_quantity * lot["price"]

                lot["quantity"] -= matched_quantity
                remaining_to_sell -= matched_quantity

                if lot["quantity"] == 0:
                    lots_by_symbol[symbol].popleft()

            # Sales beyond available lots indicate a ledger problem.
            if remaining_to_sell > 0:
                raise ValueError(
                    f"Transaction history is inconsistent for {symbol}. "
                    "A sell order exceeds recorded purchases."
                )

            sale_proceeds = quantity * price
            sale_pnl = sale_proceeds - sale_cost_basis
            realized_pnl += sale_pnl

            completed_sales.append({
                "Trade ID": int(trade["id"]),
                "Symbol": symbol,
                "Quantity Sold": quantity,
                "Sale Price": price,
                "Cost Basis": sale_cost_basis,
                "Realized P&L": sale_pnl,
            })

    holding_rows = []

    for symbol, lots in lots_by_symbol.items():
        quantity = sum(lot["quantity"] for lot in lots)

        if quantity <= 0:
            continue

        remaining_cost = sum(
            lot["quantity"] * lot["price"]
            for lot in lots
        )

        holding_rows.append({
            "Symbol": symbol,
            "Quantity": quantity,
            "Average Buy Price": remaining_cost / quantity,
            "Remaining Cost Basis": remaining_cost,
        })

    holdings = pd.DataFrame(holding_rows)

    sales = pd.DataFrame(completed_sales)

    return holdings, realized_pnl, sales


# --------------------------------------------------
# MARKET DATA
# --------------------------------------------------

@st.cache_data(ttl=60)
def get_current_price(symbol):
    history = yf.Ticker(symbol).history(
        period="5d",
        interval="1d",
        auto_adjust=True,
    )

    if history.empty or history["Close"].dropna().empty:
        raise ValueError(
            f"No recent market price found for {symbol}."
        )

    return float(history["Close"].dropna().iloc[-1])


# --------------------------------------------------
# VIRTUAL ORDER EXECUTION
# --------------------------------------------------

def execute_trade(symbol, side, quantity, price):
    if quantity <= 0 or price <= 0:
        raise ValueError("Quantity and price must be positive.")

    total = quantity * price

    with connect_db() as conn:
        conn.execute("BEGIN IMMEDIATE")

        cash_row = conn.execute(
            "SELECT cash FROM account WHERE id = 1"
        ).fetchone()

        cash = float(cash_row[0])

        position_row = conn.execute("""
            SELECT COALESCE(
                SUM(
                    CASE
                        WHEN side = 'BUY' THEN quantity
                        ELSE -quantity
                    END
                ), 0
            )
            FROM trades
            WHERE symbol = ?
        """, (symbol,)).fetchone()

        owned_quantity = int(position_row[0])

        if side == "BUY":
            if total > cash:
                raise ValueError(
                    f"Insufficient virtual cash. "
                    f"Available: ₹{cash:,.2f}"
                )

            new_cash = cash - total

        elif side == "SELL":
            if quantity > owned_quantity:
                raise ValueError(
                    f"You own {owned_quantity} shares of {symbol}; "
                    f"you cannot sell {quantity}."
                )

            new_cash = cash + total

        else:
            raise ValueError("Order must be BUY or SELL.")

        conn.execute(
            "UPDATE account SET cash = ? WHERE id = 1",
            (new_cash,),
        )

        conn.execute("""
            INSERT INTO trades (symbol, side, quantity, price, total)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, side, quantity, price, total))


# --------------------------------------------------
# APP
# --------------------------------------------------

initialise_database()

st.title("📈 StockSense AI")
st.header("Virtual Paper Trading")
st.write(
    "Practice buying and selling stocks with virtual money. "
    "No real brokerage orders are placed."
)

st.info(
    "Starting capital: ₹1,00,000. Market prices may be delayed."
)

trades = get_trades()

try:
    holdings, realized_pnl, sales = calculate_accounting(trades)
except ValueError as error:
    st.error(str(error))
    st.stop()


# --------------------------------------------------
# PRICE LOOKUP AND PORTFOLIO VALUATION
# --------------------------------------------------

prices = {}
price_errors = {}

for symbol in holdings["Symbol"].tolist() if not holdings.empty else []:
    try:
        prices[symbol] = get_current_price(symbol)
    except Exception as error:
        price_errors[symbol] = str(error)

if not holdings.empty:
    holdings["Current Price"] = holdings["Symbol"].map(prices)
    holdings["Market Value"] = (
        holdings["Quantity"] * holdings["Current Price"]
    )
    holdings["Unrealized P&L"] = (
        holdings["Market Value"]
        - holdings["Remaining Cost Basis"]
    )
else:
    holdings = pd.DataFrame(columns=[
        "Symbol",
        "Quantity",
        "Average Buy Price",
        "Remaining Cost Basis",
        "Current Price",
        "Market Value",
        "Unrealized P&L",
    ])

cash = get_cash()

# Do not present a partial valuation as a complete portfolio value.
all_prices_available = not price_errors

if all_prices_available:
    market_value = (
        float(holdings["Market Value"].sum())
        if not holdings.empty else 0.0
    )
    unrealized_pnl = (
        float(holdings["Unrealized P&L"].sum())
        if not holdings.empty else 0.0
    )
    portfolio_value = cash + market_value
    total_pnl = realized_pnl + unrealized_pnl
else:
    market_value = None
    unrealized_pnl = None
    portfolio_value = None
    total_pnl = None

st.subheader("Portfolio Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Available Cash", f"₹{cash:,.2f}")

c2.metric(
    "Market Value",
    f"₹{market_value:,.2f}"
    if market_value is not None else "Unavailable",
)

c3.metric(
    "Portfolio Value",
    f"₹{portfolio_value:,.2f}"
    if portfolio_value is not None else "Unavailable",
)

c4, c5, c6 = st.columns(3)

c4.metric("Realized P&L", f"₹{realized_pnl:,.2f}")

c5.metric(
    "Unrealized P&L",
    f"₹{unrealized_pnl:,.2f}"
    if unrealized_pnl is not None else "Unavailable",
)

c6.metric(
    "Total P&L",
    f"₹{total_pnl:,.2f}"
    if total_pnl is not None else "Unavailable",
)

if price_errors:
    st.warning(
        "Some market prices could not be retrieved. "
        "Complete portfolio value and unrealized P&L are hidden "
        "until all holding prices are available. A failed price "
        "lookup does not mean a holding is worth zero."
    )


# --------------------------------------------------
# ORDER FORM
# --------------------------------------------------

st.subheader("Place a Virtual Order")

with st.form("trade_form"):
    symbol_input = st.text_input(
        "Stock ticker",
        value="TCS.NS",
        help="Examples: TCS.NS, RELIANCE.NS, INFY.NS",
    ).strip().upper()

    side_input = st.radio(
        "Order type",
        ["BUY", "SELL"],
        horizontal=True,
    )

    quantity_input = st.number_input(
        "Quantity (shares)",
        min_value=1,
        max_value=1_000_000,
        value=1,
        step=1,
    )

    submitted = st.form_submit_button(
        "Get Price and Review Order",
        type="primary",
    )

if submitted:
    if not symbol_input or any(ch.isspace() for ch in symbol_input):
        st.error("Enter a valid ticker.")
    else:
        try:
            with st.spinner("Fetching the latest available price..."):
                reference_price = get_current_price(symbol_input)

            st.session_state["pending_trade"] = {
                "symbol": symbol_input,
                "side": side_input,
                "quantity": int(quantity_input),
                "price": reference_price,
            }

        except Exception as error:
            st.error(f"Unable to retrieve a price: {error}")

pending = st.session_state.get("pending_trade")

if pending:
    st.warning(
        f"{pending['side']} {pending['quantity']} share(s) of "
        f"{pending['symbol']} at reference price "
        f"₹{pending['price']:,.2f}. Estimated order value: "
        f"₹{pending['price'] * pending['quantity']:,.2f}."
    )

    confirm_col, cancel_col = st.columns(2)

    with confirm_col:
        if st.button("Confirm Virtual Trade", type="primary"):
            try:
                execute_trade(
                    pending["symbol"],
                    pending["side"],
                    pending["quantity"],
                    pending["price"],
                )

                del st.session_state["pending_trade"]
                get_current_price.clear()

                st.success("Virtual trade recorded.")
                st.rerun()

            except Exception as error:
                st.error(str(error))

    with cancel_col:
        if st.button("Cancel Order"):
            del st.session_state["pending_trade"]
            st.rerun()


# --------------------------------------------------
# HOLDINGS TABLE
# --------------------------------------------------

st.subheader("Your Holdings")

if holdings.empty:
    st.write("No shares currently held.")
else:
    display_columns = [
        "Symbol",
        "Quantity",
        "Average Buy Price",
        "Remaining Cost Basis",
        "Current Price",
        "Market Value",
        "Unrealized P&L",
    ]

    st.dataframe(
        holdings[display_columns].round(2),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------
# REALIZED PROFIT AND LOSS
# --------------------------------------------------

st.subheader("Realized P&L from Sales")

if sales.empty:
    st.write(
        "No completed virtual sales yet. Realized P&L will appear "
        "after you sell shares."
    )
else:
    st.dataframe(
        sales.round(2),
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------
# TRANSACTION HISTORY
# --------------------------------------------------

st.subheader("Transaction History")

if trades.empty:
    st.write("No transactions recorded yet.")
else:
    st.dataframe(
        trades.sort_values("id", ascending=False).round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download Transaction History",
        data=trades.to_csv(index=False).encode("utf-8"),
        file_name="stocksense_paper_trades.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# NOTES
# ---------------------------------------------------ljh

st.caption(
    "Educational simulator only. FIFO accounting is used for "
    "realized P&L and remaining cost basis. Brokerage fees, taxes, "
    "slippage, dividends and corporate-action adjustments are not "
    "fully modelled. Market data can be delayed. Transactions are "
    "stored locally in paper_trading.db."
)
