import yfinance as yf

ticker = yf.Ticker("TCS.NS")

data = ticker.history(period="1mo")

print(data)