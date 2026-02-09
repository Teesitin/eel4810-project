import yfinance as yf

ticker = "AAPL"

data = yf.download(
    ticker,
    period="1d",
    interval="60m",
    progress=False
)

print(f"Hourly prices for {ticker} (today):\n")

for timestamp, row in data.iterrows():
    price = float(row["Close"])
    print(f"{timestamp} -> ${price:.2f}")
