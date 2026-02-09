import yfinance as yf
import pandas as pd

TICKER = "MSFT"

# Grab today's intraday data (5-minute bars are reliable on Yahoo)
df = yf.download(
    TICKER,
    period="1d",
    interval="5m",
    progress=False
)

if df.empty:
    print("No data returned. (Market may be closed or data unavailable.)")
    raise SystemExit(0)

# Fix MultiIndex columns if yfinance returns them
if hasattr(df.columns, "levels"):
    df.columns = df.columns.get_level_values(0)

# Convert to US/Eastern time
# yfinance usually returns timezone-aware UTC index for intraday
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
df.index = df.index.tz_convert("US/Eastern")

# Keep only regular market hours (9:30–16:00 ET)
df = df.between_time("09:30", "16:00")

if df.empty:
    print("No regular-hours data found for today (ET 09:30–16:00).")
    raise SystemExit(0)

# Build the target timestamps for "every hour" + the close
day = df.index[0].date()
tz = df.index.tz

targets = [
    pd.Timestamp(f"{day} 09:30", tz=tz),
    pd.Timestamp(f"{day} 10:30", tz=tz),
    pd.Timestamp(f"{day} 11:30", tz=tz),
    pd.Timestamp(f"{day} 12:30", tz=tz),
    pd.Timestamp(f"{day} 13:30", tz=tz),
    pd.Timestamp(f"{day} 14:30", tz=tz),
    pd.Timestamp(f"{day} 15:30", tz=tz),
    pd.Timestamp(f"{day} 16:00", tz=tz),  # official close time
]

print(f"Hourly prices for {TICKER} (today, US/Eastern):\n")

for t in targets:
    # Take the most recent close at or before the target time
    sub = df.loc[df.index <= t]
    if sub.empty:
        continue  # early day / no data yet
    price = sub["Close"].iloc[-1]
    print(f"{t.strftime('%Y-%m-%d %I:%M %p %Z')} -> ${price:.2f}")
