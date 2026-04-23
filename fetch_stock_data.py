import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_API_SECRET in .env file.")

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Start from end of 2025 and walk backward
ANCHOR_END_ET = datetime(2025, 12, 31, 23, 59, 59, tzinfo=ET)

MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
ROWS_PER_FULL_DAY = 390

# Safety floor only
OLDEST_ALLOWED_ET = datetime(2010, 1, 1, 0, 0, 0, tzinfo=ET)


def parse_args() -> tuple[str, int]:
    if len(sys.argv) != 3:
        print("Usage: python fetchStockData.py <SYMBOL> <TARGET_ROWS>")
        print("Example: python fetchStockData.py MSFT 50000")
        sys.exit(1)

    symbol = sys.argv[1].upper()

    try:
        target_rows = int(sys.argv[2])
    except ValueError:
        print("TARGET_ROWS must be an integer.")
        sys.exit(1)

    if target_rows <= 0:
        print("TARGET_ROWS must be greater than 0.")
        sys.exit(1)

    return symbol, target_rows


def fetch_chunk(
    client: StockHistoricalDataClient,
    symbol: str,
    start_et: datetime,
    end_et: datetime,
) -> list[dict]:
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_et.astimezone(UTC),
        end=end_et.astimezone(UTC),
        feed=DataFeed.SIP,
    )

    barset = client.get_stock_bars(request)
    bars = barset.data.get(symbol, [])

    records = []
    for bar in bars:
        ts_et = bar.timestamp.astimezone(ET)

        records.append({
            "symbol": str(getattr(bar, "symbol", symbol)),
            "timestamp": ts_et,
            "open": None if getattr(bar, "open", None) is None else float(bar.open),
            "high": None if getattr(bar, "high", None) is None else float(bar.high),
            "low": None if getattr(bar, "low", None) is None else float(bar.low),
            "close": None if getattr(bar, "close", None) is None else float(bar.close),
            "volume": None if getattr(bar, "volume", None) is None else float(bar.volume),
            "trade_count": None if getattr(bar, "trade_count", None) is None else float(bar.trade_count),
            "vwap": None if getattr(bar, "vwap", None) is None else float(bar.vwap),
            "exchange": None if getattr(bar, "exchange", None) is None else str(getattr(bar, "exchange")),
        })

    return records


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[
            "symbol", "open", "high", "low", "close",
            "volume", "trade_count", "vwap", "exchange"
        ])

    df = pd.DataFrame(records)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def keep_regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df[df.index.dayofweek < 5].copy()

    mask = []
    for ts in df.index:
        keep = (
            (ts.hour > MARKET_OPEN_HOUR or (ts.hour == MARKET_OPEN_HOUR and ts.minute >= MARKET_OPEN_MINUTE))
            and (ts.hour < MARKET_CLOSE_HOUR)
        )
        mask.append(keep)

    return df.loc[mask].copy()


def drop_incomplete_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    date_counts = df.groupby(df.index.date).size()
    full_dates = set(date_counts[date_counts == ROWS_PER_FULL_DAY].index)

    mask = [ts.date() in full_dates for ts in df.index]
    return df.loc[mask].copy()


def build_output_path() -> Path:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "stock-data.json"


def export_to_json(df: pd.DataFrame, out_path: Path, symbol: str) -> None:
    records = []
    for ts, row in df.iterrows():
        records.append({
            "symbol": str(row["symbol"]),
            "timestamp": ts.isoformat(),
            "open": None if pd.isna(row["open"]) else float(row["open"]),
            "high": None if pd.isna(row["high"]) else float(row["high"]),
            "low": None if pd.isna(row["low"]) else float(row["low"]),
            "close": None if pd.isna(row["close"]) else float(row["close"]),
            "volume": None if pd.isna(row["volume"]) else float(row["volume"]),
            "trade_count": None if pd.isna(row["trade_count"]) else float(row["trade_count"]),
            "vwap": None if pd.isna(row["vwap"]) else float(row["vwap"]),
            "exchange": None if pd.isna(row["exchange"]) else str(row["exchange"]),
        })

    payload = {
        "symbol": symbol,
        "timezone": "America/New_York",
        "interval": "1m",
        "feed": "SIP",
        "rows": len(records),
        "fields": [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
            "exchange",
        ],
        "data": records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(records)} rows to {out_path}")


def main() -> None:
    symbol, target_rows = parse_args()

    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    all_frames = []
    current_end = ANCHOR_END_ET
    chunk_num = 1

    while current_end >= OLDEST_ALLOWED_ET:
        current_start = current_end - timedelta(days=6)
        current_start = current_start.replace(hour=0, minute=0, second=0, microsecond=0)

        print(
            f"[{chunk_num}] Fetching {symbol} "
            f"{current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}"
        )

        chunk_records = fetch_chunk(client, symbol, current_start, current_end)
        chunk_df = records_to_dataframe(chunk_records)

        if not chunk_df.empty:
            chunk_df = keep_regular_hours(chunk_df)
            all_frames.append(chunk_df)

            combined = pd.concat(all_frames).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            combined = drop_incomplete_days(combined)

            total_rows = len(combined)
            print(f"    rows so far: {total_rows}")

            if total_rows >= target_rows:
                combined = combined.tail(target_rows).copy()
                out_path = build_output_path()
                export_to_json(combined, out_path, symbol)
                print("Done.")
                return
        else:
            print("    no rows returned for this chunk")

        current_end = current_start - timedelta(seconds=1)
        chunk_num += 1

    if all_frames:
        combined = pd.concat(all_frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = drop_incomplete_days(combined)
        combined = combined.tail(target_rows).copy()
    else:
        combined = pd.DataFrame(columns=[
            "symbol", "open", "high", "low", "close",
            "volume", "trade_count", "vwap", "exchange"
        ])

    out_path = build_output_path()
    export_to_json(combined, out_path, symbol)
    print(f"Reached oldest allowed date before hitting {target_rows} rows.")


if __name__ == "__main__":
    main()