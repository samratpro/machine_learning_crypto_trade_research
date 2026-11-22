"""
Download 3 years of Binance spot OHLCV candles from the public archive
(https://data.binance.vision) – no API key needed.

Output: data/<SYMBOL>_<interval>.parquet  (and .csv)
Usage:  env\\Scripts\\python.exe download_data.py
"""
import io
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVALS = ["5m", "15m", "1h", "4h", "1d"]
YEARS = 3
SAVE_CSV = True

BASE = "https://data.binance.vision/data/spot"
OUT_DIR = Path(__file__).parent / "data"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]

session = requests.Session()


def fetch_zip(url: str, retries: int = 6) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=(15, 120))
            if r.status_code == 404:
                return None
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"\n  retry {attempt + 1} in {wait}s ({type(e).__name__}) {url.rsplit('/', 1)[-1]}")
            time.sleep(wait)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, header=None)
    # Some files have a header row
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df.columns = COLS
    return df


def month_range(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) < (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def to_datetime(ts: pd.Series) -> pd.Series:
    ts = ts.astype("int64")
    # Binance switched spot files to microseconds from 2025-01-01
    us = ts > 10**14
    out = pd.Series(pd.NaT, index=ts.index, dtype="datetime64[ns, UTC]")
    out[us] = pd.to_datetime(ts[us], unit="us", utc=True)
    out[~us] = pd.to_datetime(ts[~us], unit="ms", utc=True)
    return out


def download(symbol: str, interval: str, start: date, end: date) -> pd.DataFrame:
    frames = []
    # Full months from the monthly archive
    for y, m in month_range(start, end):
        url = f"{BASE}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{y}-{m:02d}.zip"
        df = fetch_zip(url)
        if df is None:  # monthly file not published yet -> fall back to daily
            d = date(y, m, 1)
            while d.month == m and d < end:
                frames.append(fetch_zip(f"{BASE}/daily/klines/{symbol}/{interval}/"
                                        f"{symbol}-{interval}-{d}.zip"))
                d += timedelta(days=1)
        else:
            frames.append(df)
        print(f"  {symbol} {interval} {y}-{m:02d}", end="\r")
    # Current (partial) month from the daily archive
    d = date(end.year, end.month, 1)
    while d < end:
        frames.append(fetch_zip(f"{BASE}/daily/klines/{symbol}/{interval}/"
                                f"{symbol}-{interval}-{d}.zip"))
        d += timedelta(days=1)

    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    df["open_time"] = to_datetime(df["open_time"])
    df["close_time"] = to_datetime(df["close_time"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df = (df.drop(columns="ignore")
            .drop_duplicates("open_time")
            .sort_values("open_time"))
    start_ts = pd.Timestamp(start, tz="UTC")
    return df[df["open_time"] >= start_ts].reset_index(drop=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    end = datetime.now(timezone.utc).date()          # up to yesterday (complete days)
    start = end.replace(year=end.year - YEARS)
    print(f"Range: {start} -> {end} (UTC)")
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            path = OUT_DIR / f"{symbol}_{interval}.parquet"
            if path.exists() or path.with_suffix(".csv").exists():
                print(f"{symbol} {interval}: already downloaded, skipping")
                continue
            df = download(symbol, interval, start, end)
            df.to_parquet(path, index=False)
            if SAVE_CSV:
                df.to_csv(path.with_suffix(".csv"), index=False)
            print(f"{symbol} {interval}: {len(df):,} rows  "
                  f"{df.open_time.iloc[0]} -> {df.open_time.iloc[-1]}  -> {path.name}")


if __name__ == "__main__":
    main()
