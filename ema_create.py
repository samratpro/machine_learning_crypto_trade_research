"""
EMA features (docs/ema.md) - computed on 15m candles.

Two separate tests (docs/entry.md: never apply both at the same time):
  9 / 21 / 200   bullish: EMA9 > EMA21 > EMA200    bearish: EMA200 > EMA21 > EMA9
  13 / 48 / 200  bullish: EMA13 > EMA48 > EMA200   bearish: EMA200 > EMA48 > EMA13
No momentum: middle EMA (EMA21 for the 9 test, EMA48 for the 13 test) is straight, or not in the middle.

momentum_9_21_200 / momentum_13_48_200:  1 = bullish, -1 = bearish, 0 = no momentum

Call add_emas() on the raw 15m frame BEFORE data_merge.prepare_htf_data(),
the 15m values only reach a 5m row once their 15m candle has closed (no leakage).
"""
import numpy as np
import pandas as pd

EMA_PERIODS = [9, 13, 21, 48, 200]
EMA_SETS = {"9_21_200": (9, 21, 200), "13_48_200": (13, 48, 200)}

# "straight": the middle EMA moved less than FLAT_ATR x ATR(15m) over the last FLAT_BARS candles
FLAT_BARS = 8          # 8 x 15m = 2 hours
FLAT_ATR = 0.25

# fast EMA (EMA9 / EMA13) slope: change since the previous closed 15m candle, in ATR units.
# The entry needs it in the momentum direction (bull: > 0, bear: < 0) - docs/entry.md
FAST_SLOPE_BARS = 1


def _atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_emas(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    df = df.copy()
    for p in EMA_PERIODS:
        # min_periods=p -> warm-up rows stay NaN and are dropped in the merge
        df[f"ema{p}"] = df[price_col].ewm(span=p, adjust=False, min_periods=p).mean()

    atr = df["atr"] if "atr" in df else _atr(df)
    for name, (fast, mid, slow) in EMA_SETS.items():
        f, m, s = df[f"ema{fast}"], df[f"ema{mid}"], df[f"ema{slow}"]
        bull = (f > m) & (m > s)
        bear = (s > m) & (m > f)
        df[f"bull_{name}"] = bull.astype("int8")
        df[f"bear_{name}"] = bear.astype("int8")

        slope = (m - m.shift(FLAT_BARS)) / atr          # middle EMA slope in ATR units
        flat = slope.abs() < FLAT_ATR
        df[f"ema{mid}_slope"] = slope
        df[f"ema{mid}_flat"] = flat.astype("int8")
        df[f"ema{fast}_slope"] = (f - f.shift(FAST_SLOPE_BARS)) / atr   # fast EMA slope in ATR units

        mom = np.select([bull & ~flat, bear & ~flat], [1.0, -1.0], 0.0)
        df[f"momentum_{name}"] = np.where(slope.isna(), np.nan, mom)   # warm-up -> dropped later
    return df


def ema_columns() -> list[str]:
    cols = [f"ema{p}" for p in EMA_PERIODS]
    for name, (fast, mid, slow) in EMA_SETS.items():
        cols += [f"bull_{name}", f"bear_{name}", f"ema{mid}_slope", f"ema{mid}_flat", f"ema{fast}_slope", f"momentum_{name}"]
    return cols
