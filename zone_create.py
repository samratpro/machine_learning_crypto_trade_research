"""
Zone features + 5m entry trigger (docs/zone.md).

Step 1 (per timeframe, before shift/merge):  add_zones(df_tf, tf)
    resistance = rolling N-bar High,  band = resistance +/- 0.25 ATR
    support    = rolling N-bar Low,   band = support    +/- 0.25 ATR
Step 2 (on the merged 5m master frame):      add_zone_features(df)
    ATR-normalised distances + Confluence Score (0-4)
Step 3 (on the merged 5m master frame):      add_entry_signal(df)
    3-step long entry confirmation
"""
import numpy as np
import pandas as pd

ZONE_LOOKBACK = {"15m": 20, "1h": 24, "4h": 18, "1d": 14}
ZONE_TFS = list(ZONE_LOOKBACK)
ATR_PERIOD = 14
BAND_ATR = 0.25          # band half-width, in that timeframe's ATR
NEAR_ATR_5M = 0.5        # "near" tolerance, in 5m ATR
TOP_CLOSE_PCT = 0.35     # close in top 35% of the bar
TAKER_BUY_RATIO = 0.52
VOL_AVG_BARS = 20
FAST_SLOPE_MIN = 0.05    # momentum needs the fast EMA (EMA9/13) to move at least this much per 15m candle,
                         # in 15m ATR, in the trend direction (must be > 0)


def atr(df: pd.DataFrame, period: int = ATR_PERIOD,
        high="high", low="low", close="close") -> pd.Series:
    prev_close = df[close].shift(1)
    tr = pd.concat([df[high] - df[low],
                    (df[high] - prev_close).abs(),
                    (df[low] - prev_close).abs()], axis=1).max(axis=1)
    # Wilder smoothing
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------- step 1
def add_zones(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Raw HTF frame (open/high/low/close). Adds atr, res*, sup* columns."""
    n = ZONE_LOOKBACK[tf]
    df = df.copy()
    df["atr"] = atr(df)
    df["res"] = df["high"].rolling(n, min_periods=n).max()
    df["sup"] = df["low"].rolling(n, min_periods=n).min()
    w = BAND_ATR * df["atr"]
    df["res_hi"], df["res_lo"] = df["res"] + w, df["res"] - w
    df["sup_hi"], df["sup_lo"] = df["sup"] + w, df["sup"] - w

    # zone strength (Chung & Bellotti: more touches -> stronger, older -> weaker), over the same N bars
    hi, lo = df["high"].to_numpy("float64"), df["low"].to_numpy("float64")
    for name, arr, best, inside in (("res", hi, np.argmax, lambda v, b: v >= b),
                                    ("sup", lo, np.argmin, lambda v, b: v <= b)):
        age = np.full(len(df), np.nan)
        touches = np.full(len(df), np.nan)
        if len(df) >= n:
            win = np.lib.stride_tricks.sliding_window_view(arr, n)            # rows n-1 .. end
            age[n - 1:] = n - 1 - best(win, axis=1)                            # bars since the level was set
            band = (df["res_lo"] if name == "res" else df["sup_hi"]).to_numpy("float64")[n - 1:]
            touches[n - 1:] = inside(win, band[:, None]).sum(axis=1)           # bars that reached the band
        df[f"{name}_age"], df[f"{name}_touches"] = age, touches
    return df


def zone_columns() -> list[str]:
    return ["atr", "res", "res_hi", "res_lo", "sup", "sup_hi", "sup_lo",
            "res_age", "res_touches", "sup_age", "sup_touches"]


# ---------------------------------------------------------------- step 2
def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merged 5m frame (HTF columns = last CLOSED HTF candle)."""
    df = df.copy()
    df["atr_5m"] = atr(df, high="high_5m", low="low_5m", close="close_5m")
    c, a5 = df["close_5m"], df["atr_5m"]
    tol = NEAR_ATR_5M * a5

    confluence = np.zeros(len(df), dtype="int8")
    for tf in ZONE_TFS:
        df[f"dist_res_{tf}"] = (c - df[f"res_{tf}"]) / a5
        df[f"dist_sup_{tf}"] = (c - df[f"sup_{tf}"]) / a5
        near_res = (c >= df[f"res_lo_{tf}"] - tol) & (c <= df[f"res_hi_{tf}"] + tol)
        near_sup = (c >= df[f"sup_lo_{tf}"] - tol) & (c <= df[f"sup_hi_{tf}"] + tol)
        df[f"near_zone_{tf}"] = (near_res | near_sup).astype("int8")
        confluence += df[f"near_zone_{tf}"].to_numpy()
    df["confluence"] = confluence
    df["dist_ema21_15m"] = (c - df["ema21_15m"]) / a5
    return df


# ---------------------------------------------------------------- step 3
def entry_signal(df: pd.DataFrame, name: str, slope_min: float = FAST_SLOPE_MIN) -> np.ndarray:
    """ENTRY = momentum only, for one EMA test ('9_21_200' / '13_48_200'):
    bullish momentum -> 1 (long), bearish -> -1 (short), no momentum -> 0 (no entry).
    Momentum = EMA order (fast > mid > 200 / 200 > mid > fast) + middle EMA not flat
             + fast EMA slope >= slope_min ATR per 15m candle in the trend direction
               (long: slope >= +slope_min, short: slope <= -slope_min). slope_min must be > 0.
    Nothing else filters the entry; distances, touches and zones are ML features only."""
    if not slope_min > 0:
        raise ValueError(f"SLOPE_MIN must be > 0 (got {slope_min}): a flat or opposite fast EMA is no momentum")
    fast = {"9_21_200": 9, "13_48_200": 13}[name]
    m = df[f"momentum_{name}_15m"].fillna(0).to_numpy()
    slope = df[f"ema{fast}_slope_15m"].to_numpy()
    return np.where(m * slope >= slope_min, m, 0).astype("int8")


def add_entry_signal(df: pd.DataFrame) -> pd.DataFrame:
    """ENTRY (docs/entry.md) - the only entry rule, one column per EMA test (never both at once):
      entry_9_21_200   uses momentum_9_21_200  + distance to EMA9
      entry_13_48_200  uses momentum_13_48_200 + distance to EMA13
      On each completed 5m candle: momentum BULL -> long (1), BEAR -> short (-1), NONE -> 0.
      Enter at the next candle's open. No other entry filter.
      dist_ema*_5m (close - fast EMA, ATR) and touch_ema*_5m (candle touches the middle EMA)
      are still computed, as ML features only.

    The zone.md 3-step confirmation is still computed, but only as info / ML features
    (zone_long, zone_short) - it does NOT filter the entry:
      long : low into 15m/1h support or EMA21,   close top 35% or > prev high,  taker buy > 0.52
      short: high into 15m/1h resistance or EMA21, close bottom 35% or < prev low, taker buy < 0.48
      both : volume above the 20-candle average
    """
    df = df.copy()
    h, l, c = (df[f"{k}_5m"] for k in ("high", "low", "close"))
    tol = NEAR_ATR_5M * df["atr_5m"]
    ema21 = df["ema21_15m"]
    m9, m13 = df["momentum_9_21_200_15m"], df["momentum_13_48_200_15m"]
    mom = np.select([(m9 == 1) | (m13 == 1), (m9 == -1) | (m13 == -1)], [1, -1], 0)   # info only

    rng = (h - l).replace(0, np.nan)
    close_pos = (c - l) / rng                       # 0 = closed at low, 1 = closed at high
    vol = df["volume_5m"]
    buy_ratio = df["taker_buy_base_5m"] / vol.replace(0, np.nan)
    vol_avg = vol.rolling(VOL_AVG_BARS, min_periods=VOL_AVG_BARS).mean().shift(1)
    df["taker_buy_ratio_5m"] = buy_ratio
    df["close_pos_5m"] = close_pos
    above_avg_vol = vol > vol_avg

    # ---- long
    p15, p1h, pema = l <= df["sup_hi_15m"], l <= df["sup_hi_1h"], l <= ema21 + tol
    held = c > np.minimum.reduce([np.where(p15, df["sup_lo_15m"], np.inf),
                                  np.where(p1h, df["sup_lo_1h"], np.inf),
                                  np.where(pema, ema21 - tol, np.inf)])
    df["long_pullback"] = (p15 | p1h | pema).astype("int8")
    df["long_rejection"] = (((close_pos >= 1 - TOP_CLOSE_PCT) | (c > h.shift(1))) & held).astype("int8")
    df["long_orderflow"] = ((buy_ratio > TAKER_BUY_RATIO) & above_avg_vol).astype("int8")
    df["zone_long"] = ((mom == 1) & (df["long_pullback"] == 1) & (df["long_rejection"] == 1)
                       & (df["long_orderflow"] == 1)).astype("int8")

    # ---- short (mirror)
    p15, p1h, pema = h >= df["res_lo_15m"], h >= df["res_lo_1h"], h >= ema21 - tol
    held = c < np.maximum.reduce([np.where(p15, df["res_hi_15m"], -np.inf),
                                  np.where(p1h, df["res_hi_1h"], -np.inf),
                                  np.where(pema, ema21 + tol, -np.inf)])
    df["short_pullback"] = (p15 | p1h | pema).astype("int8")
    df["short_rejection"] = (((close_pos <= TOP_CLOSE_PCT) | (c < l.shift(1))) & held).astype("int8")
    df["short_orderflow"] = ((buy_ratio < 1 - TAKER_BUY_RATIO) & above_avg_vol).astype("int8")
    df["zone_short"] = ((mom == -1) & (df["short_pullback"] == 1) & (df["short_rejection"] == 1)
                        & (df["short_orderflow"] == 1)).astype("int8")

    # ML features (not entry filters): candle touches the middle EMA, close distance to the fast EMA
    for p in (21, 48):
        ema = df[f"ema{p}_15m"]
        df[f"touch_ema{p}_5m"] = ((l <= ema) & (h >= ema)).astype("int8")
    for name, fast in (("9_21_200", 9), ("13_48_200", 13)):
        df[f"dist_ema{fast}_5m"] = (c - df[f"ema{fast}_15m"]) / df["atr_5m"]
        df[f"entry_{name}"] = entry_signal(df, name)    # momentum only
    return df
