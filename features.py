"""
ML dataset for meta-labeling (docs/ml_train.md).

    from features import build_dataset, split_holdout, walk_forward, FEATURES
    ds = build_dataset('ALL', ema_set='13_48_200')          # one row per momentum entry candle
    dev, holdout = split_holdout(ds)                         # holdout = last 6 months, untouched until the end
    for train_idx, test_idx in walk_forward(dev):            # purged + embargoed, time-ordered
        model.fit(dev.loc[train_idx, FEATURES], dev.loc[train_idx, 'label'],
                  sample_weight=dev.loc[train_idx, 'weight'])

Primary model = the momentum entry (zone_create.entry_signal): it gives the SIDE.
The ML model only decides take / skip (meta-label): label = 1 if the trade ends with ROE > 0
under the ATR stop / trail / TP + time-limit exits (stop_loss.simulate_trades), else 0.

Feature rules (docs/guide.md, research_gap.md B6/A2-A6):
  - everything is known at the close of the signal candle (checked by check_no_lookahead())
  - no raw price levels: only ATR-scaled distances, ratios, z-scores (stationary across years / coins)
  - SIDE-RELATIVE: every directional value is multiplied by the side (+1 long, -1 short), so
    "positive" always means "in favour of the trade" -> one model for longs and shorts
  - continuous values, no hand-picked 0/1 thresholds
"""
import numpy as np
import pandas as pd

from analysis.window import ROOT, EMA_SETS
from stop_loss import (simulate_trades, STOP_LOSS, TRAIL_STEP, TAKE_PROFIT, TRAIL_GAP, MAX_BARS, LEVERAGE,
                       FEE_RATE)
from zone_create import FAST_SLOPE_MIN, entry_signal

HOLDOUT_START = pd.Timestamp('2026-05-20', tz='UTC')   # last 6 months: final test only, never tune on it
ZONE_TFS = ('15m', '1h', '4h', '1d')
RET_BARS = (1, 3, 12, 48, 288)          # past returns, 5m candles
Z_BARS = 288                            # 1 day of 5m candles for flow / volume z-scores
VOL_Z_BARS = 2016                       # 1 week for the volatility regime
NEAR_ATR = 2.0                          # zones counted "near" within this many 5m ATR (count features only)


def _z(x: pd.Series, n: int) -> pd.Series:
    """Rolling z-score over the last n candles (current candle included - it is closed)."""
    m = x.rolling(n, min_periods=n // 2).mean()
    s = x.rolling(n, min_periods=n // 2).std()
    return (x - m) / s.replace(0, np.nan)


def _run_age(sig: np.ndarray) -> np.ndarray:
    """Candles since the current entry-signal run started (1 = first candle of the momentum)."""
    new = np.r_[True, sig[1:] != sig[:-1]]
    rid = np.cumsum(new)
    return pd.Series(1, index=np.arange(len(sig))).groupby(rid).cumsum().to_numpy()


def side_free(df: pd.DataFrame, ema_set: str, slope_min: float) -> pd.DataFrame:
    """Per-candle values that do not depend on the side yet (signed ones are flipped in side_features)."""
    f, m, _ = EMA_SETS[ema_set]
    other = '9_21_200' if ema_set == '13_48_200' else '13_48_200'
    o, h, l, c = (df[f'{k}_5m'] for k in ('open', 'high', 'low', 'close'))
    a5, a15 = df['atr_5m'], df['atr_15m']
    sig = entry_signal(df, ema_set, slope_min)
    x = pd.DataFrame(index=df.index)
    x['time'], x['signal'] = df['open_time'], sig

    # ---- trend (15m EMAs), signed
    x['s_mid_slope'] = df[f'ema{m}_slope_15m']
    x['s_fast_slope'] = df[f'ema{f}_slope_15m']
    x['s_spread_fast_mid'] = (df[f'ema{f}_15m'] - df[f'ema{m}_15m']) / a15
    x['s_spread_mid_200'] = (df[f'ema{m}_15m'] - df['ema200_15m']) / a15
    x['s_dist_fast'] = (c - df[f'ema{f}_15m']) / a5
    x['s_dist_mid'] = (c - df[f'ema{m}_15m']) / a5
    x['s_dist_200'] = (c - df['ema200_15m']) / a5
    x['s_other_set_momentum'] = df[f'momentum_{other}_15m']
    x['touch_mid'] = df[f'touch_ema{m}_5m']
    x['momentum_age'] = np.where(sig != 0, _run_age(sig), 0)

    # ---- candle (5m)
    rng = (h - l).replace(0, np.nan)
    x['close_pos'] = (c - l) / rng                           # flipped for shorts in side_features
    x['s_body'] = (c - o) / a5
    x['range_atr'] = (h - l) / a5
    x['up_wick'] = (h - np.maximum(o, c)) / a5
    x['down_wick'] = (np.minimum(o, c) - l) / a5

    # ---- returns, signed, in 5m ATR
    for k in RET_BARS:
        x[f's_ret_{k}'] = (c - c.shift(k)) / a5

    # ---- order flow / volume
    for tf in ('5m', '15m', '1h'):
        vol = df[f'volume_{tf}'].replace(0, np.nan)
        x[f's_taker_{tf}'] = df[f'taker_buy_base_{tf}'] / vol - 0.5
    x['s_taker_z'] = _z(x['s_taker_5m'], Z_BARS)
    x['vol_z'] = _z(np.log(df['volume_5m'].replace(0, np.nan)), Z_BARS)
    x['trades_z'] = _z(np.log(df['trades_5m'].replace(0, np.nan)), Z_BARS)

    # ---- volatility regime
    x['atr_pct'] = a5 / c
    x['atr_1h_over_5m'] = df['atr_1h'] / a5
    x['atr_z_week'] = _z(np.log(a5), VOL_Z_BARS)

    # ---- time (UTC)
    hour = df['open_time'].dt.hour + df['open_time'].dt.minute / 60
    x['hour_sin'], x['hour_cos'] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    x['weekday'] = df['open_time'].dt.weekday

    # ---- zones: raw parts kept for side_features
    for tf in ZONE_TFS:
        x[f'_res_{tf}'] = (df[f'res_{tf}'] - c) / a5           # >0: resistance above price
        x[f'_sup_{tf}'] = (c - df[f'sup_{tf}']) / a5           # >0: support below price
        width = (df[f'res_{tf}'] - df[f'sup_{tf}']).replace(0, np.nan)
        x[f'_pos_{tf}'] = (c - df[f'sup_{tf}']) / width        # 0 = at support, 1 = at resistance
        for k in ('res_age', 'res_touches', 'sup_age', 'sup_touches'):
            x[f'_{k}_{tf}'] = df[f'{k}_{tf}']

    # ---- round numbers (Osler): nearest level above / below, step = 10^(digits-2) of the price
    step = 10.0 ** (np.floor(np.log10(c)) - 1)
    x['_round_up'] = (np.ceil(c / step) * step - c) / a5
    x['_round_dn'] = (c - np.floor(c / step) * step) / a5
    return x


def side_features(x: pd.DataFrame) -> pd.DataFrame:
    """Flip every signed value by the side; zones become 'ahead' (blocking the trade) / 'behind' (support)."""
    s = x['signal'].astype(float)
    long = s > 0
    out = pd.DataFrame({'time': x['time'], 'side': s}, index=x.index)
    for col in x.columns:
        if col.startswith('s_'):
            out[col[2:]] = x[col] * s
    for col in ('touch_mid', 'momentum_age', 'range_atr', 'vol_z', 'trades_z', 'atr_pct', 'atr_1h_over_5m',
                'atr_z_week', 'hour_sin', 'hour_cos', 'weekday'):
        out[col] = x[col]
    out['close_pos'] = np.where(long, x['close_pos'], 1 - x['close_pos'])        # 1 = closed at the trade's best end
    out['wick_with'] = np.where(long, x['down_wick'], x['up_wick'])               # rejection wick in favour
    out['wick_against'] = np.where(long, x['up_wick'], x['down_wick'])
    near_ahead = near_behind = 0
    for tf in ZONE_TFS:
        ahead = np.where(long, x[f'_res_{tf}'], x[f'_sup_{tf}'])                 # room until the opposing zone
        behind = np.where(long, x[f'_sup_{tf}'], x[f'_res_{tf}'])                # distance to the zone backing us
        out[f'zone_ahead_{tf}'], out[f'zone_behind_{tf}'] = ahead, behind
        out[f'range_pos_{tf}'] = np.where(long, x[f'_pos_{tf}'], 1 - x[f'_pos_{tf}'])
        a, b = ('res', 'sup')
        out[f'ahead_touches_{tf}'] = np.where(long, x[f'_{a}_touches_{tf}'], x[f'_{b}_touches_{tf}'])
        out[f'ahead_age_{tf}'] = np.where(long, x[f'_{a}_age_{tf}'], x[f'_{b}_age_{tf}'])
        out[f'behind_touches_{tf}'] = np.where(long, x[f'_{b}_touches_{tf}'], x[f'_{a}_touches_{tf}'])
        out[f'behind_age_{tf}'] = np.where(long, x[f'_{b}_age_{tf}'], x[f'_{a}_age_{tf}'])
        near_ahead = near_ahead + ((ahead >= -NEAR_ATR) & (ahead <= NEAR_ATR))
        near_behind = near_behind + ((behind >= -NEAR_ATR) & (behind <= NEAR_ATR))
    out['zones_near_ahead'], out['zones_near_behind'] = near_ahead, near_behind  # direction-aware confluence
    out['round_ahead'] = np.where(long, x['_round_up'], x['_round_dn'])
    out['round_behind'] = np.where(long, x['_round_dn'], x['_round_up'])
    return out


FEATURES = None   # filled on first build (list of model input columns)
_META = ['time', 'symbol', 'side', 'label', 'roe', 'exit_reason', 'bars_held', 'entry_time', 'exit_time',
         'signal_idx', 'exit_idx', 'weight']


def _uniqueness(entry_idx: np.ndarray, exit_idx: np.ndarray, n: int) -> np.ndarray:
    """Average uniqueness of each trade (Lopez de Prado): mean over its candles of 1 / (trades open on it)."""
    conc = np.zeros(n + 1)
    np.add.at(conc, entry_idx, 1)
    np.add.at(conc, exit_idx + 1, -1)
    conc = np.cumsum(conc)[:n]
    inv = np.r_[0, np.cumsum(1 / np.maximum(conc, 1))]
    return (inv[exit_idx + 1] - inv[entry_idx]) / (exit_idx - entry_idx + 1)


def _load(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / 'data' / f'{symbol}_master.parquet')


def build_symbol(symbol: str, ema_set: str = '13_48_200', slope_min: float = FAST_SLOPE_MIN,
                 stop_loss=STOP_LOSS, trail_step=TRAIL_STEP, take_profit=TAKE_PROFIT, trail_gap=TRAIL_GAP,
                 max_bars=MAX_BARS, leverage=LEVERAGE, fee_rate=FEE_RATE, btc: pd.DataFrame | None = None) -> pd.DataFrame:
    df = _load(symbol)
    x = side_free(df, ema_set, slope_min)
    if btc is not None:                       # BTC as a lead for the alts (BTC itself: its own values)
        b = btc.set_index('open_time')
        bx = pd.DataFrame({'s_btc_ret_12': (b['close_5m'] - b['close_5m'].shift(12)) / b['atr_5m'],
                           's_btc_ret_48': (b['close_5m'] - b['close_5m'].shift(48)) / b['atr_5m'],
                           's_btc_momentum': b[f'momentum_{ema_set}_15m']})
        x = x.join(bx, on='time')
    feats = side_features(x)
    df['signal'] = x['signal']
    tr = simulate_trades(df, 'signal', stop_loss, trail_step, take_profit, leverage, trail_gap,
                         fee_rate=fee_rate, overlap=True, max_bars=max_bars)
    tr = tr[tr['exit_reason'] != 'open_eod']                     # outcome unknown at the end of the data
    rows = feats.loc[tr['signal_idx'].to_numpy()].reset_index(drop=True)
    rows.insert(1, 'symbol', symbol)
    rows['label'] = (tr['roe'].to_numpy() > 0).astype('int8')
    for k in ('roe', 'exit_reason', 'bars_held', 'entry_time', 'exit_time', 'signal_idx', 'exit_idx'):
        rows[k] = tr[k].to_numpy()
    rows['weight'] = _uniqueness(tr['entry_idx'].to_numpy(), tr['exit_idx'].to_numpy(), len(df))
    return rows


def build_dataset(symbols='ALL', ema_set: str = '13_48_200', slope_min: float = FAST_SLOPE_MIN,
                  verbose: bool = True, **exits) -> pd.DataFrame:
    """One row per momentum entry candle, all symbols. exits: stop_loss / trail_step / take_profit / trail_gap /
    max_bars / leverage / fee_rate (defaults from stop_loss.py)."""
    global FEATURES
    from analysis.sliding import all_symbols
    if isinstance(symbols, str):
        symbols = all_symbols() if symbols.upper() == 'ALL' else [symbols]
    btc = _load('BTCUSDT')[['open_time', 'close_5m', 'atr_5m', f'momentum_{ema_set}_15m']]
    parts = []
    for s in symbols:
        p = build_symbol(s, ema_set, slope_min, btc=btc, **exits)
        if verbose:
            print(f'{s}: {len(p):,} entries | win rate {p["label"].mean():.3f}')
        parts.append(p)
    ds = pd.concat(parts, ignore_index=True).sort_values(['time', 'symbol'], kind='stable').reset_index(drop=True)
    FEATURES = [c for c in ds.columns if c not in _META]
    return ds


def split_holdout(ds: pd.DataFrame, start=HOLDOUT_START):
    """dev = before the holdout (trades that END before it, so no label crosses), holdout = entries from start."""
    start = pd.Timestamp(start)
    dev = ds[(ds['time'] < start) & (pd.to_datetime(ds['exit_time'], utc=True) < start)]
    hold = ds[ds['time'] >= start]
    return dev.reset_index(drop=True), hold.reset_index(drop=True)


def walk_forward(dev: pd.DataFrame, n_splits: int = 5, embargo=pd.Timedelta(hours=24), min_train_frac: float = 0.3):
    """Expanding-window, time-ordered folds. Each test block follows its training data in time.
    Purge: drop training trades whose exit is at or after (test start - embargo), so no training label
    overlaps the test period (embargo >= MAX_BARS time limit). Yields (train_index, test_index)."""
    t = pd.to_datetime(dev['time'], utc=True)
    ex = pd.to_datetime(dev['exit_time'], utc=True)
    t0, t1 = t.min(), t.max()
    first = t0 + (t1 - t0) * min_train_frac
    edges = pd.date_range(first, t1, periods=n_splits + 1)
    for k in range(n_splits):
        a, b = edges[k], edges[k + 1]
        test = (t >= a) & ((t < b) if k < n_splits - 1 else (t <= b))
        train = (t < a - embargo) & (ex < a - embargo)
        yield dev.index[train], dev.index[test]


def check_no_lookahead(symbol: str = 'BTCUSDT', ema_set: str = '13_48_200', cut_rows=(150_000, 250_000)) -> bool:
    """Features computed on data cut at row k must equal the full-data features up to k
    (any difference = a feature used a later candle)."""
    df = _load(symbol)
    full = side_free(df, ema_set, FAST_SLOPE_MIN)
    ok = True
    for k in cut_rows:
        part = side_free(df.iloc[:k].copy(), ema_set, FAST_SLOPE_MIN)
        a = full.iloc[:k].drop(columns=['time']).to_numpy('float64')
        b = part.drop(columns=['time']).to_numpy('float64')
        same = np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
        bad = ~same.all(axis=0)
        if bad.any():
            ok = False
            print(f'look-ahead at cut {k}:', list(part.drop(columns=['time']).columns[bad]))
    print('features look-ahead check:', 'passed' if ok else 'FAILED')
    return ok
