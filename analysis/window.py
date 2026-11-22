"""
Data + logic for one candle window (no plotting here).

Entry rule (docs/entry.md), one EMA test at a time (ema_set = '9_21_200' or '13_48_200'):
    on each completed 5m candle: momentum BULL -> BUY, BEAR -> SELL, NONE -> no entry,
    and no entry if close is on the wrong side of the fast EMA (negative ATR distance)
Momentum (docs/ema.md, 15m): fast > mid > 200 = BULL, 200 > mid > fast = BEAR, mid EMA flat = NONE

    win = build_window('BTCUSDT', '2024-12-05 00:55', candles=30, leverage=10, overlap=True, ema_set='9_21_200')
    win = build_window('BTCUSDT', '2024-12-05 00:55', candles=(30, 60))   # candles 30..59 after START
    win.w        -> the window candles (5m master rows)
    win.table    -> per-candle rule check
    win.trades   -> trades whose entry signal is inside the window (stop-loss rules applied)
    win.skipped  -> signals skipped because a trade was open (only when overlap=False)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stop_loss import simulate_trades, STOP_LOSS, TRAIL_STEP, TAKE_PROFIT, TRAIL_GAP
from zone_create import VOL_AVG_BARS, FAST_SLOPE_MIN, entry_signal

ROOT = Path(__file__).resolve().parent.parent
_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def load(symbol: str) -> pd.DataFrame:
    """Cached read; re-reads automatically when the parquet file is rebuilt (data_merge.py)."""
    path = ROOT / 'data' / f'{symbol}_master.parquet'
    mtime = path.stat().st_mtime
    if symbol not in _cache or _cache[symbol][0] != mtime:
        _cache[symbol] = (mtime, pd.read_parquet(path))
    return _cache[symbol][1]


@dataclass
class Window:
    symbol: str
    df: pd.DataFrame          # full master data
    start: int                # row of the first window candle in df
    candles: int
    leverage: float
    overlap: bool
    ema_set: str              # '9_21_200' or '13_48_200'
    cols: dict                # rule-table column names for this EMA set
    w: pd.DataFrame           # window candles
    vol_avg: pd.Series        # 20-candle average volume (previous candles)
    table: pd.DataFrame       # per-candle rule check
    trades: pd.DataFrame      # trades started in the window
    skipped: pd.DataFrame     # signals skipped while a trade was open

    @property
    def end(self) -> int:
        return self.start + self.candles - 1


EMA_SETS = {'9_21_200': (9, 21, 200), '13_48_200': (13, 48, 200)}


def rule_columns(ema_set: str) -> dict:
    f, m, sl = EMA_SETS[ema_set]
    return {'bull': f'EMA {f}>{m}>{sl}', 'bear': f'EMA {sl}>{m}>{f}', 'slope': f'EMA{m} slope',
            'flat': f'EMA{m} not flat', 'fslope': f'EMA{f} slope', 'fslope_ok': f'EMA{f} slope ok',
            'dist': f'close-EMA{f} (ATR)', 'dist_ok': 'distance ok', 'touch': f'Not touch EMA{m}'}


def rule_table(w: pd.DataFrame, ema_set: str, slope_min: float = FAST_SLOPE_MIN) -> pd.DataFrame:
    f, m, _ = EMA_SETS[ema_set]
    c = rule_columns(ema_set)
    mom = w['momentum']
    dist = w[f'dist_ema{f}_5m']
    fslope = w[f'ema{f}_slope_15m']
    return pd.DataFrame({
        'time':       w['open_time'].dt.strftime('%m-%d %H:%M'),
        'open': w['open_5m'], 'high': w['high_5m'], 'low': w['low_5m'], 'close': w['close_5m'],
        c['bull']:    w[f'bull_{ema_set}_15m'] == 1,
        c['bear']:    w[f'bear_{ema_set}_15m'] == 1,
        c['slope']:   w[f'ema{m}_slope_15m'].round(2),
        c['flat']:    w[f'ema{m}_flat_15m'] == 0,
        'MOMENTUM':   mom.map({1: 'BULL', -1: 'BEAR', 0: 'NONE'}),
        c['fslope']:  fslope.round(2),
        c['fslope_ok']: (mom != 0) & (mom * fslope > slope_min),
        c['dist']:    dist.round(2),
        c['dist_ok']: (mom != 0) & (mom * dist >= 0),
        c['touch']:   w[f'touch_ema{m}_5m'] == 0,
        'ENTRY':      w['entry'].map({1: 'BUY', -1: 'SELL', 0: '-'}),
    }).reset_index(drop=True)


def build_window(symbol: str, start_time, candles: int = 30, leverage: float = 10,
                 overlap: bool = True, ema_set: str = '9_21_200', stop_loss: float = STOP_LOSS,
                 trail_step: float = TRAIL_STEP, take_profit: float = TAKE_PROFIT,
                 trail_gap=TRAIL_GAP, slope_min: float = FAST_SLOPE_MIN) -> Window:
    """candles = 30        -> the 30 candles starting at start_time
       candles = (30, 60)  -> candle 30 up to (not incl.) candle 60, counted from start_time (candle 0)
       slope_min: fast EMA slope (ATR units) must be > this in the trade direction (0 = any rise / fall)
       stop_loss / trail_step / take_profit: position % (ROE), e.g. 0.20 = -20% -> price % = value / leverage"""
    df = load(symbol)
    start = int(df['open_time'].searchsorted(pd.Timestamp(start_time, tz='UTC')))
    if isinstance(candles, (tuple, list)):
        a, b = candles
        if not 0 <= a < b:
            raise ValueError(f'CANDLES range {candles}: need 0 <= start < end')
        start, candles = start + a, b - a
    signal_col = f'entry_{ema_set}'
    if slope_min != FAST_SLOPE_MIN:              # re-run the entry rule with this slope minimum
        df = df.copy()
        df[signal_col] = entry_signal(df, ema_set, slope_min)
    w = df.iloc[start:start + candles].copy()
    w['entry'] = w[signal_col]
    w['momentum'] = w[f'momentum_{ema_set}_15m']
    candles = len(w)
    vol_avg = df['volume_5m'].rolling(VOL_AVG_BARS).mean().shift(1).iloc[start:start + candles]
    table = rule_table(w, ema_set, slope_min)

    # simulate from the window start; a trade may run past the window
    trades = simulate_trades(df.iloc[start:].reset_index(drop=True), leverage=leverage, keep_path=True,
                             stop_loss=stop_loss, trail_step=trail_step, take_profit=take_profit, trail_gap=trail_gap,
                             last_signal=candles - 1, overlap=overlap,
                             signal_col=signal_col)
    if not trades.empty:
        trades = trades[trades['signal_idx'] < candles].reset_index(drop=True)

    # candle state: BUY / SELL (trade opened) / SKIPPED (signal while in a trade) / BLOCKED / NO ENTRY
    has_signal = (table['ENTRY'] != '-').to_numpy()
    taken = np.zeros(candles, dtype=bool)
    if not trades.empty:
        taken[trades['signal_idx'].astype(int).to_numpy()] = True
    state = np.full(candles, 'NO ENTRY', dtype=object)
    state[(w['momentum'].to_numpy() != 0) & ~has_signal] = 'BLOCKED'   # momentum, but fast EMA slope / distance against it, or candle touches the middle EMA
    state[has_signal & ~taken] = 'SKIPPED'
    state[taken] = table['ENTRY'].to_numpy()[taken]
    table['STATE'] = state

    skipped = w[has_signal & ~taken]
    return Window(symbol, df, start, candles, leverage, overlap, ema_set, rule_columns(ema_set), w, vol_avg, table, trades, skipped)
