"""
How far does price move in our favour while the momentum lasts?  (no stop, no TP - pure momentum test)

Entry = momentum only (zone_create.entry_signal): bull -> long, bear -> short, none -> no entry.
For every entry candle i (signal at close of i, entry at the OPEN of i+1):
    the momentum "run" ends at the first candle k > i whose momentum is no longer the trade side
    (that momentum value is known at k's open, so the position is closed at open[k]).
    a        = atr_5m of the signal candle i (closed -> no look-ahead), frozen for the whole trade
    mfe_atr  = best favourable price move over candles i+1 .. k-1, in ATR     (max favourable excursion)
    mae_atr  = worst adverse price move over the same candles, in ATR (<= 0)  (max adverse excursion)
    end_atr  = move from entry to open[k] (exit when momentum ends), in ATR
    bars     = candles held (k - i - 1)
Entries whose momentum is still running at the end of the data are dropped (outcome unknown).

    from analysis.momentum_move import momentum_moves, summary
    mv = momentum_moves('ALL', ema_set='13_48_200')            # one row per entry candle
    summary(mv)                                                 # per symbol + ALL
    summary(mv, by='side')
"""
import numpy as np
import pandas as pd

from zone_create import FAST_SLOPE_MIN, entry_signal
from .window import ROOT, EMA_SETS
from .sliding import all_symbols

LEVELS = (1, 2, 3, 5, 10, 25)          # "reached k x ATR in favour" columns in summary()


def _moves_one(df: pd.DataFrame, ema_set: str, slope_min: float = FAST_SLOPE_MIN) -> pd.DataFrame:
    o, h, l = (df[f'{k}_5m'].to_numpy('float64') for k in ('open', 'high', 'low'))
    atr = df['atr_5m'].to_numpy('float64')
    side = entry_signal(df, ema_set, slope_min).astype(np.int64)
    n = len(df)

    # runs of equal momentum value: run_end[j] = last candle of j's run
    new_run = np.r_[True, side[1:] != side[:-1]]
    run_id = np.cumsum(new_run) - 1
    ends = np.r_[np.flatnonzero(new_run)[1:] - 1, n - 1]
    run_end = ends[run_id]

    # suffix max(high) / min(low) from candle j to the end of its run
    g = pd.DataFrame({'r': run_id, 'h': h, 'l': l}).iloc[::-1]
    suf_hi = g.groupby('r')['h'].cummax().to_numpy()[::-1]
    suf_lo = g.groupby('r')['l'].cummin().to_numpy()[::-1]

    i = np.flatnonzero(side != 0)
    i = i[(run_end[i] + 1 < n) & (atr[i] > 0)]          # momentum ended inside the data, ATR known
    s, a = side[i], atr[i]
    entry = o[i + 1]
    k = run_end[i] + 1                                   # first candle without the momentum -> exit at its open
    held = k - i - 1                                     # candles i+1 .. k-1
    in_run = held > 0
    nxt = np.minimum(i + 1, n - 1)
    hi = np.where(in_run, suf_hi[nxt], entry)            # best high / low while the momentum lasts
    lo = np.where(in_run, suf_lo[nxt], entry)
    best = np.where(s == 1, hi, lo)
    worst = np.where(s == 1, lo, hi)
    return pd.DataFrame({
        'time': df['open_time'].to_numpy()[i], 'side': np.where(s == 1, 'long', 'short'),
        'entry': entry, 'atr': a, 'atr_pct': a / entry,
        'mfe_atr': s * (best - entry) / a,
        'mae_atr': np.minimum(s * (worst - entry) / a, 0.0),
        'end_atr': s * (o[k] - entry) / a,
        'bars': held,
    })


def momentum_moves(symbols='ALL', ema_set: str = '13_48_200', start=None, end=None,
                   slope_min: float = FAST_SLOPE_MIN, include_holdout: bool = False) -> pd.DataFrame:
    """symbols: 'BTCUSDT' | 'ALL' | list. start / end: UTC period of the entry candles (end not included).
    include_holdout=False: entries from features.HOLDOUT_START on are left out (paper's final test)."""
    if ema_set not in EMA_SETS:
        raise ValueError(f'EMA_SET {ema_set!r}: use one of {list(EMA_SETS)}')
    if isinstance(symbols, str):
        symbols = all_symbols() if symbols.upper() == 'ALL' else [symbols]
    cols = ['open_time', 'open_5m', 'high_5m', 'low_5m', 'atr_5m', f'momentum_{ema_set}_15m',
            f'ema{EMA_SETS[ema_set][0]}_slope_15m']
    out = []
    for sym in symbols:
        df = pd.read_parquet(ROOT / 'data' / f'{sym}_master.parquet', columns=cols)
        mv = _moves_one(df, ema_set, slope_min)
        if start is not None:
            mv = mv[mv['time'] >= pd.Timestamp(start, tz='UTC')]
        if end is not None:
            mv = mv[mv['time'] < pd.Timestamp(end, tz='UTC')]
        if not include_holdout:
            from features import HOLDOUT_START
            mv = mv[mv['time'] < HOLDOUT_START]
        out.append(mv.assign(symbol=sym))
    res = pd.concat(out, ignore_index=True)
    import trial_log
    trial_log.log('momentum_move', dict(symbols='+'.join(symbols), start=start, end=end, ema_set=ema_set,
                                        slope_min=slope_min, include_holdout=include_holdout),
                  {'entries': len(res), 'mfe_median_atr': res['mfe_atr'].median(),
                   'reach5_pct': (res['mfe_atr'] >= 5).mean(), 'end_mean_atr': res['end_atr'].mean()})
    return res


def window_moves(win) -> pd.DataFrame:
    """Momentum move of every entry candle inside a window (analysis.window.build_window).
    The move may run past the window end - it is followed until the momentum ends."""
    mv = _moves_one(win.df, win.ema_set, win.slope_min)
    t = win.w['open_time']
    mv = mv[(mv['time'] >= t.iloc[0]) & (mv['time'] <= t.iloc[-1])].reset_index(drop=True)
    mv.index = mv.index + 1
    return mv


def summary(mv: pd.DataFrame, by: str = 'symbol') -> pd.DataFrame:
    """One column per group (+ ALL): how much ATR in favour the momentum gives."""
    def agg(g):
        r = {'entries': len(g),
             'mfe median (ATR)': g['mfe_atr'].median(), 'mfe mean (ATR)': g['mfe_atr'].mean(),
             'mfe p75 (ATR)': g['mfe_atr'].quantile(0.75), 'mfe p90 (ATR)': g['mfe_atr'].quantile(0.90)}
        r.update({f'reach +{k} ATR %': (g['mfe_atr'] >= k).mean() for k in LEVELS})
        r.update({'mae median (ATR)': g['mae_atr'].median(),
                  'end move mean (ATR)': g['end_atr'].mean(), 'end move > 0 %': (g['end_atr'] > 0).mean(),
                  'bars median': g['bars'].median(), 'atr % of price (median)': g['atr_pct'].median()})
        return pd.Series(r)
    res = {k: agg(g) for k, g in mv.groupby(by, sort=True)}
    if len(res) > 1:
        res['ALL'] = agg(mv)
    return pd.DataFrame(res)
