"""
Sliding-window backtest (no plotting here).

The same entry / stop-loss rules as window.py, run on every window of the period:
    WINDOW_SIZE 30, STEP 1  ->  candles 1-30, 2-31, 3-32, 4-33, ... up to the last candle
A window owns the trades whose entry signal is inside it (a trade may close after the window ends),
exactly like build_window().

    res = run_backtest('BTCUSDT', size=30)                 # one symbol, all data
    res = run_backtest('ALL', size=30)                     # every symbol in data/
    res = run_backtest(['BTCUSDT', 'ETHUSDT'], size=30, start='2024-01-01', end='2025-01-01')
    res.runs['BTCUSDT'].trades   -> trades of the period, each counted once
    res.runs['BTCUSDT'].windows  -> one row per window
    res.windows                  -> all windows of all symbols (column 'symbol')
    res.summary                  -> metric x symbol table (+ 'ALL' column when several symbols)

overlap=True : every signal candle opens its own trade, so a window's result = sum of its signals' trades
overlap=False: one trade at a time, re-started at each window's first candle (signals while in a trade are skipped);
               the period trades are the usual one-at-a-time backtest from the first candle
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from stop_loss import simulate_trades, STOP_LOSS, TRAIL_STEP, TAKE_PROFIT, TRAIL_GAP, FEE_RATE, ATR_COL, MAX_BARS
from zone_create import FAST_SLOPE_MIN, entry_signal
from .window import ROOT, EMA_SETS


def all_symbols() -> list[str]:
    return sorted(p.name.replace('_master.parquet', '') for p in (ROOT / 'data').glob('*_master.parquet'))


def _load(symbol: str, ema_set: str) -> pd.DataFrame:
    """Only the columns the backtest needs (keeps memory low for ALL symbols)."""
    f = EMA_SETS[ema_set][0]
    cols = ['open_time', 'open_5m', 'high_5m', 'low_5m', 'close_5m', f'momentum_{ema_set}_15m',
            f'ema{f}_slope_15m', ATR_COL]
    return pd.read_parquet(ROOT / 'data' / f'{symbol}_master.parquet', columns=cols)


@dataclass
class Run:                       # one symbol
    symbol: str
    first: pd.Timestamp
    last: pd.Timestamp
    candles: int
    trades: pd.DataFrame         # period trades, each counted once
    windows: pd.DataFrame        # one row per window
    signal: np.ndarray = None    # entry signal per period candle (1 long, -1 short, 0 none)
    long_roe: np.ndarray = None  # ROE a long opened after this candle would make (same stop / trail / TP rules)
    short_roe: np.ndarray = None # ROE a short would make


def _outcomes(o, h, l, c, atr, count, side, sl, step, tp, gap, lev, fee, max_bars=MAX_BARS):
    """ROE of a trade (side 1 long / -1 short) opened at the next open after every candle 0..count-1.
    sl / step / tp / gap are ATR multiples (atr of the signal candle).
    Same bar logic as stop_loss._run_trade (stop first, then TP, then trail), written lean for speed."""
    n = len(o)
    gap = step if gap is None else gap
    o, h, l, atr, cl = o.tolist(), h.tolist(), l.tolist(), atr.tolist(), c.tolist()
    last = c[-1]
    out = np.full(count, np.nan)
    cost = 2 * fee * lev
    for i in range(min(count, n - 1)):
        a = atr[i]
        if not a > 0:
            continue
        e = i + 1
        entry = o[e]
        sl_d, step_d, tp_d, gap_d = sl * a, step * a * (1 - 1e-9), tp * a, gap * a
        j, px = e, None
        end = n if max_bars is None else min(n, e + max_bars)
        if side == 1:
            stop, target = entry - sl_d, entry + tp_d
            while j < end:
                if l[j] <= stop:
                    px = min(o[j], stop) if j > e else stop
                    break
                if h[j] >= target:
                    px = max(o[j], target) if j > e else target
                    break
                if h[j] - entry >= step_d:
                    t = h[j] - gap_d
                    if t > stop:
                        stop = t
                j += 1
            if px is None and end < n:
                px = cl[end - 1]                     # time limit
            out[i] = ((last if px is None else px) / entry - 1) * lev - cost
        else:
            stop, target = entry + sl_d, entry - tp_d
            while j < end:
                if h[j] >= stop:
                    px = max(o[j], stop) if j > e else stop
                    break
                if l[j] <= target:
                    px = min(o[j], target) if j > e else target
                    break
                if entry - l[j] >= step_d:
                    t = l[j] + gap_d
                    if t < stop:
                        stop = t
                j += 1
            if px is None and end < n:
                px = cl[end - 1]                     # time limit
            out[i] = (1 - (last if px is None else px) / entry) * lev - cost
    return out


def _confusion(pred: np.ndarray, actual: np.ndarray) -> dict:
    ok = ~np.isnan(actual)
    p, a = pred[ok], actual[ok] > 0
    return {'TP': int((p & a).sum()), 'FP': int((p & ~a).sum()),
            'FN': int((~p & a).sum()), 'TN': int((~p & ~a).sum())}


def _scores(cm: dict, roe_pred=None, roe_all=None, margin=100) -> dict:
    tp, fp, fn, tn = cm['TP'], cm['FP'], cm['FN'], cm['TN']
    tot = tp + fp + fn + tn
    div = lambda a, b: a / b if b else np.nan
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    base = div(tp + fn, tot)
    return {
        'signals': tp + fp, 'TP (entry, won)': tp, 'FP (entry, lost)': fp,
        'FN (no entry, would win)': fn, 'TN (no entry, would lose)': tn,
        'precision': prec, 'recall': rec,
        'f1': div(2 * prec * rec, prec + rec) if tp else 0.0,
        'accuracy': div(tp + tn, tot), 'specificity': div(tn, tn + fp),
        'base rate (random win)': base, 'lift (precision / base)': div(prec, base),
        'avg $ entry': np.nanmean(roe_pred) * margin if roe_pred is not None and len(roe_pred) else np.nan,
        'avg $ any candle': np.nanmean(roe_all) * margin if roe_all is not None and len(roe_all) else np.nan,
    }


def classification(runs, margin: float = 100) -> pd.DataFrame:
    """Entry signal as a classifier, on every candle of the period (all signals, whatever OVERLAP is):
    LONG : predicted = long signal,  actual = a long from this candle ends with ROE > 0
    SHORT: predicted = short signal, actual = a short from this candle ends with ROE > 0
    BOTH : LONG + SHORT counts added (micro average)."""
    runs = [r for r in runs if r.long_roe is not None]
    if not runs:
        return None
    sig = np.concatenate([r.signal for r in runs])
    lr = np.concatenate([r.long_roe for r in runs])
    sr = np.concatenate([r.short_roe for r in runs])
    lng, sht = _confusion(sig == 1, lr), _confusion(sig == -1, sr)
    both = {k: lng[k] + sht[k] for k in lng}
    out = pd.DataFrame({
        'LONG': _scores(lng, lr[sig == 1], lr, margin),
        'SHORT': _scores(sht, sr[sig == -1], sr, margin),
        'BOTH': _scores(both, np.concatenate([lr[sig == 1], sr[sig == -1]]), np.concatenate([lr, sr]), margin),
    })
    out.index.name = 'metric'
    return out


ROE_BUCKETS = [-np.inf, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30, 0.45, np.inf]


def breakdown(trades: pd.DataFrame, margin: float = 100) -> dict:
    """Where profits and losses come from: by side, by exit reason, by ROE bucket."""
    if trades.empty:
        return {}
    tr = trades.assign(pnl=trades['roe'] * margin, win=trades['roe'] > 0)

    def agg(g):
        win, loss = g.loc[g['win'], 'pnl'], g.loc[~g['win'], 'pnl']
        return pd.Series({
            'trades': len(g), '% of trades': len(g) / len(tr), 'wins': int(g['win'].sum()),
            'losses': int((~g['win']).sum()), 'win rate': g['win'].mean(),
            'avg win $': win.mean(), 'avg loss $': loss.mean(), 'avg $': g['pnl'].mean(),
            'gross profit $': win.sum(), 'gross loss $': loss.sum(), 'net $': g['pnl'].sum(),
            'profit factor': win.sum() / -loss.sum() if loss.sum() < 0 else np.inf,
            'avg bars held': g['bars_held'].mean(), 'avg trail steps': g['trail_steps'].mean(),
        })
    by = lambda key: pd.DataFrame({k: agg(g) for k, g in tr.groupby(key, observed=True, sort=False)}).T
    side = by('side').reindex(['long', 'short']).dropna(how='all')
    side.loc['all'] = agg(tr)
    tr['exit'] = tr['exit_reason'] + np.where(tr['win'], ' (win)', ' (loss)')
    order = ['stop (loss)', 'trail (loss)', 'trail (win)', 'take_profit (win)', 'time (loss)', 'time (win)',
             'open_eod (loss)', 'open_eod (win)']
    exit_ = by('exit').reindex(order).dropna(how='all')
    b = ROE_BUCKETS
    labels = [f'< {b[1]:+.0%}'] + [f'{b[k]:+.0%} to {b[k + 1]:+.0%}' for k in range(1, len(b) - 2)] + [f'> {b[-2]:+.0%}']
    tr['roe bucket'] = pd.cut(tr['roe'], b, labels=labels)
    bucket = by('roe bucket').reindex(labels).dropna(how='all')
    return {'side': side, 'exit': exit_, 'roe bucket': bucket}


@dataclass
class Result:
    runs: dict                   # symbol -> Run
    size: int
    step: int
    overlap: bool
    margin_usdt: float
    windows: pd.DataFrame        # all windows, column 'symbol'
    summary: pd.DataFrame        # metric x symbol (+ ALL)

    def trades(self, symbol=None) -> pd.DataFrame:
        runs = self.runs.values() if symbol is None else [self.runs[symbol]]
        return pd.concat([r.trades for r in runs], ignore_index=True)

    def classification(self, symbol=None) -> pd.DataFrame:
        return classification(self.runs.values() if symbol is None else [self.runs[symbol]], self.margin_usdt)

    def breakdown(self, symbol=None) -> dict:
        return breakdown(self.trades(symbol), self.margin_usdt)


def _table(starts, size, t, n, wins, net, best, worst, margin) -> pd.DataFrame:
    return pd.DataFrame({
        'window': np.arange(1, len(starts) + 1),
        'candles': [f'{s + 1}-{s + size}' for s in starts],
        'start_time': t[starts], 'end_time': t[starts + size - 1],
        'trades': n, 'wins': wins, 'losses': n - wins,
        'win_rate': np.where(n > 0, wins / np.maximum(n, 1), np.nan),
        'net_roe': net, 'net_$': net * margin, 'best_roe': best, 'worst_roe': worst,
    })


def _windows_overlap(sig_roe, starts, size, t, margin):
    """sig_roe[i] = ROE of the trade opened by candle i's signal, NaN = no signal."""
    v = sliding_window_view(sig_roe, size)[starts]
    has = ~np.isnan(v)
    n = has.sum(1)
    wins = (np.where(has, v, 0) > 0).sum(1)
    net = np.where(has, v, 0).sum(1)
    best = np.where(n > 0, np.where(has, v, -np.inf).max(1), np.nan)
    worst = np.where(n > 0, np.where(has, v, np.inf).min(1), np.nan)
    return _table(starts, size, t, n, wins, net, best, worst, margin)


def _next_signal(sig_roe):
    N = len(sig_roe)
    idx = np.append(np.flatnonzero(~np.isnan(sig_roe)), N)     # signal candles + sentinel N
    return idx[np.searchsorted(idx, np.arange(N + 1))]          # nxt[i] = first signal candle >= i (N = none)


def _chain(nxt, exit_idx, s, end):
    """Signal candles taken one-at-a-time from candle s, signals before `end`."""
    N = len(nxt) - 1
    k, out = nxt[s], []
    while k < end:
        out.append(k)
        k = nxt[min(exit_idx[k] + 1, N)]
    return out


def _windows_single(sig_roe, exit_idx, starts, size, t, margin):
    nxt = _next_signal(sig_roe)
    m = len(starts)
    n, wins = np.zeros(m, int), np.zeros(m, int)
    net, best, worst = np.zeros(m), np.full(m, np.nan), np.full(m, np.nan)
    for j, s in enumerate(starts):
        r = sig_roe[_chain(nxt, exit_idx, s, s + size)]
        if len(r):
            n[j], wins[j], net[j], best[j], worst[j] = len(r), (r > 0).sum(), r.sum(), r.max(), r.min()
    return _table(starts, size, t, n, wins, net, best, worst, margin)


def _run_symbol(symbol, start, end, size, step, leverage, overlap, ema_set, stop_loss, trail_step,
                take_profit, trail_gap, slope_min, margin, classify, max_bars=MAX_BARS) -> Run | None:
    df = _load(symbol, ema_set)
    t_all = df['open_time']
    p0 = 0 if start is None else int(t_all.searchsorted(pd.Timestamp(start, tz='UTC')))
    p1 = len(df) if end is None else int(t_all.searchsorted(pd.Timestamp(end, tz='UTC')))
    N = p1 - p0
    if N < size:
        print(f'{symbol}: skipped, period has {N} candles < WINDOW_SIZE {size}')
        return None
    df = df.iloc[p0:].reset_index(drop=True)          # trades may run past the period end
    signal_col = f'entry_{ema_set}'
    df[signal_col] = entry_signal(df, ema_set, slope_min)   # momentum only (incl. fast EMA slope minimum)

    # every signal's own trade; overlap=False only changes which ones are taken
    tr = simulate_trades(df, signal_col=signal_col, leverage=leverage, stop_loss=stop_loss,
                         trail_step=trail_step, take_profit=take_profit, trail_gap=trail_gap,
                         last_signal=N - 1, overlap=True, max_bars=max_bars)
    sig_roe, exit_idx = np.full(N, np.nan), np.zeros(N, dtype=np.int64)
    if not tr.empty:
        tr = tr[tr['signal_idx'] < N].reset_index(drop=True)
        k = tr['signal_idx'].to_numpy(int)
        sig_roe[k], exit_idx[k] = tr['roe'].to_numpy(), tr['exit_idx'].to_numpy(int)

    t = df['open_time'].to_numpy()[:N]
    starts = np.arange(0, N - size + 1, step)
    if overlap:
        windows = _windows_overlap(sig_roe, starts, size, t, margin)
    else:
        windows = _windows_single(sig_roe, exit_idx, starts, size, t, margin)
        if not tr.empty:
            keep = set(_chain(_next_signal(sig_roe), exit_idx, 0, N))
            tr = tr[tr['signal_idx'].isin(keep)].reset_index(drop=True)
    tr.insert(0, 'symbol', symbol)
    windows.insert(0, 'symbol', symbol)
    run = Run(symbol, pd.Timestamp(t[0]), pd.Timestamp(t[-1]), N, tr, windows)
    if classify:
        o, h, l, c = (df[f'{k}_5m'].to_numpy(dtype='float64') for k in ('open', 'high', 'low', 'close'))
        atr = df[ATR_COL].to_numpy(dtype='float64')
        args = (stop_loss, trail_step, take_profit, trail_gap, leverage, FEE_RATE, max_bars)
        run.signal = df[signal_col].to_numpy()[:N]
        run.long_roe = _outcomes(o, h, l, c, atr, N, 1, *args)
        run.short_roe = _outcomes(o, h, l, c, atr, N, -1, *args)
    return run


def _metrics(trades: pd.DataFrame, windows: pd.DataFrame, margin: float, candles: int, first, last) -> dict:
    r = trades['roe'] if len(trades) else pd.Series(dtype=float)
    pnl = r * margin
    win_pnl, loss_pnl = pnl[pnl > 0].sum(), pnl[pnl <= 0].sum()
    equity = (trades.sort_values('exit_time')['roe'] * margin).cumsum() if len(trades) else pd.Series([0.0])
    exits = trades['exit_reason'].value_counts() if len(trades) else pd.Series(dtype=int)
    net = windows['net_$']
    return {
        # ---- period (each trade counted once)
        'from': first.strftime('%Y-%m-%d'), 'to': last.strftime('%Y-%m-%d'), 'candles': candles,
        'trades': len(r), 'wins': int((r > 0).sum()), 'losses': int((r <= 0).sum()),
        'win rate': (r > 0).mean() if len(r) else np.nan,
        'total P&L $': pnl.sum(), 'avg $ / trade': pnl.mean() if len(r) else np.nan,
        'avg win $': pnl[pnl > 0].mean(), 'avg loss $': pnl[pnl <= 0].mean(),
        'profit factor': win_pnl / -loss_pnl if loss_pnl < 0 else np.inf,
        'max drawdown $': (equity - equity.cummax()).min(),
        'avg bars held': trades['bars_held'].mean() if len(trades) else np.nan,
        'exit stop': int(exits.get('stop', 0)), 'exit trail': int(exits.get('trail', 0)),
        'exit take_profit': int(exits.get('take_profit', 0)), 'exit time': int(exits.get('time', 0)),
        'exit open_eod': int(exits.get('open_eod', 0)),
        # ---- sliding windows
        'windows': len(windows),
        'with trade %': (windows['trades'] > 0).mean(),
        'profitable %': (net > 0).mean(), 'losing %': (net < 0).mean(),
        'avg window $': net.mean(), 'median window $': net.median(),
        'best window $': net.max(), 'worst window $': net.min(), 'std window $': net.std(),
        'avg trades / window': windows['trades'].mean(),
        'window PF': net[net > 0].sum() / -net[net < 0].sum() if (net < 0).any() else np.inf,
    }


def run_backtest(symbols='BTCUSDT', size: int = 30, step: int = 1, start=None, end=None,
                 leverage: float = 10, overlap: bool = True, ema_set: str = '9_21_200',
                 stop_loss: float = STOP_LOSS, trail_step: float = TRAIL_STEP, take_profit: float = TAKE_PROFIT,
                 trail_gap=TRAIL_GAP, slope_min: float = FAST_SLOPE_MIN, max_bars=MAX_BARS,
                 margin_usdt: float = 100,
                 classify: bool = True, verbose: bool = True, include_holdout: bool = False) -> Result:
    """symbols: 'BTCUSDT' | 'ALL' (every data/*_master.parquet) | ['BTCUSDT', 'ETHUSDT', ...]
       size: candles per window; step: candles between window starts (1 = 1-30, 2-31, ...)
       start / end: period (UTC, end not included; None = first / last candle of the data).
       stop_loss / trail_step / take_profit / trail_gap: ATR multiples (atr_5m at the signal candle).
       classify: also simulate a long and a short from every candle -> precision / recall / f1 of the entry (~11 s / symbol).
       include_holdout: False (default) cuts the period at features.HOLDOUT_START - the last 6 months are the
                        paper's final test and must not be used for tuning. Every call is logged to results/trials.csv."""
    if ema_set not in EMA_SETS:
        raise ValueError(f'EMA_SET {ema_set!r}: use one of {list(EMA_SETS)}')
    size, step = int(size), int(step)
    if size < 1 or step < 1:
        raise ValueError('WINDOW_SIZE and STEP must be >= 1')
    from features import HOLDOUT_START
    if not include_holdout:
        e = None if end is None else pd.Timestamp(end, tz='UTC')
        if e is None or e > HOLDOUT_START:
            if verbose:
                print(f'period cut at the holdout start {HOLDOUT_START:%Y-%m-%d} (include_holdout=True to use it)')
            end = HOLDOUT_START.tz_localize(None)
    if isinstance(symbols, str):
        symbols = all_symbols() if symbols.upper() == 'ALL' else [symbols]
    runs = {}
    for s in symbols:
        run = _run_symbol(s, start, end, size, step, leverage, overlap, ema_set, stop_loss, trail_step,
                          take_profit, trail_gap, slope_min, margin_usdt, classify, max_bars)
        if run is not None:
            runs[s] = run
            if verbose:
                print(f'{s}: {run.candles:,} candles | {len(run.trades):,} trades | {len(run.windows):,} windows')
    if not runs:
        raise ValueError('no symbol has enough candles in this period')

    cols = {s: _metrics(r.trades, r.windows, margin_usdt, r.candles, r.first, r.last) for s, r in runs.items()}
    windows = pd.concat([r.windows for r in runs.values()], ignore_index=True)
    if len(runs) > 1:
        trades = pd.concat([r.trades for r in runs.values()], ignore_index=True)
        cols['ALL'] = _metrics(trades, windows, margin_usdt, sum(r.candles for r in runs.values()),
                               min(r.first for r in runs.values()), max(r.last for r in runs.values()))
    if classify:                                        # entry as a classifier (BOTH = long + short)
        groups = {s: [r] for s, r in runs.items()}
        if len(runs) > 1:
            groups['ALL'] = list(runs.values())
        for s, g in groups.items():
            both = classification(g, margin_usdt)['BOTH']
            for k in ('precision', 'recall', 'f1', 'accuracy', 'lift (precision / base)', 'avg $ entry', 'avg $ any candle'):
                cols[s][f'entry {k}'] = both[k]
    summary = pd.DataFrame(cols)
    import trial_log
    key = 'ALL' if 'ALL' in summary.columns else summary.columns[0]
    pick = ['trades', 'win rate', 'profit factor', 'total P&L $', 'avg $ / trade', 'max drawdown $',
            'entry precision', 'entry lift (precision / base)']
    trial_log.log('backtest', dict(symbols='+'.join(runs), start=start, end=end, size=size, step=step,
                                   leverage=leverage, overlap=overlap, ema_set=ema_set, slope_min=slope_min,
                                   stop_loss=stop_loss, trail_step=trail_step, take_profit=take_profit,
                                   trail_gap=trail_gap, max_bars=max_bars, include_holdout=include_holdout),
                  {k: summary.at[k, key] for k in pick if k in summary.index})
    return Result(runs, size, step, overlap, margin_usdt, windows, summary)


def pick_window(res: Result, which, symbol=None) -> pd.Series:
    """which: 'best' | 'worst' (over all symbols, or `symbol`) | window number (needs symbol when several)."""
    w = res.windows if symbol is None else res.windows[res.windows['symbol'] == symbol]
    if which == 'best':
        return w.loc[w['net_$'].idxmax()]
    if which == 'worst':
        return w.loc[w['net_$'].idxmin()]
    if symbol is None and len(res.runs) > 1:
        raise ValueError('INSPECT is a window number: set INSPECT_SYMBOL too')
    return w.iloc[int(which) - 1]
