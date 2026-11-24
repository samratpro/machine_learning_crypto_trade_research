"""Styled tables for the notebook (rule check per candle, trade list)."""
import numpy as np
import pandas as pd
from IPython.display import display

from . import theme


def rules(win):
    t, c = win.table, win.cols
    print(f"EMA test {win.ema_set} | momentum:", t['MOMENTUM'].value_counts().to_dict(),
          '| state:', t['STATE'].value_counts().to_dict())
    tick = lambda v: '✔' if v else '✘'
    on = lambda color: (lambda v: f'color: {color}' if v else f'color: {theme.MUTED}')
    return (t.style
            .format({k: tick for k in (c['bull'], c['bear'], c['flat'], c['fslope_ok'])}, precision=2)
            .map(on(theme.GREEN), subset=[c['bull'], c['flat'], c['fslope_ok']])
            .map(on(theme.RED), subset=[c['bear']])
            .map(lambda v: theme.CELL_COLORS.get(v, ''), subset=['MOMENTUM', 'ENTRY', 'STATE']))


def trades(win, margin_usdt=100):
    tr = win.trades
    if tr.empty:
        print('No trade in this window (no momentum) - move START and re-run.')
        return
    closed = tr['exit_idx'] < win.candles
    print(f"{len(tr)} trades | wins {(tr['roe'] > 0).sum()} | losses {(tr['roe'] <= 0).sum()} | "
          f"net {tr['roe'].sum() * margin_usdt:+,.2f}$ | closed inside window {closed.sum()}, after window {(~closed).sum()}")
    show = tr[['side', 'entry_time', 'entry', 'stop_init', 'take_profit', 'trail_steps',
               'exit_time', 'exit', 'exit_reason', 'bars_held', 'roe']].copy()
    show.index = range(1, len(show) + 1)
    show['P&L $'] = tr['roe'].to_numpy() * margin_usdt
    display(show.style
            .format({'roe': '{:+.1%}', 'P&L $': '{:+,.2f}', 'entry': '{:,.2f}', 'stop_init': '{:,.2f}',
                     'take_profit': '{:,.2f}', 'exit': '{:,.2f}'})
            .map(lambda v: f'color: {theme.GREEN}' if v > 0 else f'color: {theme.RED}', subset=['roe', 'P&L $']))
    if len(win.skipped):
        print(f'{len(win.skipped)} signal(s) skipped because a trade was already open (OVERLAP=False)')


# ---------------- sliding-window backtest (backtest_by_feature.ipynb) ----------------
_PCT = {'win rate', 'with trade %', 'profitable %', 'losing %', 'entry precision', 'entry recall', 'entry f1',
        'entry accuracy', 'precision', 'recall', 'f1', 'accuracy', 'specificity', 'base rate (random win)', '% of trades'}
_MONEY = {'total P&L $', 'avg $ / trade', 'avg win $', 'avg loss $', 'max drawdown $', 'avg window $',
          'median window $', 'best window $', 'worst window $', 'std window $', 'entry avg $ entry',
          'entry avg $ any candle', 'avg $ entry', 'avg $ any candle', 'avg $', 'gross profit $', 'gross loss $', 'net $'}
_SIGNED = {'total P&L $', 'avg $ / trade', 'avg window $', 'median window $', 'entry avg $ entry',
           'entry avg $ any candle', 'avg $ entry', 'avg $ any candle'}
_SIGN = lambda v: f'color: {theme.GREEN}' if v > 0 else (f'color: {theme.RED}' if v < 0 else f'color: {theme.MUTED}')


def _fmt(metric, v):
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return '-'
    if metric in _PCT:
        return f'{v:.1%}'
    if metric in _MONEY:
        return f'{v:+,.2f}'
    if metric in ('profit factor', 'window PF', 'entry lift (precision / base)', 'lift (precision / base)'):
        return 'inf' if np.isinf(v) else f'{v:.2f}'
    if metric in ('avg bars held', 'avg trades / window', 'avg trail steps'):
        return f'{v:.2f}'
    return f'{v:,.0f}'


def bt_summary(res):
    """The entire summary: every symbol + ALL. Top = period trades (each once), bottom = sliding windows."""
    mode = 'overlap (every signal trades)' if res.overlap else 'one trade at a time'
    print(f"window {res.size} candles | step {res.step} | {mode} | margin {res.margin_usdt}$ / trade")
    s = res.summary
    raw = s.copy()
    shown = pd.DataFrame({c: [_fmt(m, v) for m, v in s[c].items()] for c in s.columns}, index=s.index)
    shown.index.name = 'metric'

    def color(col):
        out = []
        for m, v in raw[col.name].items():
            if m in _SIGNED:
                out.append(_SIGN(v))
            elif m in ('profit factor', 'window PF', 'entry lift (precision / base)', 'lift (precision / base)'):
                out.append(f'color: {theme.GREEN}' if v > 1 else f'color: {theme.RED}')
            else:
                out.append('')
        return out
    idx = list(s.index)
    cuts = [idx.index(m) + 1 for m in ('windows', 'entry precision') if m in idx]
    return (shown.style.apply(color)
            .set_table_styles([{'selector': f'tbody tr:nth-child({k}) th, tbody tr:nth-child({k}) td',
                                'props': f'border-top: 2px solid {theme.MUTED}'} for k in cuts]))


def _metric_rows(df):
    """Format a metric x column table (rows = metrics)."""
    shown = pd.DataFrame({c: [_fmt(m, v) for m, v in df[c].items()] for c in df.columns}, index=df.index)

    def color(col):
        return [_SIGN(v) if m in _SIGNED else
                ((f'color: {theme.GREEN}' if v > 1 else f'color: {theme.RED}') if 'lift' in m else '')
                for m, v in df[col.name].items()]
    return shown.style.apply(color)


def bt_classification(res, symbol=None):
    """Entry signal as a classifier on every candle (precision / recall / f1 / accuracy)."""
    cl = res.classification(symbol)
    if cl is None:
        print('CLASSIFY = False - set it to True and re-run for precision / recall / f1')
        return
    b = cl['BOTH']
    print(f"{symbol or ', '.join(res.runs)} | {b['signals']:,.0f} entries | precision {b['precision']:.1%} "
          f"(random candle wins {b['base rate (random win)']:.1%}, lift {b['lift (precision / base)']:.2f}) | "
          f"recall {b['recall']:.1%} | f1 {b['f1']:.1%} | accuracy {b['accuracy']:.1%}")
    return _metric_rows(cl)


def bt_breakdown(res, symbol=None):
    """How entries make and lose money: by side, by exit reason, by ROE bucket."""
    b = res.breakdown(symbol)
    if not b:
        print('no trades')
        return
    a = b['side'].loc['all']
    print(f"{symbol or ', '.join(res.runs)} | {a['trades']:,.0f} trades\n"
          f"  wins   {a['wins']:>8,.0f}  avg {a['avg win $']:+7.2f}$  = gross profit {a['gross profit $']:+,.2f}$\n"
          f"  losses {a['losses']:>8,.0f}  avg {a['avg loss $']:+7.2f}$  = gross loss   {a['gross loss $']:+,.2f}$\n"
          f"  net {a['net $']:+,.2f}$ | win rate {a['win rate']:.1%} | profit factor {a['profit factor']:.2f} | "
          f"break-even win rate {-a['avg loss $'] / (a['avg win $'] - a['avg loss $']):.1%}")
    fmt = {'trades': '{:,.0f}', 'wins': '{:,.0f}', 'losses': '{:,.0f}', '% of trades': '{:.1%}', 'win rate': '{:.1%}',
           'avg win $': '{:+,.2f}', 'avg loss $': '{:+,.2f}', 'avg $': '{:+,.2f}', 'gross profit $': '{:+,.2f}',
           'gross loss $': '{:+,.2f}', 'net $': '{:+,.2f}', 'profit factor': '{:.2f}', 'avg bars held': '{:.1f}',
           'avg trail steps': '{:.2f}'}
    for name, title in (('side', 'by side'), ('exit', 'by exit reason'), ('roe bucket', 'by trade ROE')):
        display(b[name].style.format(fmt, na_rep='-').set_caption(title)
                .map(_SIGN, subset=['avg $', 'net $']))


def bt_windows(res, sort='best', rows=20, symbol=None):
    """sort: None = time order | 'best' | 'worst'. symbol: None = all symbols."""
    w = res.windows if symbol is None else res.windows[res.windows['symbol'] == symbol]
    if sort == 'best':
        w = w.nlargest(rows, 'net_$')
    elif sort == 'worst':
        w = w.nsmallest(rows, 'net_$')
    else:
        w = w.head(rows)
    show = w.reset_index(drop=True)
    show.index = range(1, len(show) + 1)
    show['start_time'] = show['start_time'].dt.strftime('%Y-%m-%d %H:%M')
    show['end_time'] = show['end_time'].dt.strftime('%Y-%m-%d %H:%M')
    return (show.style
            .format({'win_rate': '{:.0%}', 'net_roe': '{:+.1%}', 'net_$': '{:+,.2f}',
                     'best_roe': '{:+.1%}', 'worst_roe': '{:+.1%}'}, na_rep='-')
            .map(_SIGN, subset=['net_roe', 'net_$']))


# ---------------- momentum move, no stop (window_analysis.ipynb section 4) ----------------
def moves(mv):
    """One row per momentum entry in the window: best / worst / exit move in ATR."""
    if mv.empty:
        print('No momentum entry in this window - move START and re-run.')
        return
    print(f"{len(mv)} entries | LONG {(mv['side'] == 'long').sum()} · SHORT {(mv['side'] == 'short').sum()} | "
          f"median best move {mv['mfe_atr'].median():+.2f} ATR | median worst {mv['mae_atr'].median():+.2f} ATR | "
          f"exit in profit {(mv['end_atr'] > 0).mean():.0%}")
    show = mv[['time', 'side', 'entry', 'atr', 'mfe_atr', 'mae_atr', 'end_atr', 'bars']].copy()
    show['time'] = show['time'].dt.strftime('%m-%d %H:%M')
    show['side'] = show['side'].str.upper()
    display(show.style
            .format({'entry': '{:,.2f}', 'atr': '{:,.2f}', 'mfe_atr': '{:+.2f}', 'mae_atr': '{:+.2f}',
                     'end_atr': '{:+.2f}', 'bars': '{:,.0f}'})
            .map(lambda v: theme.CELL_COLORS.get(v, ''), subset=['side'])
            .map(_SIGN, subset=['mfe_atr', 'mae_atr', 'end_atr']))


def move_summary(s):
    """momentum_move.summary() table, formatted."""
    def f(metric, v):
        if pd.isna(v):
            return '-'
        if metric.endswith('%') or metric == 'atr % of price (median)':
            return f'{v:.1%}' if not metric.startswith('atr') else f'{v:.3%}'
        if '(ATR)' in metric:
            return f'{v:+.2f}'
        return f'{v:,.0f}'
    shown = pd.DataFrame({c: [f(m, v) for m, v in s[c].items()] for c in s.columns}, index=s.index)
    shown.columns = [str(c).upper() for c in shown.columns]
    shown.index.name = 'metric'
    display(shown.style.set_properties(**{'text-align': 'right'}))
