"""Styled tables for the notebook (rule check per candle, trade list)."""
from IPython.display import display

from . import theme


def rules(win):
    t, c = win.table, win.cols
    print(f"EMA test {win.ema_set} | momentum:", t['MOMENTUM'].value_counts().to_dict(),
          '| state:', t['STATE'].value_counts().to_dict())
    tick = lambda v: '✔' if v else '✘'
    on = lambda color: (lambda v: f'color: {color}' if v else f'color: {theme.MUTED}')
    return (t.style
            .format({k: tick for k in (c['bull'], c['bear'], c['flat'], c['fslope_ok'], c['dist_ok'], c['touch'])},
                    precision=2)
            .map(on(theme.GREEN), subset=[c['bull'], c['fslope_ok'], c['dist_ok']])
            .map(on(theme.RED), subset=[c['bear'], c['touch']])
            .map(on(theme.YELLOW), subset=[c['flat']])
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
