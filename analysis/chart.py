"""Plotly figure for one window: price + zones + EMAs + trades, volume, entry plan strip, trade list."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from zone_create import TAKER_BUY_RATIO
from . import theme as T

FIVE_MIN = pd.Timedelta('5min')


def _ext(s):
    """Repeat the last value so step lines/bands reach the end of the last candle."""
    s = pd.Series(s).reset_index(drop=True)
    return pd.concat([s, s.iloc[-1:]], ignore_index=True)


def _price_range(win, highlight, zoom=False):
    """zoom=False: keep candles + all 3 EMAs + highlighted trade's stop / TP in view.
       zoom=True : fit the candles + fast EMA only, so small EMA moves are visible (other lines may be cut off)."""
    levels = [win.w['low_5m'].min(), win.w['high_5m'].max()]
    emas = list(map(int, win.ema_set.split('_')))
    for p in (emas[:1] if zoom else emas):              # zoom: fast EMA only | else all 3 (incl. EMA200)
        s = win.w[f'ema{p}_15m']
        levels += [s.min(), s.max()]
    if len(win.trades):                                 # every exit price stays in view
        levels += [win.trades['exit'].min(), win.trades['exit'].max()]
    if not zoom and highlight and 1 <= highlight <= len(win.trades):
        t = win.trades.iloc[highlight - 1]
        levels += [t.stop_init, t.take_profit]
    lo, hi = min(levels), max(levels)
    pad = (hi - lo) * 0.06
    return lo - pad, hi + pad


def _zones(fig, win, x, x_ext, show_zones, y0, y1):
    w = win.w
    for tf in show_zones:
        for kind, (rgb, tag) in T.ZONE_RGB.items():
            top, bot = w[f'{kind}_hi_{tf}'], w[f'{kind}_lo_{tf}']
            edge = dict(width=1, color=f'rgba({rgb},0.7)', shape='hv')
            fig.add_trace(go.Scatter(x=x_ext, y=_ext(top), line=edge, legendgroup=kind + tf,
                                     showlegend=False, hoverinfo='skip'), 1, 1)
            fig.add_trace(go.Scatter(x=x_ext, y=_ext(bot), line=edge, fill='tonexty', legendgroup=kind + tf,
                                     fillcolor=f'rgba({rgb},{T.ZONE_ALPHA[tf]})', name=f'{tf} {tag.lower()} zone',
                                     hoverinfo='skip'), 1, 1)
            if y0 < top.iloc[0] < y1 or y0 < bot.iloc[0] < y1:
                fig.add_annotation(x=x.iloc[0], y=min(top.iloc[0], y1), yanchor='top', xanchor='left', showarrow=False,
                                   row=1, col=1, text=f'{tf} {tag}: {bot.iloc[0]:,.0f} - {top.iloc[0]:,.0f}',
                                   font=dict(size=9, color=f'rgb({rgb})'))


def _emas(fig, win, x):
    for p in map(int, win.ema_set.split('_')):        # only the selected pair's 3 EMAs
        col, color, name = f'ema{p}_15m', T.EMA_COLORS[p], f'EMA{p}'
        s = win.w[col].reset_index(drop=True)
        keep = s.ne(s.shift())          # one point per 15m update -> smooth and exact
        keep.iloc[-1] = True
        fig.add_trace(go.Scatter(x=x[keep], y=s[keep], mode='lines', name=name, line=dict(color=color, width=1.6),
                                 hovertemplate=f'{name} %{{y:,.1f}}<extra></extra>'), 1, 1)


def _candles(fig, win, x):
    w = win.w
    fig.add_trace(go.Candlestick(x=x, open=w['open_5m'], high=w['high_5m'], low=w['low_5m'], close=w['close_5m'],
                                 name='5m', showlegend=False,
                                 increasing=dict(line=dict(color=T.GREEN, width=1), fillcolor=T.GREEN),
                                 decreasing=dict(line=dict(color=T.RED, width=1), fillcolor=T.RED)), 1, 1)
    sk = win.skipped
    if len(sk):
        up = sk['entry'] == 1
        fig.add_trace(go.Scatter(x=sk['open_time'], y=np.where(up, sk['low_5m'], sk['high_5m']), mode='markers',
                                 name='skipped (already in trade)', hoverinfo='skip',
                                 marker=dict(symbol=np.where(up, 'triangle-up-open', 'triangle-down-open'),
                                             size=10, color=T.CYAN, line=dict(width=1.5))), 1, 1)


def _trade_detail(fig, win, t, n, margin_usdt):
    """Full drawing for one trade: entry / TP / stepping stop lines + labels."""
    df, start, end = win.df, win.start, win.end
    sgn = 1 if t.side == 'long' else -1
    a, b = start + int(t.entry_idx), min(start + int(t.exit_idx), end)
    tx = df['open_time'].iloc[a:b + 1].reset_index(drop=True)
    stops = pd.Series(t.stop_path[:len(tx)])
    tx_end = tx.iloc[-1] + FIVE_MIN
    roe_of = lambda px: sgn * (px / t.entry - 1) * win.leverage

    fig.add_trace(go.Scatter(x=[tx.iloc[0], tx_end], y=[t.entry] * 2, mode='lines', name=f'#{n} entry',
                             line=dict(color='white', width=1.2, dash='dot'), hoverinfo='skip'), 1, 1)
    fig.add_trace(go.Scatter(x=[tx.iloc[0], tx_end], y=[t.take_profit] * 2, mode='lines', name=f'#{n} take profit',
                             line=dict(color=T.GREEN, width=1.2, dash='dot'), hoverinfo='skip'), 1, 1)
    fig.add_trace(go.Scatter(x=pd.concat([tx, pd.Series([tx_end])], ignore_index=True), y=_ext(stops), mode='lines',
                             name=f'#{n} stop (trailing)', hoverinfo='skip',
                             line=dict(color=T.RED, width=1.8, dash='dot', shape='hv')), 1, 1)
    fig.add_annotation(x=tx.iloc[0], y=t.entry, xanchor='right', showarrow=False, row=1, col=1, yshift=10 * sgn,
                       text=f"<b>{'LONG' if sgn == 1 else 'SHORT'} #{n}</b> {t.entry:,.0f}",
                       font=dict(size=10, color=T.BLUE))
    fig.add_annotation(x=tx_end, y=t.take_profit, xanchor='left', showarrow=False, row=1, col=1,
                       text=f'TP {roe_of(t.take_profit):+.0%}', font=dict(size=10, color=T.GREEN))
    for k in range(len(stops)):
        if k == 0 or stops[k] != stops[k - 1]:
            fig.add_annotation(x=tx[k], y=stops[k], xanchor='left', showarrow=False, row=1, col=1, yshift=-9 * sgn,
                               text=f'SL {roe_of(stops[k]):+.0%}', font=dict(size=9, color=T.RED))
    if start + int(t.exit_idx) <= end:
        fig.add_annotation(x=t.exit_time, y=t.exit, xanchor='right', showarrow=False, row=1, col=1,
                           yshift=-12 * sgn, text=f'<b>OUT #{n} {t.roe * margin_usdt:+,.2f}$</b> ({t.roe:+.1%})',
                           font=dict(size=10, color=T.YELLOW))


def _trades(fig, win, margin_usdt, highlight):
    """Every trade: path entry -> exit (green win / red loss) + entry/exit markers. Returns table rows.
    Trades that exit after the window: gray dotted path to the window end, and the real exit drawn
    in the right margin (one marker per exit time/price, label = time + trade numbers)."""
    rows, later = [], {}
    end_time = win.w['open_time'].iloc[-1] + FIVE_MIN
    last_close = win.w['close_5m'].iloc[-1]
    for n, (_, t) in enumerate(win.trades.iterrows(), 1):
        sgn = 1 if t.side == 'long' else -1
        inside = win.start + int(t.exit_idx) <= win.end
        win_col = T.GREEN if t.roe > 0 else T.RED
        x1, y1 = (t.exit_time, t.exit) if inside else (end_time, last_close)
        fig.add_trace(go.Scatter(x=[t.entry_time, x1], y=[t.entry, y1], mode='lines', hoverinfo='skip',
                                 name='trade path (green win / red loss)' if inside else 'still open at window end',
                                 legendgroup='path' if inside else 'open', opacity=0.55,
                                 line=dict(color=win_col if inside else T.MUTED, width=1,
                                           dash='solid' if inside else 'dot')), 1, 1)
        fig.add_trace(go.Scatter(x=[t.entry_time], y=[t.entry], mode='markers',
                                 name='long entry' if sgn == 1 else 'short entry',
                                 marker=dict(symbol='triangle-up' if sgn == 1 else 'triangle-down', size=8,
                                             color=T.GREEN if sgn == 1 else T.RED, line=dict(width=0.5, color='black')),
                                 hovertemplate=f'#{n} {t.side.upper()} {t.entry:,.2f} -> {t.exit_reason} '
                                               f'{pd.Timestamp(t.exit_time):%m-%d %H:%M} {t.exit:,.2f} {t.roe:+.1%}<extra></extra>'), 1, 1)
        if inside:
            fig.add_trace(go.Scatter(x=[t.exit_time], y=[t.exit], mode='markers', name='exit',
                                     marker=dict(symbol='x', size=9, color=win_col, line=dict(width=1, color='black')),
                                     hovertemplate=f'#{n} exit {t.exit_reason} {t.exit:,.2f} {t.roe:+.1%}<extra></extra>'), 1, 1)
        else:
            key = (pd.Timestamp(t.exit_time), round(float(t.exit), 2), t.exit_reason)
            later.setdefault(key, []).append((n, t.roe))
        if n == highlight:
            _trade_detail(fig, win, t, n, margin_usdt)
        rows.append([n, t.side.upper(), pd.Timestamp(t.entry_time).strftime('%m-%d %H:%M'), f'{t.entry:,.2f}',
                     f'{t.stop_init:,.2f}', f'{t.take_profit:,.2f}', int(t.trail_steps),
                     pd.Timestamp(t.exit_time).strftime('%m-%d %H:%M'), f'{t.exit:,.2f}', t.exit_reason,
                     int(t.bars_held), f'{t.roe:+.1%}', f'{t.roe * margin_usdt:+,.2f}'])
    x_out = end_time + FIVE_MIN                   # right margin = "after the window"
    for (et, px, reason), tl in later.items():
        nums = ','.join(f'#{n}' for n, _ in tl)
        avg = sum(r for _, r in tl) / len(tl)
        col = T.GREEN if avg > 0 else T.RED
        fig.add_trace(go.Scatter(x=[x_out], y=[px], mode='markers', name='exit after window',
                                 marker=dict(symbol='x', size=10, color=col, line=dict(width=1, color='black')),
                                 hovertemplate=f'{nums} {reason} {et:%m-%d %H:%M} @ {px:,.2f} · avg {avg:+.1%}<extra></extra>'), 1, 1)
        fig.add_annotation(x=x_out, y=px, xanchor='left', showarrow=False, row=1, col=1, xshift=8,
                           text=f'<b>OUT {et:%H:%M}</b> {reason} {px:,.0f} ({nums})', font=dict(size=9, color=col))
    if later:
        fig.add_vline(x=end_time, line=dict(color=T.MUTED, width=1, dash='dash'), row=1, col=1)
        fig.add_annotation(x=end_time, y=1, yref='y domain', xanchor='left', showarrow=False, row=1, col=1,
                           text='after window ->', font=dict(size=9, color=T.MUTED))
    return rows


def _last_price(fig, win):
    last = win.w['close_5m'].iloc[-1]
    fig.add_hline(y=last, line=dict(color=T.BLUE, width=1, dash='dot'), row=1, col=1)
    fig.add_annotation(x=1, xref='x domain', y=last, yref='y', xanchor='left', showarrow=False, bgcolor=T.BLUE,
                       text=f'<b>{last:,.2f}</b>', font=dict(color='white', size=10))


def _volume(fig, win, x):
    br = win.w['taker_buy_ratio_5m']
    vcol = np.where(br > TAKER_BUY_RATIO, T.GREEN, np.where(br < 1 - TAKER_BUY_RATIO, T.RED, T.GRAY))
    fig.add_trace(go.Bar(x=x, y=win.w['volume_5m'], marker_color=vcol, opacity=0.8, showlegend=False, name='volume',
                         customdata=br, hovertemplate='vol %{y:,.0f} · buy %{customdata:.0%}<extra></extra>'), 2, 1)
    fig.add_trace(go.Scatter(x=x, y=win.vol_avg, mode='lines', line=dict(color=T.MUTED, width=1, dash='dot'),
                             showlegend=False, hoverinfo='skip'), 2, 1)


def _plan_strip(fig, win, x):
    t, c = win.table, win.cols
    tick = lambda v: '✔' if v else '✘'
    hover = [f"<b>#{i} {st}</b> · momentum {m}<br>{c['bull']} {tick(bu)} · {c['bear']} {tick(be)}<br>"
             f"{c['slope']} {sl:+.2f} ATR · not flat {tick(fl)}<br>{c['fslope']} {fs:+.2f} ATR · ok {tick(fo)}<br>"
             f"{c['dist']} {d:+.2f} · ok {tick(ok)}<br>{c['touch']} {tick(tc)}"
             for i, st, m, bu, be, sl, fl, fs, fo, d, ok, tc in zip(
                 t.index, t['STATE'], t['MOMENTUM'], t[c['bull']], t[c['bear']], t[c['slope']], t[c['flat']],
                 t[c['fslope']], t[c['fslope_ok']], t[c['dist']], t[c['dist_ok']], t[c['touch']])]
    fig.add_trace(go.Bar(x=x, y=[1] * len(t), marker_color=t['STATE'].map(T.STATE_COLORS), showlegend=False,
                         name='plan', hovertext=hover, hoverinfo='text',
                         text=t.index.astype(str), textposition='inside', insidetextanchor='middle',
                         textfont=dict(color='#0b0f17', size=10), textangle=0,
                         cliponaxis=False), 3, 1)     # candle # = row in tables.rules(win)
    legend = '  '.join(f'<span style="color:{col}">■ {k.lower()}</span>' for k, col in T.STATE_COLORS.items())
    label = win.ema_set.replace('_', '/')
    fig.add_annotation(text=f'ENTRY PLAN PER CANDLE (EMA {label})   {legend}', x=0,
                       xref='x3 domain', y=1.35, yref='y3 domain', showarrow=False, xanchor='left',
                       font=dict(size=10, color=T.MUTED))


def _trade_table(fig, rows):
    hdr = ['#', 'Side', 'Entry time', 'Entry', 'Stop', 'Take profit', 'Trail', 'Exit time', 'Exit', 'Reason',
           'Bars', 'ROE', 'P&L $']
    if not rows:
        rows = [['-', 'No trade in this window - move START'] + [''] * 11]
    cols = list(map(list, zip(*rows)))
    pnl_col = [T.GREEN if str(v).startswith('+') else (T.RED if str(v).startswith('-') and v != '-' else T.TXT)
               for v in cols[-1]]
    fig.add_trace(go.Table(
        columnwidth=[3, 6, 10, 9, 9, 9, 5, 10, 9, 10, 5, 7, 8],
        header=dict(values=[f'<b>{h}</b>' for h in hdr], fill_color=T.HEADER, font=dict(color=T.MUTED, size=11),
                    align='left', line_color=T.GRID, height=26),
        cells=dict(values=cols, fill_color=T.PANEL, align='left', line_color=T.GRID, height=22,
                   font=dict(color=[T.TXT] * 11 + [pnl_col, pnl_col], size=11))), 4, 1)


def plot_window(win, margin_usdt=100, show_zones=('15m', '1h'), highlight=1, height=1150, zoom=False):
    """highlight: trade number (1 = first) drawn with full stop/TP detail; None for none.
       zoom: True -> y-axis fits the candles + fast EMA (EMA slope easy to see), False -> everything in view."""
    x = win.w['open_time'].reset_index(drop=True)
    x_ext = pd.concat([x, pd.Series([x.iloc[-1] + FIVE_MIN])], ignore_index=True)
    y0, y1 = _price_range(win, highlight, zoom)

    fig = make_subplots(rows=4, cols=1, row_heights=[0.52, 0.09, 0.05, 0.34], vertical_spacing=0.03,
                        specs=[[{}], [{}], [{}], [{'type': 'table'}]])
    _zones(fig, win, x, x_ext, show_zones, y0, y1)
    _emas(fig, win, x)
    _candles(fig, win, x)
    rows = _trades(fig, win, margin_usdt, highlight)
    _last_price(fig, win)
    _volume(fig, win, x)
    _plan_strip(fig, win, x)
    _trade_table(fig, rows)

    tr = win.trades
    net = tr['roe'].sum() * margin_usdt if len(tr) else 0.0
    wins = int((tr['roe'] > 0).sum()) if len(tr) else 0
    ema_label = win.ema_set.replace('_', '/')
    title = (f"<b>{win.symbol} · 5m · EMA {ema_label}</b>   {x.iloc[0]:%Y-%m-%d %H:%M} → {x.iloc[-1]:%H:%M}   "
             f"<span style='color:{T.GREEN if net >= 0 else T.RED}'>{len(tr)} trades · {net:+,.2f}$</span>   "
             f"W {wins} / L {len(tr) - wins}   ·   {'every candle' if win.overlap else 'one at a time'}"
             f"{'' if win.overlap else f' (skipped {len(win.skipped)})'}   ·   "
             f"{win.leverage:g}x · ${margin_usdt}/trade   ·   momentum now: {win.table['MOMENTUM'].iloc[-1]}")

    later = len(win.trades) and (win.trades['exit_idx'] >= win.candles).any()
    xr = [x.iloc[0] - FIVE_MIN, x.iloc[-1] + (16 if later else 6) * FIVE_MIN]   # room for exit-after-window labels
    for r in (1, 2, 3):
        fig.update_xaxes(range=xr, row=r, col=1, showgrid=r == 1, gridcolor=T.GRID, rangeslider_visible=False,
                         showticklabels=r == 3, matches='x', tickformat='%H:%M', color=T.MUTED)
    fig.update_yaxes(range=[y0, y1], row=1, col=1, side='right', gridcolor=T.GRID, tickformat=',.0f', color=T.MUTED)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=2, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=3, col=1)
    fig.update_layout(**T.base_layout(title, height))
    fig.update_layout(hovermode='closest', uniformtext=dict(minsize=7, mode='show'))     # many trades: hover one item at a time

    seen = set()                                 # one legend entry per name
    for trace in fig.data:
        if trace.type == 'table':
            continue
        if trace.name in seen:
            trace.showlegend = False
        seen.add(trace.name)
    return fig
