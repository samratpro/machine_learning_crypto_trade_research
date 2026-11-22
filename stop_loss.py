"""
Stop-loss / trailing / take-profit simulation (docs/stop_loss.md).

  - Keep 20% loss position  -> initial stop at -20% position PnL
  - Trail each 10% move     -> once the trade reaches +10%, the stop follows the best price
                               10% behind it and never moves down
                               (+10% -> stop 0% break-even, +15% -> +5%, +30% -> +20%, ...)
                               TRAIL_GAP sets that distance (default = TRAIL_STEP); e.g. 0.05 -> +10% -> stop +5%
  - Take profit reach 50%   -> exit at +50% position PnL

Percentages are position PnL (ROE), i.e. price move x LEVERAGE.
With LEVERAGE = 10: stop = -2% price, trail step = 1% price, TP = +5% price.

Works for long and short (short = mirror image).

Bar handling (5m, conservative - no look-ahead inside a bar):
  1. entry at the OPEN of the bar after the signal
  2. check stop (gap through the stop fills at the open)
  3. check take profit
  4. only then ratchet the trailing stop using this bar's high
  If stop and TP are both inside one bar, the stop is assumed hit first.
"""
import numpy as np
import pandas as pd

STOP_LOSS = 0.20
TRAIL_STEP = 0.10
TAKE_PROFIT = 0.50
TRAIL_GAP = None          # how far the stop follows behind the best price (position %); None = TRAIL_STEP
LEVERAGE = 10
FEE_RATE = 0.00055        # per side, on notional (Bybit taker)


def _run_trade(o, h, l, e, side, sl_px, step_px, tp_px, gap_px=None):
    """Walk one trade from bar e. side = 1 long, -1 short.
    Returns (exit_idx, exit_px, reason, steps, stops); stops[k] = stop active during bar e+k."""
    n = len(o)
    gap_px = step_px if gap_px is None else gap_px
    entry = o[e]
    stop = entry * (1 - side * sl_px)
    tp = entry * (1 + side * tp_px)
    steps, stops = 0, []
    j = e
    while j < n:
        stops.append(stop)
        stop_hit = l[j] <= stop if side == 1 else h[j] >= stop
        if stop_hit:
            gap = (min if side == 1 else max)(o[j], stop) if j > e else stop
            return j, gap, ("stop" if steps == 0 else "trail"), steps, stops
        tp_hit = h[j] >= tp if side == 1 else l[j] <= tp
        if tp_hit:
            gap = (max if side == 1 else min)(o[j], tp) if j > e else tp
            return j, gap, "take_profit", steps, stops
        best = h[j] if side == 1 else l[j]
        gain = side * (best / entry - 1)                 # best price move this bar (price %)
        steps = max(steps, int(np.floor(gain / step_px + 1e-9)))
        if gain >= step_px - 1e-12:                      # in profit by one step -> follow the price
            trail = best * (1 - side * gap_px)           # gap_px behind the best price
            stop = max(stop, trail) if side == 1 else min(stop, trail)
        j += 1
    return None, None, "open_eod", steps, stops


def simulate_trades(df: pd.DataFrame, signal_col: str = "entry_9_21_200",
                    stop_loss=STOP_LOSS, trail_step=TRAIL_STEP,
                    take_profit=TAKE_PROFIT, leverage=LEVERAGE, trail_gap=TRAIL_GAP,
                    fee_rate=FEE_RATE, keep_path=False, last_signal=None,
                    overlap=False) -> pd.DataFrame:
    """signal_col: 1 = buy (long), -1 = sell (short), 0 = nothing.
    Signal is read at the candle close, entry at the next candle open.
    overlap=False: one position at a time (signals while in a trade are skipped).
    overlap=True : every signal candle opens its own trade (docs/entry.md "each 5 min candle").
    keep_path=True adds 'stop_path' (stop per bar) for charting.
    last_signal: stop looking for new signals after this row (faster window tests)."""
    t = df["open_time"].to_numpy()
    o = df["open_5m"].to_numpy(dtype="float64")
    h = df["high_5m"].to_numpy(dtype="float64")
    l = df["low_5m"].to_numpy(dtype="float64")
    c = df["close_5m"].to_numpy(dtype="float64")
    sig = df[signal_col].to_numpy()

    sl_px, step_px, tp_px = stop_loss / leverage, trail_step / leverage, take_profit / leverage
    gap_px = None if trail_gap is None else trail_gap / leverage
    trades = []
    i, n = 0, len(df)
    stop_at = n - 1 if last_signal is None else min(n - 1, last_signal + 1)
    while i < stop_at:
        if not sig[i]:
            i += 1
            continue
        side = 1 if sig[i] > 0 else -1
        e = i + 1
        j, exit_px, reason, steps, stops = _run_trade(o, h, l, e, side, sl_px, step_px, tp_px, gap_px)
        if j is None:                            # still open at end of data
            j, exit_px = n - 1, c[-1]
        entry = o[e]
        px_ret = side * (exit_px / entry - 1)
        row = {
            "side": "long" if side == 1 else "short",
            "signal_idx": i, "entry_idx": e, "exit_idx": j,
            "signal_time": t[i], "entry_time": t[e], "exit_time": t[j],
            "entry": entry, "exit": exit_px, "exit_reason": reason,
            "stop_init": entry * (1 - side * sl_px), "take_profit": entry * (1 + side * tp_px),
            "trail_steps": steps, "bars_held": j - e + 1,
            "price_ret": px_ret, "roe": px_ret * leverage - 2 * fee_rate * leverage,
        }
        if keep_path:
            row["stop_path"] = stops
        trades.append(row)
        i = i + 1 if overlap else j + 1          # next candle, or next candle after exit
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series({"trades": 0})
    r = trades["roe"]
    wins, losses = r[r > 0], r[r <= 0]
    # equity if every trade risks a fixed fraction of 10% of the account
    equity = (1 + 0.10 * r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    return pd.Series({
        "trades": len(r),
        "win_rate": len(wins) / len(r),
        "avg_roe": r.mean(),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "total_roe": r.sum(),
        "equity_10pct_alloc": equity.iat[-1],
        "max_dd_10pct_alloc": dd,
        "avg_bars_held": trades["bars_held"].mean(),
        **{f"exit_{k}": v for k, v in trades["exit_reason"].value_counts().items()},
    })
