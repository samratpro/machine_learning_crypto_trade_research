"""
Stop-loss / trailing / take-profit simulation (docs/stop_loss.md).

Every distance is measured in ATR (atr_5m = 14-period ATR of the 5m candles, value at the signal candle,
which is already closed when the trade opens -> no look-ahead). The ATR is frozen for the whole trade.

  - Initial stop            -> STOP_LOSS x ATR against the entry
  - Trail                   -> once the best price is TRAIL_STEP x ATR in profit, the stop follows the best
                               price TRAIL_GAP x ATR behind it and never moves back
                               TRAIL_GAP = None -> same as TRAIL_STEP (first trail = break-even)
  - Take profit             -> TAKE_PROFIT x ATR in favour of the entry
  - Time limit              -> still open after MAX_BARS candles -> exit at that candle's close ('time')

So BTC (small ATR) gets a tighter stop in price % than SOL (big ATR), but the same stop in "normal moves".
LEVERAGE does not move any level any more; it only turns the price move into position PnL (ROE).

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

# all in ATR units (x atr_5m at the signal candle)
STOP_LOSS = 5.0
TRAIL_STEP = 5.0
TAKE_PROFIT = 25.0
TRAIL_GAP = None          # how far the stop follows behind the best price (ATR); None = TRAIL_STEP
MAX_BARS = 288            # time limit: 288 x 5m = 24 h (95% of trades end before it); None = no limit
ATR_COL = "atr_5m"
LEVERAGE = 5
FEE_RATE = 0.00055        # per side, on notional (Bybit taker)


def _run_trade(o, h, l, e, side, sl_d, step_d, tp_d, gap_d=None, max_bars=MAX_BARS, c=None):
    """Walk one trade from bar e. side = 1 long, -1 short.
    sl_d / step_d / tp_d / gap_d are PRICE distances (k x ATR).
    Returns (exit_idx, exit_px, reason, steps, stops); stops[k] = stop active during bar e+k."""
    n = len(o)
    gap_d = step_d if gap_d is None else gap_d
    entry = o[e]
    stop = entry - side * sl_d
    tp = entry + side * tp_d
    steps, stops = 0, []
    end = n if max_bars is None else min(n, e + max_bars)
    j = e
    while j < end:
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
        gain = side * (best - entry)                     # best price move this bar (price units)
        steps = max(steps, int(np.floor(gain / step_d + 1e-9)))
        if gain >= step_d * (1 - 1e-9):                  # in profit by one step -> follow the price
            trail = best - side * gap_d                  # gap_d behind the best price
            stop = max(stop, trail) if side == 1 else min(stop, trail)
        j += 1
    if end < n:                                          # time limit reached -> exit at the last candle's close
        return end - 1, c[end - 1], "time", steps, stops
    return None, None, "open_eod", steps, stops


def simulate_trades(df: pd.DataFrame, signal_col: str = "entry_9_21_200",
                    stop_loss=STOP_LOSS, trail_step=TRAIL_STEP,
                    take_profit=TAKE_PROFIT, leverage=LEVERAGE, trail_gap=TRAIL_GAP,
                    fee_rate=FEE_RATE, keep_path=False, last_signal=None,
                    overlap=False, atr_col=ATR_COL, max_bars=MAX_BARS) -> pd.DataFrame:
    """signal_col: 1 = long, -1 = short, 0 = nothing.
    stop_loss / trail_step / take_profit / trail_gap: ATR multiples (atr_col at the signal candle).
    max_bars: time limit in 5m candles (None = hold until stop / TP / end of data).
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
    atr = df[atr_col].to_numpy(dtype="float64")
    sig = df[signal_col].to_numpy()

    trades = []
    i, n = 0, len(df)
    stop_at = n - 1 if last_signal is None else min(n - 1, last_signal + 1)
    while i < stop_at:
        if not sig[i] or not atr[i] > 0:
            i += 1
            continue
        side = 1 if sig[i] > 0 else -1
        e = i + 1
        a = atr[i]
        gap_d = None if trail_gap is None else trail_gap * a
        j, exit_px, reason, steps, stops = _run_trade(o, h, l, e, side, stop_loss * a, trail_step * a,
                                                      take_profit * a, gap_d, max_bars, c)
        if j is None:                            # still open at end of data
            j, exit_px = n - 1, c[-1]
        entry = o[e]
        px_ret = side * (exit_px / entry - 1)
        row = {
            "side": "long" if side == 1 else "short",
            "signal_idx": i, "entry_idx": e, "exit_idx": j,
            "signal_time": t[i], "entry_time": t[e], "exit_time": t[j],
            "entry": entry, "exit": exit_px, "exit_reason": reason, "atr": a,
            "stop_init": entry - side * stop_loss * a, "take_profit": entry + side * take_profit * a,
            "trail_steps": steps, "bars_held": j - e + 1,
            "price_ret": px_ret, "atr_move": side * (exit_px - entry) / a,
            "roe": px_ret * leverage - 2 * fee_rate * leverage,
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
