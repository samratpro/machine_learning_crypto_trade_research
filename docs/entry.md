### When entry
- Entry will take each 5 min candle completed
- If momentum is bullish then long entry
- If momentum is bearish then short entry
- If no momentum then no entry
- Momentum includes the fast EMA slope: EMA9 / EMA13 must move at least SLOPE_MIN ATR per 15m candle in the trend direction (SLOPE_MIN > 0), otherwise no momentum
- Nothing else decides the entry (no distance / EMA touch filter)
- When EMA9 test it will consider EMA9 EMA21 and EMA200 (Never apply both ema same time)
- When EMA13 test it will consider EMA13 EMA48 and EMA200 (Never apply both ema same time)
- Close distance to the fast EMA, middle EMA touch and zones are kept as ML features only (training stage)

### Momentum move test (analysis/momentum_move.py)
- No stop loss / take profit: hold while the momentum lasts, exit at the open of the first candle without it
- Measure the favourable move (MFE), adverse move (MAE) and exit move in ATR (atr_5m of the signal candle)
