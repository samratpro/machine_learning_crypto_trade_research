### How to set stop loss
All distances are in ATR (atr_5m = 14-period ATR of the 5m candles, at the signal candle, frozen for the trade).
Defaults in stop_loss.py = the notebook settings = the ML label exits (docs/ml_train.md):
- Initial stop: 5 x ATR against the entry
- Trail: once the trade is 5 x ATR in profit, the stop follows the best price 5 x ATR behind (never moves back)
- Take profit: 25 x ATR
- Time limit: still open after 288 candles (24 h) -> exit at that candle's close
- Leverage 5: does not move any level; it only scales the ROE (fee 0.055% per side on notional)
