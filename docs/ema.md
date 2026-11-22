### 9 / 21 / 200 - Base on 15 minutes candles
- EMA9 top EMA21 middle and EMA200 bottom market bulish momenterm


### 13 / 48 / 200 - Base on 15 minutes candles
- EMA13 top EMA48 middle and EMA200 bottom market bulish momenterm
- EMA200 top EMA48 middle and EMA13 bottom market bearish momenterm

### No momenterm
- If EMA48 or EMA21 is straight like almost no up down no momenterm
- If EMA48 or EMA21 does not in middle no momenterm

---
### Implemented as (ema_create.py)
- Two separate tests, never mixed: **9/21/200** (middle = EMA21) and **13/48/200** (middle = EMA48)
- **Bullish**: fast > middle > EMA200 (EMA9 > EMA21 > EMA200, or EMA13 > EMA48 > EMA200)
- **Bearish**: EMA200 > middle > fast (EMA200 > EMA21 > EMA9, or EMA200 > EMA48 > EMA13)
  - *Assumption:* the bearish line for 9/21/200 is not written above; it is implemented as the mirror of the bullish line
- **Straight (flat)**: the middle EMA moved less than **0.25 × ATR(15m)** over the last **8 candles of 15m (2 hours)**
  - settings: `FLAT_ATR = 0.25`, `FLAT_BARS = 8` in ema_create.py
- **No momentum**: middle EMA flat, or the EMAs not in bullish / bearish order (middle EMA not in the middle)
- Columns: `momentum_9_21_200`, `momentum_13_48_200` → 1 = bullish, −1 = bearish, 0 = no momentum
- Computed on closed 15m candles only (shift 1 before merging into 5m), so a 5m candle never sees an unfinished 15m candle