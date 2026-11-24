# Zone draw rules
- For zone draw 15 minutes, 1 hours, 4hours, 1day candle should be follow
- But entry will follow 5 minute candle

### Build Zones as Bands (Not Single Lines):

- 15m: Rolling 20-bar High/Low (± 0.25 ATR) for active intraday levels.
- 1h: Rolling 24-bar High/Low (± 0.25 ATR) for main session structure.
- 4h: Rolling 18-bar High/Low (± 0.25 ATR) for multi-day swing zones.
- 1D: Rolling 14-bar High/Low (± 0.25 ATR) for major macro boundaries.
- Always compute zones on each timeframe first, then merge into 5m by close time (a 5m candle sees an HTF candle only once it has closed).

### Normalize Zones for ML:

- Convert raw prices into distances: (Close_5m - Zone_Level) / ATR_5m.
- Add a Confluence Score (0 to 4) counting how many timeframe zones overlap near current price.

### Best 5m Entry Trigger (3-Step Confirmation):

- Trend Filter: 15m EMAs aligned (9 > 21 > 200 and/or 13 > 48 > 200) + price pulls back near a 15m/1h Support Zone or 15m EMA21.
- Candle Rejection: 5m low pierces the zone, closes in the top 35% of the bar (pin bar/hammer), or closes above the prior 5m candle's high.
- Order Flow: Aggressive buyer volume (Taker_buy_base / Volume > 0.52) with above-average volume.

