# ML Train Process

Meta-labeling (López de Prado 2018; Joubert 2022): the momentum rule picks the **side**, the model decides **take or skip**.

## 1. Primary model (fixed, not trained)
- Entry = momentum only (`zone_create.entry_signal`), EMA test `13_48_200` (9/21/200 is the robustness check, never mixed)
- Momentum = EMA order + middle EMA not flat (`FLAT_ATR 0.25` over 8 × 15m) + fast EMA slope ≥ `SLOPE_MIN 0.05` ATR per 15m candle in the trend direction
- Signal at the close of a 5m candle, entry at the next open

## 2. Label (frozen before training)
- `stop_loss.simulate_trades` with the defaults in `stop_loss.py`: stop 5 × ATR, trail 5 × ATR step / gap, take profit 25 × ATR, **time limit 288 candles (24 h)**, fee 0.055% per side, leverage 5
- `label = 1` if ROE > 0, else 0. `roe` is kept for profit metrics and position sizing
- Trades still open at the end of the data are dropped
- Changing any of these after looking at results = a new trial (log it)

## 3. Features (`features.py`, 67 columns)
All known at the signal candle's close (checked by `features.check_no_lookahead()`), no raw price levels, **side-relative** (positive = in favour of the trade), continuous.

| group | features |
|---|---|
| trend (15m) | `mid_slope`, `fast_slope`, `spread_fast_mid`, `spread_mid_200`, `dist_fast`, `dist_mid`, `dist_200`, `other_set_momentum`, `touch_mid`, `momentum_age` |
| candle (5m) | `close_pos`, `body`, `range_atr`, `wick_with`, `wick_against` |
| returns | `ret_1`, `ret_3`, `ret_12`, `ret_48`, `ret_288` (in 5m ATR) |
| order flow / volume | `taker_5m`, `taker_15m`, `taker_1h`, `taker_z`, `vol_z`, `trades_z` |
| volatility regime | `atr_pct`, `atr_1h_over_5m`, `atr_z_week` |
| time (UTC) | `hour_sin`, `hour_cos`, `weekday` |
| zones × 4 TF (15m, 1h, 4h, 1d) | `zone_ahead_*`, `zone_behind_*`, `range_pos_*`, `ahead_touches_*`, `ahead_age_*`, `behind_touches_*`, `behind_age_*`, `zones_near_ahead`, `zones_near_behind` |
| round numbers | `round_ahead`, `round_behind` |
| BTC lead | `btc_ret_12`, `btc_ret_48`, `btc_momentum` |

Not used as features: the old 0/1 flags (`near_zone_*`, `confluence`, `zone_long/short`, `*_pullback/rejection/orderflow`) - they are hard-threshold, direction-blind or almost always 0.

## 4. Data split
- **Holdout = 2026-05-20 → end** (`features.HOLDOUT_START`). Untouched until the final run. `run_backtest()` and `momentum_moves()` cut it off by default.
- Dev = everything before; only trades that also **exit** before the holdout.
- CV = `features.walk_forward(dev)`: 5 expanding, time-ordered folds; training trades that exit within 24 h (embargo ≥ time limit) of the test start are purged.
- Sample weight = `weight` (average uniqueness of overlapping trades). With `overlap=True` the ~440k rows are only ≈ 11k effective independent trades.

## 5. Model
- LightGBM first (Gu, Kelly & Xiu 2020), small trees, early stopping on the fold's tail - not on the test block
- Calibrate probabilities (isotonic) on out-of-fold predictions; probability → position size (Meyer et al. 2023)

## 6. Report (per fold, per coin, per year, then once on the holdout)
- Precision, **base rate**, lift, break-even win rate, ROC-AUC, net profit factor, Sharpe, max drawdown
- Baselines on the same trades: take every momentum entry; random take/skip at the same rate; random-side entries; buy & hold
- Trial correction: SPA / Reality Check and PBO over `results/trials.csv`, Deflated Sharpe of the final model
- Cost sensitivity: fee 0.02–0.1% per side; note spot data vs perp trading (funding not included)

## 7. Reproduce
```
python download_data.py      # raw_data/
python data_merge.py         # data/*_master.parquet
python check_merge.py        # merge / look-ahead test
python -c "import features as F; F.check_no_lookahead(); F.build_dataset().to_parquet('data/ml_13_48_200.parquet')"
```
Versions: `requirements.txt`.
