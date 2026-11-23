# Guide — What to Avoid in the Research Process

Practical rules for this project, drawn from [paper_analysis.md](paper_analysis.md) and [research_gap.md](research_gap.md). Ordered by how badly each mistake would mislead you.

> **One-line rule:** decide your test data, costs and evaluation metric *before* you look at results, and never change them after.

---

## 1. Don't treat overlapping trades as independent

With `overlap=True`, a signal on every 5m candle opens its own trade. Neighbouring trades share almost the same price path, so ~414k trades are really far fewer independent bets.

- ❌ Random `train_test_split` or plain K-fold: neighbouring candles land in both train and test, so the model scores well on data it has effectively already seen.
- ✅ Time-ordered **walk-forward** splits with an **embargo** (gap) at least as long as the longest trade (`bars_held`).
- ✅ Weight samples by uniqueness and use a block bootstrap for confidence intervals.

## 2. Don't tune on the data you report

Each tweak to `FLAT_ATR`, slope, `TRAIL_GAP`, the EMA set or the thresholds, checked against the same 3 years, makes the result look better than it really is.

- ❌ Keep the "best" setting and report its numbers as the result.
- ✅ Set aside the **last ~6 months** as a final holdout now. Don't touch it until the very end.
- ✅ Log **every** configuration you try (CSV: settings + metrics), so you can correct for the number of trials later (SPA test, Deflated Sharpe, probability of backtest overfitting).

## 3. Don't judge precision without the base rate

46.6% precision only means something next to the 44.9% that a random candle wins.

- ❌ Report "56% precision" alone.
- ✅ Always show together: **precision, base rate, lift, break-even win rate** (49.4% now), on the same data and settings.

## 4. Don't leak future information

- ✅ Never let a 5m row see a 15m / 1h / 4h / 1d bar that hasn't closed. Keep the `shift(1)` before merging, and **add a test that proves it**.
- ❌ Compute features over the whole dataset before splitting, for example normalisation or z-scores fitted on all data, or averages that include the current or future bars.
- ❌ Use any outcome column (`roe`, `exit_reason`, `bars_held`, `exit_idx`) as a feature.

## 5. Don't use unrealistic costs or the wrong venue

- ❌ Backtest on Binance **spot** prices and then trade 10× **perps** on Bybit.
- ✅ Use perp klines, and add **funding**, **slippage** and a check on **liquidation distance** at 10×.
- ❌ Assume taker fills at the exact next open. Add a slippage term and test cost sensitivity (0.02–0.1% per side).

## 6. Don't bake hard thresholds into features

- ❌ Feed the model only 0/1 flags like `taker_buy > 0.52`, "top 35% close" or `NEAR_ATR 0.5`.
- ✅ Give it the **continuous values** (taker ratio, close position, ATR distances, zone touch count, zone age). The hand-picked cut-offs throw information away and count as more tuning.

## 7. Don't use the same % stop for every coin

A −2% price stop means something different for BTC than for SOL or XRP.

- ✅ Also try **ATR- or volatility-scaled** stop / trail / take-profit barriers.
- ✅ Add a **time limit** (max holding period) to the trade label.

## 8. Don't trust pooled results

- ❌ Rely on one number pooled across 5 coins and 3 years.
- ✅ Report results **per coin and per year**. Edges often come from one period and fade later.
- ✅ The 5 coins move together, so don't count them as 5 independent tests. Resample by timestamp and also report BTC alone.

## 9. Don't expect ML to rescue a rule with no edge

- ❌ Jump to LSTMs or complex models first.
- ✅ Start with **LightGBM / XGBoost as a meta-labeling model**: train only on candles where the EMA rule fires, and predict P(win).
- ✅ If it can't beat the base rate on the out-of-sample data, a more complex model won't either.
- ✅ Calibrate probabilities, then use them for position sizing instead of a fixed 10×.

## 10. Don't keep secrets or live orders in code

- ❌ API keys in `config.py`, or anywhere in git.
- ❌ A module that places real orders when it is imported.
- ✅ Revoke the current key, load credentials from environment variables, add the config file to `.gitignore`, and remove the key from git history before pushing.

---

## Checklist before reporting any result

- [ ] Final holdout untouched until now
- [ ] Walk-forward CV with an embargo; no random splits
- [ ] HTF look-ahead test passes
- [ ] No outcome columns among the features
- [ ] Perp prices, fees, funding and slippage included
- [ ] Precision shown with base rate, lift and break-even win rate
- [ ] Results per coin and per year
- [ ] All configurations tried are logged; multiple-testing correction reported
- [ ] Cost-sensitivity table included
