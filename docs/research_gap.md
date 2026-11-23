# Research Gap — Where This Project Can Contribute

Companion to [paper_analysis.md](paper_analysis.md). Part A lists gaps in the **published literature** that this project can fill. Part B lists gaps in **this project's own method** that a reviewer would flag, each with a fix. Part C proposes the research questions and paper framing.

---

## Part A — Gaps in the literature

### A1. TA-in-crypto evidence is mostly daily, not 5-minute with higher-timeframe context
- Hudson & Urquhart (2021), Detzel et al. (2021) and Liu & Tsyvinski (2021) all work at **daily or weekly** frequency.
- Corbet et al. (2019) and Shen et al. (2022) go intraday, but only for **BTC** and with **single-timeframe** rules.
- **Gap:** no published study tests a *multi-timeframe* design across several major coins. A design of that kind takes a 15m trend filter, 15m/1h/4h/1d S/R zones and a 5m execution trigger.

### A2. Studies predict direction, not the outcome of a specific exit policy
- Most crypto ML papers (Jaquart et al. 2021; Akyildirim et al. 2023; arXiv:2602.10785) predict **next-bar or next-horizon direction**, then bolt on a simple trading rule.
- **Gap:** few papers label each candle by the realised outcome of a **path-dependent leveraged exit**: stop, trailing stop and take-profit at 10×. Fewer still study how ML precision changes **across exit policies**, which your `sliding.py` grid already makes possible.

### A3. Meta-labeling has little empirical crypto evidence
- Meta-labeling is well formalised (López de Prado 2018; Joubert 2022; Meyer et al. 2023), but published evidence is mostly synthetic or equities.
- **Gap:** an out-of-sample test of meta-labeling on top of a **transparent, rule-based crypto primary model**, with the lift reported over both the primary rule *and* the random-candle base rate.

### A4. S/R zones are rarely modelled as multi-timeframe *bands* with strength features
- Osler (2000, 2003) uses published bank levels. Chung & Bellotti (2021) use single-series levels and show that bounce probability **rises with prior touches and decays with age**.
- Corbet et al. (2019) use simple range breakouts.
- **Gap:** no study combines **ATR-width bands from 4 timeframes**, a **confluence score**, and **touch-count/age** features as ML inputs in crypto.

### A5. Kline taker-flow vs continuation or reversal is unresolved
- Cont et al. (2014) show OFI explains *contemporaneous* moves. Kline taker-ratio studies (arXiv:2608.21888) find small next-bar power and even **mean reversion** after flow-driven bars.
- **Gap:** whether aggressive taker flow *at a trend pullback into a zone* signals continuation (your hypothesis) or exhaustion. That is a conditional question nobody has answered.

### A6. Stop-loss theory is untested intraday in leveraged crypto
- Kaminski & Lo (2014) predict that stops add value **only under momentum**. Han, Zhou & Zhu test this only on monthly equity momentum.
- **Gap:** an empirical test of Kaminski–Lo at **5m in leveraged crypto**. Compare stop/trail value on *momentum-filtered* candles vs *random* candles. Your per-candle `long_roe` / `short_roe` arrays already make this test cheap.

### A7. Multiple-testing correction is rare in practitioner-style crypto rule research
- Park & Irwin (2007) found that data snooping is the main weakness of TA studies. Few crypto intraday papers report SPA, PBO or Deflated Sharpe alongside results.
- **Gap:** a fully trial-corrected report of a multi-parameter crypto rule system.

---

## Part B — Gaps in this project's current method (fix before ML)

| # | Gap in the repo | Why it matters | Fix |
|---|---|---|---|
| B1 | **Lift is tiny:** precision 46.6% vs base 44.9% (lift 1.04) and PF 0.89 in the saved notebook run. The "56%" in the last commit isn't documented. | The primary rule barely beats a random candle, so ML has to supply almost all of the edge. | Log every configuration and its metrics in a results table (CSV). Record the exact settings behind the 56% figure. |
| B2 | **Overlapping trades** (`overlap=True`): about 414k trades that share most of their price path | Effective sample size is far smaller than it looks. Standard errors are overstated, and K-fold CV leaks across folds. | Purged and embargoed walk-forward CV (embargo ≥ max `bars_held`), sample-uniqueness weights, block bootstrap for confidence intervals |
| B3 | **No time barrier** in the label (a trade may run until end of data) | Labels mix 10-minute and multi-day trades, and the label horizon is unbounded, which makes purging hard | Add a max holding period (vertical barrier) to `_run_trade` |
| B4 | **Fixed % barriers** (−2% / +5% price) for all coins | BTC and SOL or XRP have very different volatility, so the same stop is tight for one coin and loose for another | ATR-scaled or realised-volatility-scaled barriers. Compare both. |
| B5 | **Data/venue mismatch:** Binance **spot** klines, but the plan is to trade 10× perps and `config.py` targets **Bybit** | Spot ≠ perp price. Perps add **funding**, liquidation and slippage. | Use Binance or Bybit **perp** klines, add funding-rate cost, add a slippage term, and check liquidation distance at 10× |
| B6 | **Hard thresholds** (`TAKER_BUY_RATIO 0.52`, top 35% close, `NEAR_ATR 0.5`) | Hand-tuned cut-offs are both snooping and information loss | Feed the continuous values to the model and let the trees find the split points |
| B7 | **Missing features** the literature says matter | Shen et al.: vol/volume regime. Osler: round numbers. Chung & Bellotti: zone touches and age. | Add hour-of-day / session, realised-vol regime, volume z-score, distance to round number, zone touch count, zone age, BTC return as a lead feature for alt coins |
| B8 | **Parameter search not tracked** | Reality Check, SPA and PBO all need the full set of trials | Record every configuration tried (EMA set, flat/slope settings, trail gap, ...) and report SPA p-values and PBO |
| B9 | **No year-by-year or coin-by-coin out-of-sample report** | Fischer & Krauss showed edges vanish over time. Pooled numbers can hide a single good year. | Walk-forward by quarter. Show metrics per year × coin. |
| B10 | **Cross-coin dependence:** 5 coins that move together are pooled as independent | Overstates evidence | Cluster or block resampling by timestamp. Also report a BTC-only result. |
| B11 | **Look-ahead risk in HTF merge** (the bug that cost the 2602.10785 authors 0.20 AUC) | Silent inflation of results | Add a test that the 5m row never sees an unfinished 15m/1h/4h/1d bar. The same applies to the 1d zone (`shift(1)` on daily). |
| B12 | **Fixed leverage, no sizing** | Discards model confidence | Calibrated probability → fractional sizing (Meyer et al. 2023). Compare with fixed size. |

---

## Part C — Proposed research questions and framing

**Working title:** *"Meta-Labeling a Multi-Timeframe Trend-Pullback Rule in Leveraged Cryptocurrency Markets: Evidence at 5-Minute Frequency"*

**RQ1 (primary rule):** Does a 15m EMA-ribbon momentum filter raise the probability that a 5m leveraged trade under a fixed stop/trail/TP policy ends profitable, relative to the unconditional base rate, after costs and multiple-testing correction? *(fills A1, A7)*

**RQ2 (meta-labeling):** Does a GBDT meta-model trained on multi-timeframe zone, candle, order-flow and regime features raise precision and net profit factor out of sample, beyond the primary rule? *(fills A3, A4)*

**RQ3 (order flow):** Conditional on a trend pullback into a zone, does high taker-buy (or sell) imbalance predict continuation or exhaustion? *(fills A5)*

**RQ4 (exits):** Do stop-loss and trailing-stop rules add more value on momentum-filtered candles than on random candles, as Kaminski–Lo predict? How does the answer change with ATR-scaled barriers? *(fills A2, A6)*

**Hypotheses:**
- H1: precision lift of the primary rule > 1 but small, and it does not survive SPA alone.
- H2: meta-labeling raises lift and turns PF > 1 net of fees and funding.
- H3: taker-flow effect is non-monotone (Explainable-Patterns-style concavity).
- H4: trailing stops add value only in the momentum-filtered subset.

**Evaluation protocol:**
1. Purged, embargoed walk-forward CV.
2. Hold out the **last 6 months**, untouched until the end.
3. Report precision lift, ROC-AUC, net PF, Sharpe, DSR and PBO, per coin and per year.
4. Run an SPA test over every configuration tried.
5. Run cost sensitivity at 0.02–0.1% per side, plus funding.

**Expected contribution:** the first multi-coin, 5m, multi-timeframe test of a rule-plus-meta-label pipeline in crypto. Every result would be reported net of perp costs and corrected for data snooping. That fills the methodological weaknesses Park & Irwin (2007) identified in technical-analysis research.
