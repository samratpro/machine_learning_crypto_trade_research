# Paper Analysis — Literature Behind This Project

This file maps published research to each part of the current strategy and the planned ML step (`docs/ml_train.md`).

## 0. What this project is, in one paragraph

- **Universe and data:** 5 Binance spot pairs (BTC, ETH, SOL, BNB, XRP / USDT), 3 years, 5m bars with 15m / 1h / 4h / 1d context.
- **Trend filter:** a 15m EMA-ribbon momentum filter, either 9/21/200 or 13/48/200. The middle EMA must not be flat (< 0.25 ATR over 8 bars) and the fast EMA's slope must point the same way.
- **Entry:** taken on the 5m bar, and blocked if the bar touches the middle EMA.
- **Zone features:** multi-timeframe rolling high/low support/resistance bands (± 0.25 ATR), with ATR-normalised distances and a 0–4 confluence score.
- **Candle and order flow:** candle-rejection features plus kline taker-buy ratio.
- **Exits:** 10× leverage, −20% ROE stop, trailing stop after +10%, +50% take-profit, 0.055% taker fee per side.
- **Current result:** in the saved notebook run (all 5 coins) the entry has **46.6% precision vs a 44.9% base rate (lift 1.04)**, PF 0.89, against a break-even win rate of 49.4%. The latest commit message says "56% precision by filter feature only". The notebook output does not show the configuration behind that number.
- **Next step:** train an ML model on top of these rules.

Each section below lists a paper, what it found, and **what it means for this repo**. Verification level: ✅ = bibliographic data and abstract checked online; ◐ = preprint / not peer-reviewed; findings are taken from abstracts and summaries, not full-text reading.

---

## 1. Does technical analysis work at all?

### 1.1 Brock, Lakonishok & LeBaron (1992) ✅
*"Simple Technical Trading Rules and the Stochastic Properties of Stock Returns", Journal of Finance 47(5), 1731–1764.*
- Tested moving-average and trading-range-break rules on the Dow Jones, 1897–1986. Rule returns were inconsistent with a random walk, AR(1), GARCH-M and EGARCH.
- **Relevance:** your two building blocks are (a) an MA rule (the EMA ribbon) and (b) a trading-range rule (rolling N-bar high/low = Donchian / range-break zones). BLL is the canonical citation for both. It also introduced the **bootstrap-under-null-model** test you can reuse: simulate GARCH paths and check whether your rule's edge beats them.

### 1.2 Lo, Mamaysky & Wang (2000) ✅
*"Foundations of Technical Analysis", Journal of Finance 55(4), 1705–1770.*
- Automated chart-pattern recognition with kernel regression and found that patterns carry *incremental* information.
- **Relevance:** this is the template for turning discretionary rules (your "zone", "rejection candle") into precise algorithms and then testing them statistically. Your `zone_create.py` is already this kind of formalisation.

### 1.3 Park & Irwin (2007) ✅
*"What Do We Know About the Profitability of Technical Analysis?", Journal of Economic Surveys 21(4), 786–826.*
- Of 95 modern studies, 56 were positive, 20 negative and 19 mixed. Most suffer from **data snooping, ex-post rule selection, and poor treatment of risk and transaction costs**.
- **Relevance:** this is the reviewer checklist your paper will be judged against (see §6).

### 1.4 Sullivan, Timmermann & White (1999), White (2000), Hansen (2005) ✅
*Journal of Finance 54(5); Econometrica 68(5); JBES 23(4).*
- **White's Reality Check** and Hansen's **SPA test** correct for the fact that many rules or parameter sets were tried. On ~100 years of DJIA data, the best rule's edge weakened after adjustment and in later samples.
- **Relevance:** you have already varied `FLAT_ATR`, `FLAT_BARS`, `FAST_SLOPE_MIN`, the EMA sets, `TRAIL_GAP`, leverage and others. Any final result must be reported with SPA / Reality-Check p-values over **all** configurations tried.

---

## 2. Technical analysis in crypto specifically

### 2.1 Hudson & Urquhart (2021) ✅
*"Technical trading and cryptocurrencies", Annals of Operations Research 297, 191–220.*
- Tested ~15,000 rules across 5 rule classes and found significant predictability for every class in every coin, with many rules surviving multiple-testing correction. The data is **daily**.
- **Relevance:** this is the strongest support for using MA and range rules in crypto. Your 5m horizon is outside their sample, which is a gap you can claim (see research_gap.md).

### 2.2 Corbet, Eraslan, Lucey & Sensoy (2019) ✅
*"The effectiveness of technical trading rules in cryptocurrency markets", Finance Research Letters 31, 32–37.*
- Tested MA-oscillator and trading-range-breakout (support/resistance) rules on **high-frequency** Bitcoin data.
- **Relevance:** this is the closest published precedent to your 5m MA + S/R design. Cite it as the direct predecessor.

### 2.3 Detzel, Liu, Strauss, Zhou & Zhu (2021) ✅
*"Learning and predictability via technical analysis: Evidence from bitcoin and stocks with hard-to-value fundamentals", Financial Management 50(1), 107–137.*
- An equilibrium model in which **TA arises rationally** when fundamentals are hard to value. Empirically, MA ratios predict Bitcoin returns in and out of sample.
- **Relevance:** this gives a *theoretical* justification for an MA-based filter in crypto, which is useful in your paper's motivation section.

### 2.4 Liu & Tsyvinski (2021) ✅
*"Risks and Returns of Cryptocurrency", Review of Financial Studies 34(6), 2689–2727.*
- Crypto returns are not explained by traditional factors. Strong **time-series momentum** and investor-attention effects are.
- **Relevance:** this is the macro-level justification for a *trend-following* (momentum-aligned) entry, the core of `ema_create.py`.

### 2.5 Shen, Urquhart & Wang (2022) ✅
*"Bitcoin intraday time-series momentum", Financial Review 57(2), 319–344.*
- Intraday momentum exists in BTC and is **strongest in high-volume / high-volatility periods**. It is driven by liquidity provision.
- **Relevance:** this points directly to the missing **volume/volatility regime and time-of-day features** in your feature set. Your filter currently treats every hour of the day and every volatility state the same way.

---

## 3. Support / resistance zones

### 3.1 Osler (2000; 2003) ✅
*"Support for Resistance: Technical Analysis and Intraday Exchange Rates", FRBNY Economic Policy Review 6(2); "Currency Orders and Exchange Rate Dynamics", Journal of Finance 58(5), 1791–1819.*
- Published S/R levels predicted intraday trend interruptions. The mechanism is **order clustering**: take-profit orders sit at round numbers (reversal), and stop-losses sit just beyond them (acceleration through the level).
- **Relevance:** your zones use rolling highs/lows only. Osler suggests adding **distance to round numbers** and treating a *break through a zone* (stop cascade) differently from a *touch* (reversal).

### 3.2 Chung & Bellotti (2021) ◐
*"Evidence and Behaviour of Support and Resistance Levels in Financial Time Series", arXiv:2101.07410.*
- Algorithmically discovered levels reverse price in a statistically significant way. **Levels with more prior bounces are more likely to bounce again, and bounce probability decays with time.**
- **Relevance:** this suggests three features your zones do not have yet: **touch count, zone age, and bars since last touch**. Your current `confluence` counts timeframes, not zone strength.

### 3.3 Chan et al. (2022), Mathematics 10(20):3888 ◐
- Adding S/R features to ML trading models increased aggregate profit across 8 FX pairs (reported +65%).
- **Relevance:** this is supporting evidence that zone features belong *in the ML model*, not as hard filters. That matches your design choice that `zone_long` / `zone_short` are features only.

---

## 4. Candlestick / rejection patterns

### 4.1 Marshall, Young & Rose (2006) ✅
*"Candlestick technical trading strategies: Can they create value for investors?", Journal of Banking & Finance 30(8), 2303–2323.*
- Found no value in candlestick patterns on the DJIA after a bootstrap test. Later work (Lu, Chen & Hsu 2015) found that profitability depends heavily on the **exit rule**.
- **Relevance:** expect `long_rejection` / `short_rejection` to be weak alone. Their value, if any, will show up as an *interaction* with zone proximity, which tree models capture well.

---

## 5. Order flow (taker-buy ratio)

### 5.1 Cont, Kukanov & Stoikov (2014) ✅
*"The Price Impact of Order Book Events", Journal of Financial Econometrics 12(1), 47–88.*
- Order-flow imbalance (OFI) linearly explains **contemporaneous** short-interval price changes, with slope inversely proportional to depth.
- **Relevance:** your `taker_buy_ratio_5m` is a crude, kline-level proxy for OFI. The CKS result is about *same-bar* impact, so predictive power for the *next* bars should be expected to be small.

### 5.2 Kline taker-flow studies in crypto ◐
- *"Short-horizon mean reversion in cryptocurrency markets" (arXiv:2608.21888)* uses Binance kline taker-buy volume exactly as you do and reports a next-bar sign AUC of only **~0.54**, with mean reversion *after* flow-driven bars.
- *"Explainable Patterns in Cryptocurrency Microstructure" (arXiv:2602.00776)* uses GBDT on 5 coins and finds OFI effects that are monotone and **concave at extremes**.
- **Relevance:** use the taker ratio as a **continuous feature**, possibly with its rolling z-score, instead of the hard `> 0.52` threshold. Also test the *sign* of its effect. Strong taker buying can precede mean reversion, not continuation.

---

## 6. ML on top of rules: labeling, meta-labeling and validation

### 6.1 López de Prado (2018), *Advances in Financial Machine Learning*, Wiley
- **Triple-barrier labeling:** each label is whichever barrier is hit first (profit-take, stop-loss or time limit). Barriers are scaled by volatility.
- **Meta-labeling:** a primary model gives the side, and a secondary ML model predicts *whether to take the trade* and how large to make it.
- **Purged K-fold CV with embargo**, **sample uniqueness weights** for overlapping labels, and **feature importance (MDI/MDA/SFI)**.
- **Relevance:** this is the most important reference for `ml_train.md`.
  - Your `sliding._outcomes()` already computes a triple-barrier-style label (stop / trail / TP), but **without the time barrier** and with **fixed % barriers rather than volatility-scaled ones**.
  - Your EMA entry rule *is* a primary model, so the ML step is literally meta-labeling. The target is `1 if roe > 0`, predicted only on candles where `entry_* != 0`.
  - With `overlap=True`, consecutive trades share most of their price path, so labels are **heavily overlapping**. Plain K-fold or `train_test_split` will leak. You need purging, an embargo and uniqueness weights.

### 6.2 Joubert (2022); Meyer, Barziy & Joubert (2023) ✅
*"Meta-Labeling: Theory and Framework", Journal of Financial Data Science 4(3), 31–44; "Meta-Labeling: Calibration and Position Sizing", JFDS 5(2), 23–40.*
- Formalises meta-labeling. Calibrated probabilities should drive **position size**, not just a take/skip decision.
- **Relevance:** after training, calibrate (isotonic/Platt) and size each trade by its probability. With a fixed 10× leverage you currently throw that information away.

### 6.3 Bailey, Borwein, López de Prado & Zhu (2017); Bailey & López de Prado (2014) ✅
*"The Probability of Backtest Overfitting", Journal of Computational Finance; "The Deflated Sharpe Ratio", Journal of Portfolio Management.*
- **CSCV/PBO** estimates how likely the best in-sample configuration is to underperform out of sample. The **Deflated Sharpe Ratio** corrects for the number of trials and for non-normal returns.
- **Relevance:** report PBO across all your parameter grids and the DSR of the final strategy. A short-history, multi-tweak project like this one is exactly the case these tools were built for.

### 6.4 Gu, Kelly & Xiu (2020) ✅
*"Empirical Asset Pricing via Machine Learning", Review of Financial Studies 33(5), 2223–2273.*
- Trees and neural nets win because they capture **nonlinear interactions**. The dominant predictors are momentum, liquidity and volatility variants.
- **Relevance:** this justifies GBDT (LightGBM/XGBoost) as the first model. Your features (trend × zone × flow) are interaction-heavy.

### 6.5 Fischer & Krauss (2018) ✅
*"Deep learning with LSTM networks for financial market predictions", EJOR 270(2), 654–669.*
- LSTM on the S&P 500 was profitable from 1992 to 2009, but **excess returns went to roughly zero after costs from 2010**.
- **Relevance:** this is a warning that edges decay. Report results **per year** and per coin, not only pooled.

### 6.6 Crypto ML at intraday frequency
- **Jaquart, Dann & Weinhardt (2021)**, *Journal of Finance and Data Science* ✅: all ML models beat random on short-horizon BTC direction, and **predictability rises with horizon**.
- **Akyildirim, Cepni, Corbet & Uddin (2023)**, *Annals of OR* 330 ✅: six ML algorithms on BTC futures at 5–60 min horizons.
- **Multi-timeframe GBDT on BTC (arXiv:2602.10785)** ◐: gross +36% OOS became **negligible after costs** at AUC ≈ 0.60. A **look-ahead bug in higher-timeframe alignment inflated AUC by ~0.20** before it was fixed.
- **Islas Anguiano & García-Medina (2025), arXiv:2511.00665** ◐: LightGBM vs LSTM vs TA on BTC. High trading frequency eroded LightGBM's returns after costs.
- **Relevance:**
  - Your `shift(1)` before merging HTF data is exactly the fix that paper needed. Keep a unit test for it, for example by asserting that `close_15m` at a 5m row equals the *previous closed* 15m close.
  - Expect a realistic AUC of 0.55–0.60. With TP/SL asymmetry and 1.1% round-trip ROE cost at 10×, check the precision you need *before* training (break-even ≈ 49.4% today).

---

## 7. Stop-loss and trailing exits

### 7.1 Kaminski & Lo (2014) ✅
*"When do stop-loss rules stop losses?", Journal of Financial Markets 18, 234–254.*
- Under a random walk, stop-loss rules **always reduce** expected return. They add value only when returns show **momentum / serial correlation**.
- **Relevance:** your stop and trail only help if the EMA filter really selects momentum regimes. Test the exit policy *conditionally*: momentum-filtered candles vs random candles. Lo & Remorov (2017) extend the analysis to serial correlation.

### 7.2 Han, Zhou & Zhu (2016) ✅
*"Taming Momentum Crashes: A Simple Stop-Loss Strategy" (SSRN / working paper).*
- A 10% stop cut worst monthly momentum losses from −49.8% to about −11%. Results are **gross of costs**, and stop-heavy trading raises turnover.
- **Relevance:** this supports your stop design, but your fixed −2% price stop is the same for BTC and SOL or XRP, whose volatility differs a lot. The literature (and López de Prado) points to **volatility-scaled barriers** (k × ATR).

---

## 8. Regimes

- **HMM / regime models for crypto** (e.g., Koki, Leonardos & Piliouras, arXiv:1909.10957 ◐; sparse-jump-model feature selection, *Digital Finance* 2024 ◐): trend and reversal features matter *per regime*, and 3–4 state models fit crypto best.
- **Relevance:** your "flat middle EMA = no momentum" rule is a hand-built two-state regime filter. A volatility/trend regime label is a natural extra ML feature, or a way to split the data for evaluation.

---

## 9. Takeaways for `ml_train.md`

| # | Recommendation | Source |
|---|---|---|
| 1 | Frame ML as **meta-labeling**: the EMA rule gives the side and the model gives P(win). Train only on candles where a signal fires. | López de Prado 2018; Joubert 2022 |
| 2 | Label = existing stop/trail/TP outcome, **plus a time barrier**. Try ATR-scaled barriers. | López de Prado 2018; Kaminski & Lo 2014 |
| 3 | **Purged + embargoed walk-forward CV**, with sample-uniqueness weights because trades overlap | López de Prado 2018 |
| 4 | Model: **LightGBM/XGBoost** first; NN only if GBDT plateaus | Gu, Kelly & Xiu 2020 |
| 5 | Add **zone strength** (touch count, age), **round-number distance**, **time of day**, **volatility/volume regime**, and a **continuous taker-flow z-score** | Chung & Bellotti; Osler; Shen et al. |
| 6 | Report **precision lift over base rate**, net PF, and **per-year / per-coin** results | Fischer & Krauss 2018 |
| 7 | Correct for trials: **SPA test, PBO, Deflated Sharpe** | Hansen 2005; Bailey et al. |
| 8 | Calibrate probabilities and use them for **position sizing** | Meyer, Barziy & Joubert 2023 |
| 9 | Keep the HTF `shift(1)` and **test it** | arXiv:2602.10785 |

---

## References (links)

- Brock, Lakonishok, LeBaron (1992) — https://ideas.repec.org/a/bla/jfinan/v47y1992i5p1731-64.html
- Lo, Mamaysky, Wang (2000) — https://www.nber.org/papers/w7613
- Park & Irwin (2007) — https://ideas.repec.org/a/bla/jecsur/v21y2007i4p786-826.html
- Sullivan, Timmermann, White (1999) — https://bashtage.github.io/kevinsheppard.com/files/teaching/mfe/advanced-econometrics/Sullivan_Timmermann_White.pdf
- White (2000) Reality Check — https://users.ssc.wisc.edu/~bhansen/718/White2000.pdf
- Hudson & Urquhart (2021) — https://link.springer.com/article/10.1007/s10479-019-03357-1
- Corbet et al. (2019) — https://econpapers.repec.org/RePEc:eee:finlet:v:31:y:2019:i:c:p:32-37
- Detzel et al. (2021) — https://ideas.repec.org/a/bla/finmgt/v50y2021i1p107-137.html
- Liu & Tsyvinski (2021) — https://www.nber.org/papers/w24877
- Shen, Urquhart, Wang (2022) — https://centaur.reading.ac.uk/100181
- Osler (2000) — https://www.newyorkfed.org/research/epr/00v06n2/0007osle ; Osler (2003) — https://econpapers.repec.org/RePEc:bla:jfinan:v:58:y:2003:i:5:p:1791-1819
- Chung & Bellotti (2021) — https://arxiv.org/abs/2101.07410
- Chan et al. (2022) — https://ideas.repec.org/a/gam/jmathe/v10y2022i20p3888-d947666.html
- Marshall, Young, Rose (2006) — https://econpapers.repec.org/RePEc:eee:jbfina:v:30:y:2006:i:8:p:2303-2323
- Cont, Kukanov, Stoikov (2014) — https://arxiv.org/abs/1011.6402
- Short-horizon mean reversion in crypto (preprint) — https://arxiv.org/pdf/2608.21888
- Explainable Patterns in Crypto Microstructure (preprint) — https://arxiv.org/html/2602.00776
- Joubert (2022) meta-labeling — https://en.wikipedia.org/wiki/Meta-Labeling (citation summary)
- Bailey et al. PBO — https://papers.ssrn.com/abstract=2326253 ; Deflated Sharpe — https://papers.ssrn.com/abstract=2460551
- Gu, Kelly, Xiu (2020) — https://www.nber.org/papers/w25398
- Fischer & Krauss (2018) — https://econpapers.repec.org/RePEc:eee:ejores:v:270:y:2018:i:2:p:654-669
- Jaquart, Dann, Weinhardt (2021) — https://doaj.org/article/cfc09801314441198e3e1c92a3914e54
- Akyildirim et al. (2023) — https://research-api.cbs.dk/ws/files/77072641/akyildirim_et_al_forecasting_mid_price_movement_acceptedversion.pdf
- Multi-timeframe GBDT on BTC (preprint) — https://arxiv.org/html/2602.10785v1
- Islas Anguiano & García-Medina (2025) — https://arxiv.org/abs/2511.00665v1
- Kaminski & Lo (2014) — https://dspace.mit.edu/handle/1721.1/114876
- Han, Zhou, Zhu — https://www.econbiz.de/Record/taming-momentum-crashes-a-simple-stop-loss-strategy-han-yufeng/10013006637
- Crypto HMM (preprint) — https://arxiv.org/pdf/1909.10957.pdf ; sparse jump model — https://link.springer.com/article/10.1007/s42521-024-00123-2

> Note: findings come from abstracts, publisher pages and summaries found on 2026-11-23. Read the full text before citing a specific number in a paper of your own.
