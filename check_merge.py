"""
Check data/*_master.parquet against raw_data/*.csv (run after data_merge.py):

    python check_merge.py

For every 5m row (decision at the 5m candle CLOSE = open_time + 5m) and every HTF (15m / 1h / 4h / 1d):
  - merged HTF close == close of the last HTF candle that has closed by the decision time
  - no row sees the HTF candle that is still open at the decision time (look-ahead)
Also: duplicate / missing candles, and HTF candles consistent with the 5m candles.
Exits with code 1 if anything fails.
"""
import sys

import numpy as np
import pandas as pd

TF = {'15m': '15min', '1h': '1h', '4h': '4h', '1d': '1D'}
SYMBOLS = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']


def raw(sym, tf):
    r = pd.read_csv(f'raw_data/{sym}_{tf}.csv', usecols=['open_time', 'high', 'low', 'close'])
    r['open_time'] = pd.to_datetime(r['open_time'], utc=True)
    return r.sort_values('open_time').reset_index(drop=True)


def asof(times, h, key, col='close'):
    return pd.merge_asof(pd.DataFrame({'t': times}), h[[key, col]].rename(columns={key: 't'}),
                         on='t', direction='backward')[col].to_numpy()


def main() -> int:
    failed = False
    for sym in SYMBOLS:
        m = pd.read_parquet(f'data/{sym}_master.parquet')
        t = pd.to_datetime(m['open_time'], utc=True)
        decide = t + pd.Timedelta('5min')                     # 5m candle close = decision time
        r5 = raw(sym, '5m')
        step = r5['open_time'].diff().dropna()
        same5 = np.allclose(m['close_5m'].to_numpy('float64'),
                            r5.set_index('open_time')['close'].reindex(t).to_numpy(), rtol=1e-6)
        print(f'\n=== {sym}: {len(m):,} rows  {t.iloc[0]} -> {t.iloc[-1]}')
        print(f'  5m : dup {r5.open_time.duplicated().sum()} | gaps {(step > pd.Timedelta("5min")).sum()} | '
              f'matches raw 5m {same5}')
        failed |= not same5
        for tf, freq in TF.items():
            h = raw(sym, tf)
            dur = pd.Timedelta(freq)
            h['close_at'] = h['open_time'] + dur
            got = m[f'close_{tf}'].to_numpy('float64')
            exp = asof(decide, h, 'close_at')                 # last candle closed by the decision time
            ok = np.isclose(got, exp, rtol=1e-6)
            # the HTF candle that contains this 5m candle; unfinished at the decision time unless it closes with it
            still_open = asof(t, h, 'open_time')
            open_bar = h.set_index('open_time')
            is_open = asof(t, h, 'open_time', 'close_at') > decide.to_numpy()
            leak = np.isclose(got, still_open, rtol=1e-9) & ~ok & is_open
            rs = (r5.set_index('open_time').resample(freq).agg({'high': 'max', 'low': 'min', 'close': 'last'})
                  .dropna())
            hh = open_bar[['high', 'low', 'close']].reindex(rs.index).dropna()
            cons = np.isclose(hh.to_numpy(), rs.loc[hh.index].to_numpy(), rtol=1e-6).all(1).mean()
            gaps = (h['open_time'].diff().dropna() > dur).sum()
            print(f'  {tf:>3}: last closed candle {ok.mean():8.3%} | look-ahead rows {leak.sum()} | '
                  f'dup {h.open_time.duplicated().sum()} gaps {gaps} | matches 5m resample {cons:.3%}')
            if not ok.all():
                bad = np.flatnonzero(~ok)[:3]
                print('     mismatch e.g.', [(str(t.iloc[b]), got[b], exp[b]) for b in bad])
            failed |= (not ok.all()) or leak.any()
    print('\nRESULT:', 'FAILED' if failed else 'all checks passed')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
