"""
Log of every configuration tried (needed for the SPA test, PBO and Deflated Sharpe in the paper).

Every run_backtest() / momentum_moves() call appends one row to results/trials.csv automatically:
time, kind, all settings, headline metrics. Never delete rows - the number of trials is part of the result.

    import trial_log
    trial_log.load()          # the table
"""
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PATH = Path(__file__).resolve().parent / 'results' / 'trials.csv'


def log(kind: str, settings: dict, metrics: dict) -> None:
    PATH.parent.mkdir(exist_ok=True)
    row = {'logged_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind,
           **{f'set_{k}': v for k, v in settings.items()}, **{f'res_{k}': v for k, v in metrics.items()}}
    old = load()
    if old is not None and set(row) - set(old.columns):            # new columns -> rewrite with the union
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(PATH, index=False)
        return
    cols = list(old.columns) if old is not None else list(row)
    new = not PATH.exists()
    with PATH.open('a', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in cols})


def load() -> pd.DataFrame | None:
    return pd.read_csv(PATH) if PATH.exists() and PATH.stat().st_size else None
