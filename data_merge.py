import pandas as pd
import os

from ema_create import add_emas, ema_columns
from zone_create import add_zones, zone_columns, add_zone_features, add_entry_signal

# Directory paths and symbol configurations
data_dir = "./raw_data/"  # Update this path to your CSV folder
output_dir = "./data/"
os.makedirs(output_dir, exist_ok=True)

symbols = ["BNBUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
higher_timeframes = ["15m", "1h", "4h", "1d"]

# Exact feature columns from your CSV schema
feature_columns = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"
]

TF_DURATION = {"5m": pd.Timedelta("5min"), "15m": pd.Timedelta("15min"), "1h": pd.Timedelta("1h"),
               "4h": pd.Timedelta("4h"), "1d": pd.Timedelta("1D")}


def prepare_htf_data(df, tf_suffix, extra_columns=()):
    """
    Processes higher timeframe (HTF) data. Each HTF row is keyed by the time its candle CLOSES
    ('available_at' = open_time + duration), so it can only be merged into 5m rows that close
    at or after that moment -> no unfinished candle is ever visible (no future data leakage).
    """
    # Convert Open_time to datetime and sort chronologically
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    
    # Keep only Open_time and relevant numeric features (dropping redundant Close_time)
    cols_to_keep = ["open_time"] + [col for col in [*feature_columns, *extra_columns] if col in df.columns]
    df = df[cols_to_keep].copy()
    
    # Rename columns with timeframe suffix (e.g., close_15m, volume_1h)
    rename_map = {col: f"{col}_{tf_suffix}" for col in cols_to_keep if col != "open_time"}
    df = df.rename(columns=rename_map)
    
    # Key every HTF candle by its close time: it becomes visible only once it has closed
    df["available_at"] = df["open_time"] + TF_DURATION[tf_suffix]
    return df.drop(columns="open_time")

# Process each cryptocurrency pair
for symbol in symbols:
    print(f"Processing {symbol}...")
    
    # 1. Load base 5m dataset
    base_path = os.path.join(data_dir, f"{symbol}_5m.csv")
    df_base = pd.read_csv(base_path)
    
    df_base["open_time"] = pd.to_datetime(df_base["open_time"])
    df_base = df_base.sort_values("open_time").reset_index(drop=True)
    
    # Keep base columns and add '_5m' suffix to features for clarity
    base_cols = ["open_time"] + [col for col in feature_columns if col in df_base.columns]
    df_base = df_base[base_cols].copy()
    df_base = df_base.rename(columns={col: f"{col}_5m" for col in base_cols if col != "open_time"})
    
    # 2. Merge higher timeframes sequentially using point-in-time alignment.
    #    The decision is made when the 5m candle CLOSES (entry at the next open), so a 5m row sees
    #    every HTF candle that has closed by then - including the one closing together with it
    #    (5m 02:55-03:00 sees the 15m 02:45-03:00 candle).
    df_base["decision_time"] = df_base["open_time"] + TF_DURATION["5m"]
    for tf in higher_timeframes:
        htf_path = os.path.join(data_dir, f"{symbol}_{tf}.csv")
        df_htf = pd.read_csv(htf_path)

        # HTF rolling indicators are computed BEFORE prepare_htf_data() on the HTF candles themselves
        extra = zone_columns()
        df_htf = add_zones(df_htf, tf)
        if tf == "15m":
            df_htf = add_emas(df_htf)
            extra = extra + ema_columns()
        df_htf_closed = prepare_htf_data(df_htf, tf, extra)

        df_base = pd.merge_asof(
            df_base,
            df_htf_closed,
            left_on="decision_time",
            right_on="available_at",
            direction="backward"
        ).drop(columns="available_at")
    df_base = df_base.drop(columns="decision_time")

    # 3. Drop initial NaN rows caused by indicator warm-up
    df_base = df_base.dropna().reset_index(drop=True)

    # 4. 5m-level zone features (ATR distances, confluence) + entry trigger
    df_base = add_zone_features(df_base)
    df_base = add_entry_signal(df_base)
    df_base = df_base.dropna().reset_index(drop=True)

    # optimize memory to float32    
    float_cols = df_base.select_dtypes(include=["float64"]).columns
    df_base[float_cols] = df_base[float_cols].astype("float32")
    
    int_cols = df_base.select_dtypes(include=["int64"]).columns
    df_base[int_cols] = df_base[int_cols].astype("int32")
    
    # 5. Save master dataset in Parquet format
    output_file = os.path.join(output_dir, f"{symbol}_master.parquet")
    df_base.to_parquet(output_file, index=False)
    print(f"Saved: {output_file} | Shape: {df_base.shape}")