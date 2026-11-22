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

def prepare_htf_data(df, tf_suffix, extra_columns=()):
    """
    Processes higher timeframe (HTF) data and shifts it by 1 period 
    to strictly prevent future data leakage.
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
    
    # Shift all HTF features forward by 1 period so only closed candles are visible
    shifted_cols = list(rename_map.values())
    df[shifted_cols] = df[shifted_cols].shift(1)
    return df

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
    
    # 2. Merge higher timeframes sequentially using point-in-time alignment
    for tf in higher_timeframes:
        htf_path = os.path.join(data_dir, f"{symbol}_{tf}.csv")
        df_htf = pd.read_csv(htf_path)
        
        # HTF rolling indicators are computed BEFORE prepare_htf_data() so they get shift(1)-ed too
        extra = zone_columns()
        df_htf = add_zones(df_htf, tf)
        if tf == "15m":
            df_htf = add_emas(df_htf)
            extra = extra + ema_columns()
        df_htf_shifted = prepare_htf_data(df_htf, tf, extra)
        
        df_base = pd.merge_asof(
            df_base, 
            df_htf_shifted, 
            on="open_time", 
            direction="backward"
        )
        
    # 3. Drop initial NaN rows caused by shift(1) / indicator warm-up
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