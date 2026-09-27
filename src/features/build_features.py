import os
import glob
import pandas as pd
import numpy as np

def process_features():
    raw_files = sorted(glob.glob("data/raw/*.parquet"), key=os.path.getctime)
    if not raw_files:
        raise FileNotFoundError("Data raw tidak ditemukan.")
    
    latest_file = raw_files[-1]
    df_latest = pd.read_parquet(latest_file)
    
    df_latest['category_id'] = df_latest['category_id'].fillna(0).astype(int)
    
    df_latest['snapshot_timestamp'] = pd.to_datetime(df_latest['snapshot_timestamp']).dt.tz_localize(None)
    df_latest['published_at'] = pd.to_datetime(df_latest['published_at']).dt.tz_localize(None)
    
    df_latest['video_age_hours'] = (df_latest['snapshot_timestamp'] - df_latest['published_at']).dt.total_seconds() / 3600
    df_latest['video_age_hours'] = df_latest['video_age_hours'].round(2)
    
    df_latest['upload_hour'] = df_latest['published_at'].dt.hour
    
    df_latest['view_growth_6h'] = 0 
    df_latest['like_growth_6h'] = 0
    
    if len(raw_files) >= 2:
        prev_file = raw_files[-2]
        df_prev = pd.read_parquet(prev_file)
        
        df_merged = df_latest.merge(
            df_prev[['video_id', 'view_count', 'like_count']], 
            on='video_id', 
            how='left', 
            suffixes=('', '_prev')
        )
        
        df_latest['view_growth_6h'] = (df_merged['view_count'] - df_merged['view_count_prev']).fillna(0).astype(int)
        df_latest['like_growth_6h'] = (df_merged['like_count'] - df_merged['like_count_prev']).fillna(0).astype(int)
        
    else:
        print("[INFO] Hanya ada 1 snapshot. Nilai growth di set ke 0.")
    
    os.makedirs("data/processed", exist_ok=True)
    filename = os.path.basename(latest_file).replace("raw_", "processed_")
    output_path = os.path.join("data/processed", filename)
    
    df_latest.to_parquet(output_path, index=False)
    print(f"[SUCCESS] Fitur tersimpan ke: {output_path}")

if __name__ == "__main__":
    process_features()