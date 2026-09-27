import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3/videos"

def fetch_trending_videos():
    params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": "ID",
        "maxResults": 50,
        "key": API_KEY
    }
    
    response = requests.get(BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json().get("items", [])

def save_raw_data(items):
    if not items:
        return
    
    snapshot_time = datetime.now()
    extracted_data = []
    
    for item in items:
        video_data = {
            "snapshot_timestamp": snapshot_time,
            "video_id": item.get("id"),
            "title": item.get("snippet", {}).get("title"),
            "category_id": item.get("snippet", {}).get("categoryId"),
            "published_at": item.get("snippet", {}).get("publishedAt"),
            "view_count": item.get("statistics", {}).get("viewCount", 0),
            "like_count": item.get("statistics", {}).get("likeCount", 0),
            "comment_count": item.get("statistics", {}).get("commentCount", 0)
        }
        extracted_data.append(video_data)

    df = pd.DataFrame(extracted_data)

    # print(df)
    
    df['snapshot_timestamp'] = pd.to_datetime(df['snapshot_timestamp'])
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['view_count'] = df['view_count'].astype(int)
    df['like_count'] = df['like_count'].astype(int)
    df['comment_count'] = df['comment_count'].astype(int)
    
    os.makedirs("data/raw", exist_ok=True)
    timestamp_str = snapshot_time.strftime("%Y%m%d_%H%M")
    filepath = f"data/raw/raw_trending_{timestamp_str}.parquet"
    
    df.to_parquet(filepath, index=False)
    print(f"[SUCCESS] Tersimpan ke: {filepath}")

if __name__ == "__main__":
    items = fetch_trending_videos()
    save_raw_data(items)