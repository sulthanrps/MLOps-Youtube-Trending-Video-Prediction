import argparse
import json
import sys
from pathlib import Path

import cloudpickle
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_request.json"
MODEL_PKL = ROOT / "models" / "production" / "model" / "model.pkl"


def build_sample(n: int) -> dict:
    sys.path.insert(0, str(ROOT / "src" / "models"))
    import train as T  # pakai feature engineering yang sama dengan training

    df = T.add_features(T.load_data())
    latest = df[df["snapshot_timestamp"] == df["snapshot_timestamp"].max()].head(n)
    instances = []
    for _, r in latest.iterrows():
        row = {"video_id": str(r["video_id"])}
        for c in T.FEATURES:
            v = r[c]
            row[c] = int(v) if c in ("is_first_appearance", "appearances_so_far", "category_id",
                                     "publish_hour_wib", "publish_dayofweek") else float(v)
        instances.append(row)
    payload = {"instances": instances}
    SAMPLE.write_text(json.dumps(payload, indent=2))
    print(f"[SAMPLE] {len(instances)} video dari snapshot {latest['snapshot_timestamp'].iloc[0]} "
          f"-> {SAMPLE.relative_to(ROOT)}")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--refresh", action="store_true", help="buat ulang sample_request.json dari data")
    ap.add_argument("-n", type=int, default=5, help="jumlah video untuk --refresh")
    a = ap.parse_args()

    payload = build_sample(a.n) if a.refresh else json.loads(SAMPLE.read_text())

    health = requests.get(f"{a.url}/health", timeout=10).json()
    print(f"[HEALTH] {health}")

    resp = requests.post(f"{a.url}/predict", json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    print(f"[API] model {body['model_name']} v{body['model_version']} | latency {body['latency_ms']} ms")

    with open(MODEL_PKL, "rb") as f:
        model = cloudpickle.load(f)
    with open(ROOT / "models" / "production" / "model_metadata.yaml") as f:
        feats = yaml.safe_load(f)["features"]
    X = pd.DataFrame(payload["instances"])[feats]
    local = model.predict_proba(X)[:, 1]

    print(f"\n{'video_id':<14}{'API':>8}{'lokal':>8}  label")
    ok = True
    for p, l in zip(body["predictions"], local):
        same = abs(p["probability"] - round(float(l), 4)) < 1e-4
        ok &= same
        print(f"{str(p['video_id']):<14}{p['probability']:>8.4f}{l:>8.4f}  {p['label']}{'' if same else '  <- BEDA'}")

    print("\nHASIL:", "LULUS - prediksi kontainer identik dengan model Production" if ok
          else "GAGAL - prediksi kontainer berbeda dengan model lokal")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()