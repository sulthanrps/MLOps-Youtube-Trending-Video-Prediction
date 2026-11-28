import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
INT_FEATURES = {"is_first_appearance", "appearances_so_far", "category_id",
                "publish_hour_wib", "publish_dayofweek"}


def main():
    features = yaml.safe_load((ROOT / "models/production/model_metadata.yaml").read_text())["features"]
    instances = json.loads((ROOT / "sample_request.json").read_text())["instances"]
    data = [[int(r[c]) if c in INT_FEATURES else float(r[c]) for c in features] for r in instances]
    payload = {"dataframe_split": {"columns": features, "data": data}}
    out = ROOT / "sample_serving_request.json"
    out.write_text(json.dumps(payload, indent=2))
    ids = [r.get("video_id") for r in instances]
    (ROOT / "sample_serving_ids.json").write_text(json.dumps(ids))
    print(f"{len(data)} baris x {len(features)} fitur -> {out.name}")


if __name__ == "__main__":
    main()