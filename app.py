import os
import time
from pathlib import Path
from typing import List, Optional

import cloudpickle
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/production"))

# ---------------------------------------------------------------------------
# Muat metadata dan model sekali saat layanan dinyalakan
# ---------------------------------------------------------------------------
with open(MODEL_DIR / "model_metadata.yaml") as f:
    META = yaml.safe_load(f)

FEATURES: List[str] = META["features"]
THRESHOLD = float(META.get("decision_threshold", 0.5))
MODEL_NAME = META["model"]["name"]
MODEL_VERSION = str(META["model"]["version"])

with open(MODEL_DIR / "model" / "model.pkl", "rb") as f:
    MODEL = cloudpickle.load(f)


# ---------------------------------------------------------------------------
# Skema input/output
# ---------------------------------------------------------------------------
class VideoFeatures(BaseModel):
    """17 fitur hasil feature engineering (lihat src/models/train.py)."""

    video_id: Optional[str] = Field(None, description="Opsional; dikembalikan apa adanya di response")
    view_count_log: float
    like_count_log: float
    comment_count_log: float
    video_age_hours: float
    views_per_hour_lifetime: float
    like_ratio: float
    comment_ratio: float
    view_growth_per_hour: float
    like_growth_per_hour: float
    comment_growth_per_hour: float
    hours_since_prev: float
    is_first_appearance: int = Field(..., ge=0, le=1)
    appearances_so_far: int = Field(..., ge=1)
    hours_since_first_seen: float
    category_id: int
    publish_hour_wib: int = Field(..., ge=-1, le=23)
    publish_dayofweek: int = Field(..., ge=-1, le=6)


class PredictRequest(BaseModel):
    instances: List[VideoFeatures] = Field(..., min_length=1, max_length=500)


class Prediction(BaseModel):
    video_id: Optional[str]
    prediction: int
    label: str
    probability: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    model_version: str
    threshold: float
    latency_ms: float
    predictions: List[Prediction]


# Pastikan skema API sama persis dengan fitur yang diharapkan model
_api_fields = set(VideoFeatures.model_fields) - {"video_id"}
if _api_fields != set(FEATURES):
    raise RuntimeError(f"Fitur API tidak cocok dengan model. "
                       f"Kurang: {set(FEATURES) - _api_fields}, lebih: {_api_fields - set(FEATURES)}")


# ---------------------------------------------------------------------------
# Aplikasi
# ---------------------------------------------------------------------------
app = FastAPI(
    title="YouTube Trending 24h Classifier API",
    description="Prediksi apakah video Top 50 Trending YouTube Indonesia masih bertahan ~24 jam kemudian.",
    version=MODEL_VERSION,
)


@app.get("/")
def root():
    return {"message": "YouTube Trending 24h Classifier API", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {
        "model": META["model"],
        "run_id": META["mlflow"]["run_id"],
        "metrics_test": META.get("metrics_test", {}),
        "lineage": META.get("lineage", {}),
        "features": FEATURES,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    rows = [inst.model_dump(exclude={"video_id"}) for inst in req.instances]
    X = pd.DataFrame(rows)[FEATURES]  # urutan kolom harus sama dengan saat training
    proba = MODEL.predict_proba(X)[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000

    preds = [
        Prediction(
            video_id=inst.video_id,
            prediction=int(p >= THRESHOLD),
            label="bertahan" if p >= THRESHOLD else "keluar",
            probability=round(float(p), 4),
        )
        for inst, p in zip(req.instances, proba)
    ]
    return PredictResponse(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        threshold=THRESHOLD,
        latency_ms=round(latency_ms, 2),
        predictions=preds,
    )