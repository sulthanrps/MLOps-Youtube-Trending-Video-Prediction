import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cloudpickle
import pandas as pd
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/production"))
MODEL_URI = os.getenv("MODEL_URI", "").strip()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")


def log(msg: str):
    print(f"[MODEL] {msg}", flush=True)


def resolve_registry_version(uri: str):
    """models:/<nama>@<alias> atau models:/<nama>/<stage|versi> -> objek ModelVersion."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    body = uri[len("models:/"):]
    if "@" in body:
        name, alias = body.split("@", 1)
        return client.get_model_version_by_alias(name, alias)
    name, ref = body.split("/", 1)
    if ref.isdigit():
        return client.get_model_version(name, ref)
    return client.get_latest_versions(name, stages=[ref])[0]


def load_from_registry(uri: str, retries: int = 10, delay: int = 3):
    """Tarik artefak model dari mlflow-server (dengan retry saat server belum siap)."""
    import mlflow

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            mv = resolve_registry_version(uri)
            dst = tempfile.mkdtemp(prefix="model-")
            local = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=dst)
            with open(Path(local) / "model.pkl", "rb") as f:
                model = cloudpickle.load(f)
            log(f"model dimuat dari {uri} (registry v{mv.version}, stage {mv.current_stage}, "
                f"run {mv.run_id}) via {TRACKING_URI}")
            return model, mv
        except Exception as e:
            last_err = e
            log(f"percobaan {attempt}/{retries} gagal: {type(e).__name__}: {str(e)[:150]}")
            time.sleep(delay)
    # Gagal keras, tidak diam-diam memakai model lokal: kegagalan harus terlihat
    raise RuntimeError(f"gagal memuat model dari {uri}: {last_err}")


# ---------------------------------------------------------------------------
# Muat metadata dan model sekali saat layanan dinyalakan
# ---------------------------------------------------------------------------
with open(MODEL_DIR / "model_metadata.yaml") as f:
    META = yaml.safe_load(f)

FEATURES: List[str] = META["features"]
THRESHOLD = float(META.get("decision_threshold", 0.5))
MODEL_NAME = META["model"]["name"]

if MODEL_URI:
    MODEL, _MV = load_from_registry(MODEL_URI)
    MODEL_SOURCE = "mlflow-registry"
    MODEL_VERSION = str(_MV.version)
    MODEL_RUN_ID = _MV.run_id
    MODEL_STAGE = _MV.current_stage
else:
    with open(MODEL_DIR / "model" / "model.pkl", "rb") as f:
        MODEL = cloudpickle.load(f)
    MODEL_SOURCE = "local-file"
    MODEL_VERSION = str(META["model"]["version"])
    MODEL_RUN_ID = META["mlflow"]["run_id"]
    MODEL_STAGE = META["model"].get("stage", "")
    log(f"model dimuat dari file lokal {MODEL_DIR}/model/model.pkl (v{MODEL_VERSION})")


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
    return {"status": "ok", "model_loaded": MODEL is not None,
            "model_source": MODEL_SOURCE, "model_version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return {
        "source": MODEL_SOURCE,
        "model_uri": MODEL_URI or str(MODEL_DIR / "model"),
        "tracking_uri": TRACKING_URI or None,
        "model_name": MODEL_NAME,
        "registry_version": MODEL_VERSION,
        "stage": MODEL_STAGE,
        "run_id": MODEL_RUN_ID,
        "trained_version_dagshub": META["model"]["version"],
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