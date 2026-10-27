import argparse
import subprocess
import sys
from pathlib import Path
 
import matplotlib
import numpy as np
import pandas as pd
import yaml
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
 
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
RAW_DVC = ROOT / "data" / "raw.dvc"
 
EXPERIMENT_NAME = "youtube-trending-24h"
LABEL_MIN_H, LABEL_TARGET_H, LABEL_MAX_H = 20, 24, 28
WIB_OFFSET_H = 7  # published_at dari API dalam UTC
RANDOM_STATE = 42
 
# Target kualitas model dari LK-01
LK01_TARGETS = {"accuracy": 0.75, "precision": 0.70, "recall": 0.70, "f1": 0.70}
 
FEATURES = [
    "view_count_log", "like_count_log", "comment_count_log",
    "video_age_hours", "views_per_hour_lifetime",
    "like_ratio", "comment_ratio",
    "view_growth_per_hour", "like_growth_per_hour", "comment_growth_per_hour",
    "hours_since_prev", "is_first_appearance",
    "appearances_so_far", "hours_since_first_seen",
    "category_id", "publish_hour_wib", "publish_dayofweek",
]
 
# Konfigurasi untuk --grid (minimal 3 run sesuai instruksi LK-06)
GRID = [
    {"model": "logreg", "C": 1.0},
    {"model": "random_forest", "n_estimators": 200, "max_depth": 6},
    {"model": "xgboost", "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    {"model": "xgboost", "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"model": "xgboost", "n_estimators": 500, "max_depth": 6, "learning_rate": 0.03},
]
 
 
# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize(None) if s.dt.tz is not None else s
 
 
def load_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("processed_trending_*.parquet"))
    if not files:
        sys.exit(f"Tidak ada file di {DATA_DIR}. Jalankan 'dvc pull' terlebih dahulu.")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
 
    df["snapshot_timestamp"] = to_naive(df["snapshot_timestamp"])
    if "published_at" in df.columns:
        df["published_at"] = to_naive(df["published_at"])
    for c in ["view_count", "like_count", "comment_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(-1).astype(int)
 
    df = df.drop_duplicates(["video_id", "snapshot_timestamp"]).reset_index(drop=True)
    print(f"[DATA] {len(files)} file, {len(df)} baris, "
          f"{df['snapshot_timestamp'].nunique()} snapshot, {df['video_id'].nunique()} video unik")
    return df
 
 
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """label = 1 jika video ada di snapshot terdekat ke t+24 jam (jendela 20-28 jam)."""
    snap_sets = df.groupby("snapshot_timestamp")["video_id"].apply(set)
    times = pd.DatetimeIndex(snap_sets.index)
    lo, hi = pd.Timedelta(hours=LABEL_MIN_H), pd.Timedelta(hours=LABEL_MAX_H)
    target = pd.Timedelta(hours=LABEL_TARGET_H)
 
    target_snapshot = {}
    for t in times:
        cands = times[(times >= t + lo) & (times <= t + hi)]
        if len(cands):
            offsets = np.abs((cands - (t + target)).total_seconds())
            target_snapshot[t] = cands[offsets.argmin()]
        else:
            target_snapshot[t] = pd.NaT
 
    df["label_snapshot"] = df["snapshot_timestamp"].map(target_snapshot)
    df["label"] = [
        float(vid in snap_sets[ts]) if pd.notna(ts) else np.nan
        for vid, ts in zip(df["video_id"], df["label_snapshot"])
    ]
    return df
 
 
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fitur hanya dari kondisi saat ini dan riwayat sebelumnya (tanpa data masa depan)."""
    df = df.sort_values(["video_id", "snapshot_timestamp"]).copy()
    g = df.groupby("video_id")
 
    df["hours_since_prev"] = g["snapshot_timestamp"].diff().dt.total_seconds() / 3600
    for c, name in [("view_count", "view"), ("like_count", "like"), ("comment_count", "comment")]:
        delta = df[c] - g[c].shift()
        df[f"{name}_growth_per_hour"] = (delta / df["hours_since_prev"]).replace([np.inf, -np.inf], np.nan)
    df["is_first_appearance"] = df["hours_since_prev"].isna().astype(int)
    df["appearances_so_far"] = g.cumcount() + 1
    first_seen = g["snapshot_timestamp"].transform("min")
    df["hours_since_first_seen"] = (df["snapshot_timestamp"] - first_seen).dt.total_seconds() / 3600
 
    if "published_at" in df.columns:
        df["video_age_hours"] = (df["snapshot_timestamp"] - df["published_at"]).dt.total_seconds() / 3600
        published_wib = df["published_at"] + pd.Timedelta(hours=WIB_OFFSET_H)
        df["publish_hour_wib"] = published_wib.dt.hour
        df["publish_dayofweek"] = published_wib.dt.dayofweek
    else:
        df["publish_hour_wib"] = -1
        df["publish_dayofweek"] = -1
    df["video_age_hours"] = df["video_age_hours"].clip(lower=0)
 
    df["views_per_hour_lifetime"] = df["view_count"] / df["video_age_hours"].clip(lower=1)
    df["like_ratio"] = df["like_count"] / df["view_count"].clip(lower=1)
    df["comment_ratio"] = df["comment_count"] / df["view_count"].clip(lower=1)
    for c in ["view_count", "like_count", "comment_count"]:
        df[f"{c}_log"] = np.log1p(df[c])
 
    df[FEATURES] = df[FEATURES].fillna(0)
    return df.sort_values(["snapshot_timestamp", "video_id"]).reset_index(drop=True)
 
 
def time_split(df: pd.DataFrame, test_frac: float):
    """Snapshot terakhir -> test. Train dipotong LABEL_MAX_H jam sebelum test (purge)
    karena label baris train dihitung dari snapshot hingga 28 jam ke depan."""
    labeled = df[df["label"].notna()].copy()
    labeled["label"] = labeled["label"].astype(int)
    times = np.sort(labeled["snapshot_timestamp"].unique())
    split_time = pd.Timestamp(times[int(len(times) * (1 - test_frac))])
 
    test = labeled[labeled["snapshot_timestamp"] >= split_time]
    train = labeled[labeled["snapshot_timestamp"] <= split_time - pd.Timedelta(hours=LABEL_MAX_H)]
    if train.empty or test.empty or train["label"].nunique() < 2:
        sys.exit("[SPLIT] Data belum cukup untuk split train/test. Tunggu snapshot bertambah.")
    print(f"[SPLIT] train {len(train)} baris (s/d {train['snapshot_timestamp'].max()}) | "
          f"test {len(test)} baris (mulai {split_time})")
    return train, test, split_time
 
 
# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(cfg: dict, y_train: pd.Series):
    balance = cfg.get("balance", True)
    if cfg["model"] == "xgboost":
        pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
        return XGBClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"], subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            scale_pos_weight=(neg / pos) if balance and pos else 1.0,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )
    if cfg["model"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"], max_depth=cfg.get("max_depth"),
            class_weight="balanced" if balance else None,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    if cfg["model"] == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=cfg["C"], class_weight="balanced" if balance else None,
                               max_iter=2000, random_state=RANDOM_STATE),
        )
    raise ValueError(f"Model tidak dikenal: {cfg['model']}")
 
 
def data_version_tags() -> dict:
    tags = {}
    try:
        out = yaml.safe_load(RAW_DVC.read_text())["outs"][0]
        tags["data_raw_md5"] = out.get("md5", "")
        tags["data_raw_nfiles"] = str(out.get("nfiles", ""))
    except Exception:
        pass
    try:
        tags["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return tags
 
 
def log_sklearn_model(model, signature, input_example):
    # MLflow 3 memakai argumen `name`, MLflow 2 memakai `artifact_path`
    try:
        mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=input_example)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path="model", signature=signature, input_example=input_example)
 
 
def run_name_of(cfg: dict) -> str:
    parts = [cfg["model"]] + [f"{k}={v}" for k, v in cfg.items() if k not in ("model", "balance")]
    return "_".join(parts)
 
 
def train_one(cfg: dict, train: pd.DataFrame, test: pd.DataFrame, split_time, n_total: int) -> dict:
    X_train, y_train = train[FEATURES], train["label"]
    X_test, y_test = test[FEATURES], test["label"]
 
    with mlflow.start_run(run_name=run_name_of(cfg)) as run:
        # --- Parameter ---
        for k, v in cfg.items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_features", len(FEATURES))
        mlflow.log_param("label_window_hours", f"{LABEL_MIN_H}-{LABEL_MAX_H}")
        mlflow.log_param("n_train", len(train))
        mlflow.log_param("n_test", len(test))
        mlflow.log_param("n_rows_total", n_total)
        mlflow.log_param("train_pos_rate", round(float(y_train.mean()), 4))
        mlflow.log_param("test_split_time", str(split_time))
        mlflow.set_tags(data_version_tags())
 
        # --- Training ---
        model = build_model(cfg, y_train)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
 
        # --- Metrik ---
        metrics = {
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "train_f1": f1_score(y_train, model.predict(X_train), zero_division=0),
        }
        if y_test.nunique() == 2:
            metrics["test_roc_auc"] = roc_auc_score(y_test, proba)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        meets = all(metrics[f"test_{m}"] >= t for m, t in LK01_TARGETS.items())
        mlflow.set_tag("meets_lk01_target", str(meets))
 
        # --- Artefak visual ---
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False,
                                                display_labels=["keluar (0)", "bertahan (1)"])
        ax.set_title("Confusion matrix (test)")
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
 
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            order = np.argsort(importances)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(np.array(FEATURES)[order], importances[order])
            ax.set_title("Feature importance")
            fig.tight_layout()
            mlflow.log_figure(fig, "feature_importance.png")
            plt.close(fig)
 
        # --- Model ---
        signature = infer_signature(X_test, proba)
        log_sklearn_model(model, signature, X_test.head(3))
 
        print(f"[RUN] {run_name_of(cfg):<55} F1={metrics['test_f1']:.3f} "
              f"Acc={metrics['test_accuracy']:.3f} run_id={run.info.run_id}")
        return {"run_name": run_name_of(cfg), "run_id": run.info.run_id, **metrics}
 
 
# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def cfg_from_args(a) -> dict:
    cfg = {"model": a.model, "balance": not a.no_balance}
    if a.model == "xgboost":
        cfg.update(n_estimators=a.n_estimators, max_depth=a.max_depth, learning_rate=a.learning_rate,
                   subsample=a.subsample, colsample_bytree=a.colsample_bytree)
    elif a.model == "random_forest":
        cfg.update(n_estimators=a.n_estimators, max_depth=a.max_depth)
    else:
        cfg.update(C=a.C)
    return cfg
 
 
def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="Training + MLflow tracking (LK-06)")
    ap.add_argument("--model", choices=["xgboost", "random_forest", "logreg"], default="xgboost")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--no-balance", action="store_true", help="matikan penyeimbangan kelas")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--grid", action="store_true", help="jalankan semua konfigurasi di GRID")
    ap.add_argument("--experiment", default=EXPERIMENT_NAME)
    args = ap.parse_args()
 
    if not mlflow.get_tracking_uri().startswith(("http", "databricks")):
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)
    print(f"[MLFLOW] tracking: {mlflow.get_tracking_uri()} | experiment: {args.experiment}")
 
    df = add_features(add_label(load_data()))
    n_lab = int(df["label"].notna().sum())
    pos_rate = df["label"].mean()
    print(f"[LABEL] {n_lab} baris berlabel | proporsi bertahan (label 1): {pos_rate:.1%}")
    train, test, split_time = time_split(df, args.test_frac)
 
    configs = [dict(c, balance=not args.no_balance) for c in GRID] if args.grid else [cfg_from_args(args)]
    results = [train_one(cfg, train, test, split_time, len(df)) for cfg in configs]
 
    if len(results) > 1:
        summary = pd.DataFrame(results).sort_values("test_f1", ascending=False)
        print("\n== Ringkasan (urut test_f1) ==")
        cols = ["run_name", "test_f1", "test_accuracy", "test_precision", "test_recall", "train_f1"]
        print(summary[cols].round(3).to_string(index=False))
        print(f"\nKandidat terbaik: {summary.iloc[0]['run_name']} (run_id {summary.iloc[0]['run_id']})")
 
 
if __name__ == "__main__":
    main()