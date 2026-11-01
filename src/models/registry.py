import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import train as T  #  (pakai ulang load_data, add_label, add_features, time_split, FEATURES)

MODEL_NAME = "youtube-trending-24h-classifier"
EXPORT_DIR = ROOT / "models" / "production"
STAGE_ALIAS = {"Staging": "staging", "Production": "production"}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_tracking():
    load_dotenv(ROOT / ".env")
    if not mlflow.get_tracking_uri().startswith(("http", "databricks")):
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    print(f"[MLFLOW] {mlflow.get_tracking_uri()}")


def get_version(c: MlflowClient, name: str, version=None, stage=None, alias=None):
    if version:
        return c.get_model_version(name, str(version))
    if alias:
        return c.get_model_version_by_alias(name, alias)
    latest = c.get_latest_versions(name, stages=[stage])
    if not latest:
        sys.exit(f"Tidak ada versi {name} dengan stage {stage}.")
    return latest[0]


# ----------------------------------------------------------------------------
def cmd_list(a):
    c = MlflowClient()
    versions = sorted(c.search_model_versions(f"name='{a.name}'"), key=lambda v: int(v.version))
    print(f"\n== {a.name}: {len(versions)} versi ==")
    for v in versions:
        v = c.get_model_version(a.name, v.version)
        run = c.get_run(v.run_id)
        m, p = run.data.metrics, run.data.params
        aliases = ",".join(v.aliases) if getattr(v, "aliases", None) else "-"
        print(f"v{v.version:<3} stage={v.current_stage:<10} alias={aliases:<22} "
              f"F1={m.get('test_f1', float('nan')):.3f} Acc={m.get('test_accuracy', float('nan')):.3f} "
              f"| n_est={p.get('n_estimators', '-')} depth={p.get('max_depth', '-')} "
              f"lr={p.get('learning_rate', '-')} n_rows={p.get('n_rows_total', '-')} | run={v.run_id[:8]}")


def cmd_register(a):
    c = MlflowClient()
    mv = mlflow.register_model(f"runs:/{a.run_id}/model", a.name)
    if a.description:
        c.update_model_version(a.name, mv.version, description=a.description)
    run = c.get_run(a.run_id)
    for k in ("data_raw_md5", "data_raw_nfiles", "git_commit"):
        if k in run.data.tags:
            c.set_model_version_tag(a.name, mv.version, k, run.data.tags[k])
    print(f"[REGISTER] {a.name} v{mv.version} <- run {a.run_id} ({run.info.run_name})")
    cmd_list(a)


def cmd_compare(a):
    """Champion vs challenger: semua versi dievaluasi pada data test terbaru yang sama."""
    df = T.add_features(T.add_label(T.load_data()))
    _, test, split_time = T.time_split(df, a.test_frac)
    X, y = test[T.FEATURES], test["label"]
    print(f"\n[COMPARE] data test bersama: {len(test)} baris, mulai {split_time}")

    c = MlflowClient()
    rows = []
    for ver in a.versions:
        mv = c.get_model_version(a.name, str(ver))
        run = c.get_run(mv.run_id)
        trained_split = pd.Timestamp(run.data.params.get("test_split_time", "1970-01-01"))
        if trained_split > split_time + pd.Timedelta(minutes=1):
            print(f"  [PERINGATAN] v{ver} dilatih dengan data hingga {trained_split}; "
                  f"sebagian data test ini sudah pernah dilihat versi tersebut.")
        model = mlflow.sklearn.load_model(f"models:/{a.name}/{ver}")
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "version": f"v{ver}",
            "train_rows": run.data.params.get("n_train", "-"),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, proba) if y.nunique() == 2 else float("nan"),
        })
    res = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print(res.round(3).to_string(index=False))
    print(f"\nKandidat Production (F1 tertinggi): {res.iloc[0]['version']}")


def cmd_promote(a):
    c = MlflowClient()
    v = str(a.version)
    c.transition_model_version_stage(a.name, v, a.stage,
                                     archive_existing_versions=(a.stage == "Production"))
    alias = STAGE_ALIAS.get(a.stage)
    if alias:
        c.set_registered_model_alias(a.name, alias, v)
    if a.stage == "Production":
        c.set_registered_model_alias(a.name, "champion", v)
        try: 
            if c.get_model_version_by_alias(a.name, "staging").version == v:
                c.delete_registered_model_alias(a.name, "staging")
        except Exception:
            pass
    c.set_model_version_tag(a.name, v, f"{a.stage.lower()}_at", now_iso())
    if a.reason:
        c.set_model_version_tag(a.name, v, f"{a.stage.lower()}_reason", a.reason)
    print(f"[PROMOTE] {a.name} v{v} -> {a.stage}")
    cmd_list(a)


def cmd_export(a):
    """Unduh artefak model + metadata ke models/production/ agar dapat dilacak DVC."""
    c = MlflowClient()
    mv = get_version(c, a.name, a.version, a.stage)
    mv = c.get_model_version(a.name, mv.version)
    run = c.get_run(mv.run_id)

    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    EXPORT_DIR.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{a.name}/{mv.version}",
                                        dst_path=str(EXPORT_DIR / "model"))

    meta = {
        "model": {
            "name": a.name,
            "version": int(mv.version),
            "stage": mv.current_stage,
            "aliases": list(getattr(mv, "aliases", []) or []),
            "uri_stage": f"models:/{a.name}/{mv.current_stage}",
            "uri_version": f"models:/{a.name}/{mv.version}",
            "registered_at": datetime.fromtimestamp(mv.creation_timestamp / 1000, timezone.utc)
                                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "description": mv.description or "",
        },
        "mlflow": {
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_id": run.info.experiment_id,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
        },
        "params": dict(run.data.params),
        "metrics_test": {k: round(val, 4) for k, val in run.data.metrics.items()},
        "lineage": {
            "data_source_branch": "data-updates",
            "data_raw_md5": run.data.tags.get("data_raw_md5", ""),
            "data_raw_nfiles": run.data.tags.get("data_raw_nfiles", ""),
            "code_git_commit": run.data.tags.get("git_commit", ""),
            "training_script": "src/models/train.py",
        },
        "features": list(T.FEATURES),
        "decision_threshold": 0.5,
        "exported_at": now_iso(),
    }
    with open(EXPORT_DIR / "model_metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    print(f"[EXPORT] v{mv.version} ({mv.current_stage}) -> {EXPORT_DIR.relative_to(ROOT)}/")
    print("Langkah berikutnya:\n  dvc add models/production\n"
          "  git add models/production.dvc models/.gitignore\n  dvc push models/production.dvc")


def cmd_verify(a):
    """Simulasi layanan inferensi: muat model Production dan prediksi snapshot terbaru."""
    c = MlflowClient()
    mv = get_version(c, a.name, a.version, a.stage)
    uri = f"models:/{a.name}/{a.stage}" if not a.version else f"models:/{a.name}/{a.version}"

    t0 = time.perf_counter()
    model = mlflow.pyfunc.load_model(uri)
    load_s = time.perf_counter() - t0
    print(f"\n[VERIFY] mlflow.pyfunc.load_model('{uri}')")
    print(f"  versi: v{mv.version} | stage: {mv.current_stage} | run_id: {model.metadata.run_id}")
    print(f"  waktu load model: {load_s:.2f} s")

    df = T.add_features(T.load_data())
    latest_ts = df["snapshot_timestamp"].max()
    latest = df[df["snapshot_timestamp"] == latest_ts].reset_index(drop=True)
    X = latest[T.FEATURES]

    t0 = time.perf_counter()
    model.predict(X.head(1))
    latency_ms = (time.perf_counter() - t0) * 1000
    preds = model.predict(X)
    proba = mlflow.sklearn.load_model(uri).predict_proba(X)[:, 1]

    status = "memenuhi" if latency_ms <= 500 else "melebihi"
    print(f"  latensi prediksi 1 video: {latency_ms:.1f} ms ({status} target LK-01 <= 500 ms)")
    print(f"  snapshot terbaru: {latest_ts} | {len(latest)} video | "
          f"diprediksi bertahan 24 jam: {int(preds.sum())}")

    out = latest.assign(pred=preds, proba=proba).sort_values("proba", ascending=False)
    title_col = "title" if "title" in out.columns else "video_id"
    print("\n  5 video dengan peluang bertahan tertinggi:")
    for _, r in out.head(5).iterrows():
        print(f"   {r['proba']:.2f}  {str(r[title_col])[:70]}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=MODEL_NAME)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list")

    s = sub.add_parser("register")
    s.add_argument("--run-id", required=True)
    s.add_argument("--description", default="")

    s = sub.add_parser("compare")
    s.add_argument("--versions", nargs="+", type=int, required=True)
    s.add_argument("--test-frac", type=float, default=0.25)

    s = sub.add_parser("promote")
    s.add_argument("--version", type=int, required=True)
    s.add_argument("--stage", choices=["Staging", "Production", "Archived", "None"], required=True)
    s.add_argument("--reason", default="")

    for cmd in ("export", "verify"):
        s = sub.add_parser(cmd)
        s.add_argument("--stage", default="Production")
        s.add_argument("--version", type=int, default=None)

    a = ap.parse_args()
    setup_tracking()
    {"list": cmd_list, "register": cmd_register, "compare": cmd_compare,
     "promote": cmd_promote, "export": cmd_export, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()