import os
import sys
import time
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

NAME = os.getenv("MODEL_NAME", "youtube-trending-24h-classifier")
SRC = Path(os.getenv("SEED_MODEL_DIR", "/seed/production"))
ALIAS = "champion"
EXPERIMENT = "youtube-trending-24h-serving"


def log(msg):
    print(f"[SEEDER] {msg}", flush=True)


def wait_for_server(client: MlflowClient, retries: int = 20, delay: int = 3):
    for i in range(1, retries + 1):
        try:
            client.search_experiments(max_results=1)
            return
        except Exception as e:  # server belum siap
            log(f"menunggu {mlflow.get_tracking_uri()} ({i}/{retries}): {type(e).__name__}")
            time.sleep(delay)
    sys.exit("mlflow-server tidak dapat dihubungi")


def main():
    log(f"tracking URI: {mlflow.get_tracking_uri()}")
    c = MlflowClient()
    wait_for_server(c)

    try:
        mv = c.get_model_version_by_alias(NAME, ALIAS)
        log(f"model sudah terdaftar: {NAME} v{mv.version} (@{ALIAS}, stage {mv.current_stage}), dilewati")
        return
    except MlflowException:
        pass

    model_dir, meta_file = SRC / "model", SRC / "model_metadata.yaml"
    if not (model_dir / "MLmodel").exists() or not meta_file.exists():
        sys.exit(f"artefak tidak ditemukan di {SRC}. Jalankan 'dvc pull models/production.dvc' di host.")
    meta = yaml.safe_load(meta_file.read_text())
    src_version = meta["model"]["version"]
    lineage = meta.get("lineage", {})

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"seed-{NAME}-dagshub-v{src_version}") as run:
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        mlflow.log_artifact(str(meta_file))
        mlflow.log_params({k: str(v)[:250] for k, v in meta.get("params", {}).items()})
        mlflow.log_metrics({k: float(v) for k, v in meta.get("metrics_test", {}).items()})
        mlflow.set_tags({
            "seeded_from": "models/production (DVC, LK-07)",
            "source_registry": meta["mlflow"]["tracking_uri"],
            "source_version": str(src_version),
            "source_run_id": meta["mlflow"]["run_id"],
            "data_raw_md5": str(lineage.get("data_raw_md5", "")),
            "code_git_commit": str(lineage.get("code_git_commit", "")),
        })
    log(f"artefak diunggah ke mlflow-server (run {run.info.run_id})")

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", NAME)
    c.update_model_version(
        NAME, mv.version,
        description=(f"Salinan model Production v{src_version} dari DagsHub "
                     f"(run {meta['mlflow']['run_id']}), di-seed dari models/production.dvc."),
    )
    c.transition_model_version_stage(NAME, mv.version, "Production")
    c.set_registered_model_alias(NAME, ALIAS, mv.version)
    for k in ("source_version", "data_raw_md5", "code_git_commit"):
        val = {"source_version": str(src_version)}.get(k, str(lineage.get(k, "")))
        c.set_model_version_tag(NAME, mv.version, k, val)
    log(f"terdaftar: {NAME} v{mv.version} -> stage Production, alias @{ALIAS}")


if __name__ == "__main__":
    main()