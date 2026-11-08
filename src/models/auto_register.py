import argparse
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "youtube-trending-24h-classifier"
COPY_TAGS = ("data_raw_md5", "data_raw_nfiles", "git_commit", "trigger", "ci_run_url")


def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--name", default=MODEL_NAME)
    a = ap.parse_args()

    c = MlflowClient()
    run = c.get_run(a.run_id)
    if run.data.tags.get("validation_status") != "passed":
        sys.exit(f"Run {a.run_id} belum lolos validasi; registrasi dibatalkan.")

    mv = mlflow.register_model(f"runs:/{a.run_id}/model", a.name)
    v = mv.version
    tags = run.data.tags
    c.update_model_version(
        a.name, v,
        description=(f"Didaftarkan otomatis oleh CI dari commit {tags.get('git_commit', '?')} "
                     f"({run.info.run_name}). Log: {tags.get('ci_run_url', '-')}"),
    )
    for k in COPY_TAGS:
        if k in tags:
            c.set_model_version_tag(a.name, v, k, tags[k])
    c.set_model_version_tag(a.name, v, "validation_status", "passed")

    # Staging: hanya satu versi aktif; versi Staging sebelumnya otomatis diarsipkan
    c.transition_model_version_stage(a.name, v, "Staging", archive_existing_versions=True)
    c.set_registered_model_alias(a.name, "staging", v)
    c.set_model_version_tag(a.name, v, "staging_reason", "Lolos validasi otomatis CI (LK-08)")

    m = run.data.metrics
    text = (f"## 📦 Model terdaftar: `{a.name}` v{v} → **Staging**\n"
            f"- Run: `{run.info.run_name}` (`{a.run_id}`)\n"
            f"- Commit: `{tags.get('git_commit', '?')}`\n"
            f"- F1 {m.get('test_f1', 0):.3f} · Accuracy {m.get('test_accuracy', 0):.3f} · "
            f"ROC-AUC {m.get('test_roc_auc', 0):.3f}\n"
            f"- Promosi ke Production tetap melalui review manual.\n")
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"version={v}\n")


if __name__ == "__main__":
    main()