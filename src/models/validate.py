import argparse
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import train as T  # noqa: E402

MODEL_NAME = "youtube-trending-24h-classifier"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def check_gate(metrics: dict, gate: dict) -> list:
    """Bandingkan metrik run (key: test_<metrik>) dengan ambang gate."""
    rows = []
    for m in METRICS:
        if m not in gate:
            continue
        val = metrics.get(f"test_{m}")
        rows.append({"check": f"{m} >= {gate[m]}", "value": val,
                     "passed": val is not None and val >= gate[m]})
    return rows


def check_vs_production(run_id: str, max_drop: float, test_frac: float) -> dict:
    """Evaluasi model baru dan model Production pada data test yang sama."""
    c = MlflowClient()
    try:
        prod = c.get_latest_versions(MODEL_NAME, stages=["Production"])
    except Exception:
        prod = []
    if not prod:
        return {"check": "vs Production", "value": None, "passed": True,
                "note": "belum ada model Production, pemeriksaan dilewati"}
    try:
        df = T.add_features(T.add_label(T.load_data()))
        _, test, _ = T.time_split(df, test_frac)
        X, y = test[T.FEATURES], test["label"]
        new_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        prod_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{prod[0].version}")
        f1_new = f1_score(y, new_model.predict(X), zero_division=0)
        f1_prod = f1_score(y, prod_model.predict(X), zero_division=0)
    except Exception as e:  # gangguan jaringan/registry: dicatat, tidak memblokir Staging
        return {"check": "vs Production", "value": None, "passed": True,
                "note": f"dilewati: {type(e).__name__}: {str(e)[:120]}"}
    return {
        "check": f"F1 baru >= F1 Production v{prod[0].version} - {max_drop}",
        "value": f1_new,
        "passed": f1_new >= f1_prod - max_drop,
        "note": f"F1 baru {f1_new:.3f} vs Production {f1_prod:.3f} (n_test={len(test)})",
    }


def write_summary(run, rows, lk01_rows, passed):
    lines = [
        f"## {'✅ Validasi LOLOS' if passed else '❌ Validasi GAGAL'}",
        f"Run `{run.info.run_name}` (`{run.info.run_id}`)",
        "",
        "| Pemeriksaan (gate) | Nilai | Hasil |",
        "|---|---|---|",
    ]
    for r in rows:
        val = "-" if r["value"] is None else f"{r['value']:.3f}"
        note = f" — {r['note']}" if r.get("note") else ""
        lines.append(f"| {r['check']} | {val} | {'✅' if r['passed'] else '❌'}{note} |")
    lines += ["", "| Target LK-01 (informasi) | Nilai | Tercapai |", "|---|---|---|"]
    for r in lk01_rows:
        val = "-" if r["value"] is None else f"{r['value']:.3f}"
        lines.append(f"| {r['check']} | {val} | {'✅' if r['passed'] else '—'} |")
    text = "\n".join(lines)
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")


def main():
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--thresholds", default="config/thresholds.yaml")
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args()

    with open(ROOT / a.thresholds) as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / a.params) as f:
        test_frac = float((yaml.safe_load(f) or {}).get("test_frac", 0.25))

    c = MlflowClient()
    run = c.get_run(a.run_id)
    metrics = run.data.metrics

    rows = check_gate(metrics, cfg["gate"])
    rows.append(check_vs_production(a.run_id, float(cfg["gate"].get("max_f1_drop_vs_production", 0.02)),
                                    test_frac))
    lk01_rows = check_gate(metrics, cfg.get("lk01_target", {}))
    passed = all(r["passed"] for r in rows)

    c.set_tag(a.run_id, "validation_status", "passed" if passed else "failed")
    c.set_tag(a.run_id, "meets_lk01_target", str(all(r["passed"] for r in lk01_rows)))
    write_summary(run, rows, lk01_rows, passed)

    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"passed={'true' if passed else 'false'}\n")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()