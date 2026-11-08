import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import train as T
from validate import check_gate

ROOT = Path(__file__).resolve().parents[1]
T0 = pd.Timestamp("2026-10-01 00:00")


def snapshots(rows):
    """rows: list (jam_sejak_T0, video_id, view_count) -> DataFrame mirip hasil load_data()."""
    recs = []
    for h, vid, views in rows:
        recs.append({
            "snapshot_timestamp": T0 + pd.Timedelta(hours=h),
            "video_id": vid,
            "published_at": T0 - pd.Timedelta(hours=10),
            "view_count": float(views),
            "like_count": float(views) * 0.02,
            "comment_count": float(views) * 0.001,
            "category_id": 10,
        })
    return pd.DataFrame(recs)


def row(df, h, vid):
    return df[(df["snapshot_timestamp"] == T0 + pd.Timedelta(hours=h)) & (df["video_id"] == vid)].iloc[0]


# ---------------------------------------------------------------- label
def test_label_1_jika_video_masih_ada_24_jam_kemudian():
    out = T.add_label(snapshots([(0, "A", 100), (0, "B", 100), (24, "A", 200)]))
    assert row(out, 0, "A")["label"] == 1.0
    assert row(out, 0, "B")["label"] == 0.0


def test_label_kosong_jika_tidak_ada_snapshot_di_jendela_20_28_jam():
    out = T.add_label(snapshots([(0, "A", 100), (30, "A", 200)]))
    assert out["label"].isna().all()


def test_label_memakai_snapshot_terdekat_ke_24_jam():
    # kandidat 21 jam (A ada) dan 25 jam (A tidak ada); terdekat ke 24 jam adalah 25 jam
    out = T.add_label(snapshots([(0, "A", 1), (21, "A", 2), (25, "B", 3)]))
    assert row(out, 0, "A")["label"] == 0.0


# ---------------------------------------------------------------- fitur
def test_fitur_lengkap_dan_tanpa_nan():
    df = snapshots([(0, "A", 100), (6, "A", 160), (6, "B", 50), (12, "B", 90)])
    out = T.add_features(T.add_label(df))
    assert len(T.FEATURES) == 17
    assert set(T.FEATURES) <= set(out.columns)
    assert not out[T.FEATURES].isna().any().any()


def test_pertumbuhan_dinormalisasi_per_jam():
    out = T.add_features(snapshots([(0, "A", 100), (4, "A", 500)]))
    assert row(out, 4, "A")["view_growth_per_hour"] == pytest.approx(100.0)
    assert row(out, 0, "A")["is_first_appearance"] == 1


def test_fitur_tidak_memakai_data_masa_depan():
    base = snapshots([(0, "A", 100), (6, "A", 160), (12, "A", 250)])
    future_changed = base.copy()
    future_changed.loc[future_changed["snapshot_timestamp"] == T0 + pd.Timedelta(hours=12), "view_count"] = 9e9
    f_base = T.add_features(base)
    f_changed = T.add_features(future_changed)
    past = f_base["snapshot_timestamp"] <= T0 + pd.Timedelta(hours=6)
    pd.testing.assert_frame_equal(f_base.loc[past, T.FEATURES], f_changed.loc[past, T.FEATURES])


# ---------------------------------------------------------------- split
def test_split_waktu_dengan_purge_28_jam():
    # 40 snapshot tiap 6 jam; setiap video bertahan 8 snapshot (48 jam) -> ada label 0 dan 1
    rows = [(s * 6, f"v{j}", 1000 + s) for s in range(40) for j in range(s, s + 8)]
    df = T.add_features(T.add_label(snapshots(rows)))
    train, test, split = T.time_split(df, 0.25)
    assert train["snapshot_timestamp"].max() <= split - pd.Timedelta(hours=T.LABEL_MAX_H)
    assert test["snapshot_timestamp"].min() >= split
    assert train["label"].nunique() == 2


# ---------------------------------------------------------------- validasi
def test_gate_lolos_dan_gagal():
    gate = {"f1": 0.60, "recall": 0.55}
    res = {r["check"]: r["passed"] for r in check_gate({"test_f1": 0.65, "test_recall": 0.50}, gate)}
    assert res == {"f1 >= 0.6": True, "recall >= 0.55": False}


def test_gate_gagal_jika_metrik_tidak_ada():
    assert check_gate({}, {"f1": 0.6})[0]["passed"] is False


# ---------------------------------------------------------------- konsistensi API
def test_skema_api_sama_dengan_fitur_training():
    app_py = ROOT / "app.py"
    if not app_py.exists():
        pytest.skip("app.py tidak ditemukan")
    tree = ast.parse(app_py.read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "VideoFeatures")
    fields = {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)} - {"video_id"}
    assert fields == set(T.FEATURES)