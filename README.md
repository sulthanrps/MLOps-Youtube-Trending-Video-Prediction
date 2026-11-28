# MLOps — Prediksi Keberlangsungan Video Trending YouTube Indonesia

Sistem MLOps yang memprediksi apakah video di **Top 50 Trending YouTube Indonesia** masih berada di daftar tersebut **sekitar 24 jam kemudian** (klasifikasi biner). Data diambil otomatis setiap 6 jam, di-versioning dengan DVC, model dilatih dan dilacak dengan MLflow, lalu versi terbaik dikelola melalui MLflow Model Registry.

| Komponen | Teknologi |
|---|---|
| Sumber data | YouTube Data API v3 (`chart=mostPopular`, `regionCode=ID`) |
| Otomatisasi | GitHub Actions (cron setiap 6 jam) |
| Versioning data & model | DVC dengan remote DagsHub |
| Eksperimen & registry | MLflow (server DagsHub) |
| Model | XGBoost (`XGBClassifier`) |
| Lingkungan | GitHub Codespaces, Python 3.11 |

---

## 🚀 Model yang Aktif untuk Inferensi

| | |
|---|---|
| **Nama model** | `youtube-trending-24h-classifier` |
| **Versi aktif** | **v2** — stage `Production`, alias `@champion` dan `@production` |
| **Run MLflow** | `3192df741dad4980aa6d0bbc83584685` |
| **Algoritma** | XGBoost: `n_estimators=150`, `max_depth=3`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8` |
| **Data latih** | 101 snapshot (4.957 baris, 1.335 video unik); train 3.137 baris hingga 22 Okt 2026 |
| **Metadata DVC** | `models/production.dvc` → `models/production/` (artefak + `model_metadata.yaml`), tag Git `model-v2` |

### Alasan v2 dipilih

v1 dan v2 dievaluasi pada **data test yang sama** (1.132 baris mulai 24 Okt 2026, belum pernah dilihat kedua model):

| Versi | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **v2 (Production)** | **0,743** | **0,667** | **0,627** | **0,646** | **0,784** |
| v1 (Archived) | 0,724 | 0,656 | 0,557 | 0,602 | 0,752 |

v2 unggul di semua metrik, terutama recall (+7 poin) dan F1 (+4,4 poin). F1 v1 turun dari 0,648 (saat dilatih di LK-06) menjadi 0,602 pada periode data yang lebih baru, sehingga model lama sudah tidak lagi mewakili pola trending terkini. v2 dilatih dengan data dua kali lebih banyak dan periode yang lebih baru.

### Riwayat versi

| Versi | Stage | Dilatih dengan | Keterangan |
|---|---|---|---|
| v2 | **Production** | 101 snapshot | Retraining LK-07, aktif untuk inferensi |
| v1 | Archived | 53 snapshot | Model terbaik LK-06, digantikan v2 |

### Memuat model

```python
import mlflow.pyfunc

# Dari MLflow Model Registry (butuh MLFLOW_TRACKING_URI dan kredensial DagsHub)
model = mlflow.pyfunc.load_model("models:/youtube-trending-24h-classifier/Production")
# atau dengan alias: "models:/youtube-trending-24h-classifier@champion"

pred = model.predict(X)   # X: DataFrame berisi 17 fitur (lihat models/production/model_metadata.yaml)
```

Atau dari salinan yang di-versioning DVC, tanpa akses ke server MLflow:

```bash
dvc pull models/production.dvc
```

```python
model = mlflow.pyfunc.load_model("models/production/model")   # folder yang berisi file MLmodel
```

Verifikasi cepat (memprediksi snapshot terbaru dan mengukur latensi):

```bash
python src/models/registry.py verify --stage Production
```

---

## 🧩 Menjalankan Seluruh Sistem dengan Docker Compose

Satu perintah menjalankan empat layanan dalam jaringan `mlops-net`: database metadata, MLflow Tracking Server + Model Registry, seeder model, dan API inferensi yang **menarik model dari MLflow Model Registry**.

| Layanan | Image | Port host | Peran |
|---|---|---|---|
| `mlflow-db` | `postgres:16-alpine` | – (internal) | Metadata experiment, run, dan registry |
| `mlflow-server` | `docker/mlflow/Dockerfile` | `5000` | MLflow UI, Model Registry, penyimpanan artefak |
| `model-seeder` | sama dengan `mlflow-server` | – | Sekali jalan: daftarkan `models/production/` ke registry (stage Production, alias `@champion`) |
| `api-service` | `Dockerfile` (root) | `8000` | REST API, memuat `models:/youtube-trending-24h-classifier@champion` dari `http://mlflow-server:5000` |
| `model-serving` ×3 | `youtube-trending-mlflow-serve` (`mlflow models build-docker`) | – (internal) | MLflow Models scoring server, 3 replika |
| `model-lb` | `nginx:1.27-alpine` | `8090` | Load balancer round-robin ke replika `model-serving` |

Urutan start dijaga dengan `depends_on` + healthcheck:
`mlflow-db` (healthy) → `mlflow-server` (healthy) → `model-seeder` (selesai) → `api-service`.

Data persisten disimpan di volume `mlflow-db-data` (PostgreSQL) dan `mlflow-artifacts` (file model), sehingga registry tetap ada setelah kontainer dimatikan.

### Prasyarat

```bash
docker compose version                 # Docker Compose v2
dvc pull models/production.dvc         # artefak model Production (LK-07)
```

Tambahkan kredensial database ke file `.env` di root repo (contoh di `.env.example`; file `.env` tidak ikut ter-commit):

```
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=<password huruf dan angka>
POSTGRES_DB=mlflow
```

### Menjalankan (satu perintah)

```bash
docker compose up -d --build
docker compose ps
```

Hasil yang diharapkan: `mlflow-db`, `mlflow-server`, dan `api-service` berstatus **Up (healthy)**, sedangkan `model-seeder` berstatus **Exited (0)** karena tugasnya sudah selesai.

- API: http://localhost:8000/docs
- MLflow UI: http://localhost:5000 (di Codespaces buka lewat tab **Ports**)

### Verifikasi komunikasi antar-layanan

```bash
docker compose logs model-seeder                    # model didaftarkan ke registry lokal
docker compose logs api-service | grep MODEL        # "model dimuat dari models:/...@champion ... via http://mlflow-server:5000"
curl http://localhost:8000/model-info               # "source": "mlflow-registry"
docker compose exec api-service getent hosts mlflow-server
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
python scripts/test_api.py                          # prediksi API identik dengan model lokal
```

### Menghentikan

```bash
docker compose down        # kontainer dan jaringan dihapus, volume (data registry) tetap ada
docker compose down -v     # hapus juga volume: registry dan artefak ikut terhapus
```

---

## ⚖️ Model Serving MLflow dan Horizontal Scaling

Selain API FastAPI, model Production juga dilayani langsung dengan **MLflow Models** (`mlflow models serve` / `mlflow models build-docker`) dan dijalankan sebagai **3 replika** di belakang load balancer Nginx.

```
klien -> model-lb (Nginx, :8090) --round-robin--> model-serving replika 1 / 2 / 3 (:8080)
```

### Menjalankan model serving secara lokal (tanpa Docker)

Stack Docker Compose harus berjalan agar `mlflow-server` (registry) dapat diakses di port 5000.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models serve -m "models:/youtube-trending-24h-classifier/Production" \
    -h 0.0.0.0 -p 5001 --env-manager local
```

### Membangun image serving dan menjalankan replika

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
mlflow models build-docker -m "models:/youtube-trending-24h-classifier/Production" \
    -n youtube-trending-mlflow-serve
docker compose up -d
docker compose ps model-serving model-lb
```

### Endpoint

| Method | URL | Fungsi |
|---|---|---|
| GET | `http://localhost:8090/ping` | Cek replika siap (dijawab salah satu replika) |
| GET | `http://localhost:8090/version` | Versi MLflow di image serving |
| POST | `http://localhost:8090/invocations` | Prediksi (load balancer → 3 replika) |
| GET | `http://localhost:8090/lb-health` | Status load balancer Nginx |
| POST | `http://localhost:5001/invocations` | Prediksi dari `mlflow models serve` lokal |

Setiap response dari port 8090 membawa header `X-Upstream-Addr` berisi IP replika yang menjawab.

Format body `/invocations` mengikuti MLflow (`dataframe_split`), berisi 17 fitur dengan urutan kolom pada `models/production/model_metadata.yaml`:

```bash
python scripts/make_serving_payload.py          # sample_request.json -> sample_serving_request.json
curl -i -X POST http://localhost:8090/invocations \
     -H "Content-Type: application/json" -d @sample_serving_request.json
```

Response: `{"predictions": [0, 0, 1, ...]}`, dengan 1 = masih di Top 50 sekitar 24 jam kemudian, 0 = keluar.

### Uji beban

```bash
python scripts/load_test.py -n 300 -c 20
```

Melaporkan throughput, latensi p50/p95, sebaran request per replika, dan memastikan semua replika memberi prediksi yang sama.

### Menambah atau mengurangi replika secara dinamis

```bash
docker compose up -d --scale model-serving=5 --no-recreate   # naik ke 5 replika
docker compose restart model-lb                              # Nginx membaca ulang daftar replika
docker compose ps model-serving

docker compose up -d --scale model-serving=2                 # turun ke 2 replika
docker compose restart model-lb
```

Opsi `--scale` menimpa nilai `deploy.replicas: 3` di `docker-compose.yaml` hanya untuk eksekusi tersebut. Untuk mengubah jumlah default secara permanen, ubah nilai `replicas` di file. Nginx me-resolve nama `model-serving` saat start, sehingga perlu di-restart setelah jumlah replika berubah.

---

## 🐳 Menjalankan Layanan Inferensi dengan Docker (satu kontainer)

Model Production dikemas sebagai Docker Image berisi REST API (FastAPI). Kontainer berjalan mandiri tanpa koneksi ke DagsHub atau MLflow.

### Prasyarat

- Docker sudah terpasang (cek dengan `docker --version`).
- Artefak model sudah diunduh dari DVC:

```bash
dvc pull models/production.dvc
ls models/production/model/      # harus ada model.pkl, MLmodel, requirements.txt
```

### Build dan run

```bash
docker build -t youtube-trending-api:v2 .
docker run -d --name yt-api -p 8000:8000 youtube-trending-api:v2
docker ps                          # STATUS: Up ... (healthy)
```

### Endpoint

| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | Status layanan |
| GET | `/model-info` | Versi model, metrik, silsilah data, dan daftar 17 fitur |
| POST | `/predict` | Prediksi untuk 1–500 video |
| GET | `/docs` | Dokumentasi interaktif (Swagger UI) |

### Contoh request

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

Format body `/predict`: setiap elemen `instances` berisi 17 fitur hasil feature engineering, ditambah `video_id` opsional:

```json
{
  "instances": [
    {
      "video_id": "abc123",
      "view_count_log": 12.3, "like_count_log": 8.1, "comment_count_log": 5.2,
      "video_age_hours": 30.5, "views_per_hour_lifetime": 7200.0,
      "like_ratio": 0.02, "comment_ratio": 0.001,
      "view_growth_per_hour": 1500.0, "like_growth_per_hour": 30.0, "comment_growth_per_hour": 2.0,
      "hours_since_prev": 6.0, "is_first_appearance": 0, "appearances_so_far": 4,
      "hours_since_first_seen": 18.0, "category_id": 10, "publish_hour_wib": 19, "publish_dayofweek": 4
    }
  ]
}
```

Response berisi `prediction` (1 = bertahan, 0 = keluar), `label`, dan `probability` untuk setiap video.

### Validasi prediksi kontainer

Membandingkan prediksi API dengan model Production lokal. Hasilnya harus identik:

```bash
python scripts/test_api.py                  # pakai sample_request.json
python scripts/test_api.py --refresh -n 10  # buat sampel baru dari snapshot terbaru (butuh data DVC)
```

### Menghentikan kontainer

```bash
docker stop yt-api && docker rm yt-api
```

---

## 📁 Struktur Repositori

```
.
├── docker-compose.yaml         # orkestrasi: mlflow-db, mlflow-server, model-seeder, api-service
├── .env.example                # contoh kredensial PostgreSQL untuk Docker Compose
├── app.py                      # REST API inferensi (FastAPI)
├── Dockerfile                  # image layanan inferensi
├── .dockerignore
├── requirements-serving.txt    # dependensi API (versi di-pin)
├── sample_request.json         # contoh body POST /predict
├── .github/workflows/
│   └── ingestion-preprocessing_pipeline.yml   # ingestion otomatis + versioning data tiap 6 jam
├── config/
├── docker/mlflow/Dockerfile    # image MLflow server (backend PostgreSQL)
├── docker/nginx/nginx.conf     # load balancer replika model-serving
├── data/
│   ├── raw.dvc                 # pointer DVC data mentah
│   └── processed.dvc           # pointer DVC data hasil feature engineering
├── models/
│   └── production.dvc          # pointer DVC model Production + metadata
├── notebooks/
├── scripts/
│   ├── migrate_artifacts.py    # migrasi snapshot dari GitHub Actions artifacts
│   ├── seed_registry.py        # isi registry lokal dengan model Production (Docker Compose)
│   ├── make_serving_payload.py # buat body request format MLflow /invocations
│   ├── load_test.py            # uji beban ke replika model-serving
│   └── test_api.py             # uji /predict dan bandingkan dengan model lokal
├── src/
│   ├── data/ingestion.py       # ambil Top 50 Trending dari YouTube Data API
│   ├── features/build_features.py
│   └── models/
│       ├── train.py            # training + MLflow tracking
│       └── registry.py         # manajemen Model Registry (register, compare, promote, export, verify)
├── tests/
└── requirements.txt
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Buat file `.env` di root repo (tidak ikut ter-commit):

```
YOUTUBE_API_KEY=<api key>
MLFLOW_TRACKING_URI=https://dagshub.com/sulthanrps/MLOps-Youtube-Trending-Video-Prediction.mlflow
MLFLOW_TRACKING_USERNAME=sulthanrps
MLFLOW_TRACKING_PASSWORD=<token DagsHub>
```

Kredensial DVC (disimpan di `.dvc/config.local`, tidak ikut ter-commit):

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <token DagsHub>
```

---

## 📦 Versioning Data (DVC)

File Parquet disimpan di storage DVC DagsHub; Git hanya menyimpan pointer `.dvc`.

| Branch | Isi |
|---|---|
| `main` | Kode, pointer model, dan versi data milestone (tag `data-v1`, `data-v2`) |
| `data-updates` | Pointer data terbaru yang di-commit otomatis oleh GitHub Actions setiap 6 jam |

Mengambil data terbaru tanpa berpindah dari `main`:

```bash
git fetch origin
git checkout origin/data-updates -- data/raw.dvc data/processed.dvc
dvc pull data/raw.dvc data/processed.dvc
# ... analisis / training ...
git restore --staged --worktree data/raw.dvc data/processed.dvc
```

Audit perbedaan versi data:

```bash
dvc diff data-v1 data-v2
git log --oneline -- data/raw.dvc
```

---

## 🧪 Training dan Pembaruan Model

```bash
# 1. Latih model (hasil tercatat di MLflow DagsHub)
python src/models/train.py --model xgboost --n-estimators 150 --max-depth 3 --learning-rate 0.08
python src/models/train.py --grid                       # 5 konfigurasi sekaligus

# 2. Daftarkan run terbaik sebagai versi baru
python src/models/registry.py register --run-id <run_id> --description "<alasan>"

# 3. Bandingkan dengan versi Production pada data test yang sama
python src/models/registry.py compare --versions <lama> <baru>

# 4. Promosikan pemenang: None -> Staging -> Production; arsipkan versi lama
python src/models/registry.py promote --version <baru> --stage Staging
python src/models/registry.py promote --version <baru> --stage Production --reason "<alasan>"
python src/models/registry.py promote --version <lama> --stage Archived

# 5. Catat model Production ke DVC
python src/models/registry.py export --stage Production
dvc add models/production
git add models/production.dvc && git commit -m "model: production v<baru>"
git tag -a model-v<baru> -m "Model Production v<baru>"
dvc push models/production.dvc && git push origin main --tags

# 6. Verifikasi kesiapan inferensi
python src/models/registry.py verify --stage Production
```

Commit kode sebelum training agar tag `git_commit` di MLflow menunjuk ke kode yang benar-benar dipakai.

---

## 🏷️ Tag Git

| Tag | Isi |
|---|---|
| `data-v1` | Dataset awal: 24 snapshot (LK-04 + migrasi artifacts) |
| `data-v2` | Dataset v2: 25 snapshot (simulasi continual learning, LK-05) |
| `model-v2` | Model Production v2 tercatat via DVC (LK-07) |