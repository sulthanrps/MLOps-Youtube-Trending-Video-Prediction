# MLOps YouTube Trending

Proyek MLOps untuk memprediksi apakah video yang sedang berada dalam Top 50 trending YouTube di Indonesia akan tetap berada dalam Top 50 dalam 24 jam ke depan berdasarkan data historis dan data trending yang diperoleh secara berkala dari YouTube Data API.

Sistem juga dirancang untuk memberikan insight bagi content creator mengenai kategori dan waktu upload yang memiliki kecenderungan performa lebih baik berdasarkan pola historis video yang berhasil bertahan dalam Top 50 trending.

Hasil prediksi dan rekomendasi bersifat eksperimental dan tidak menjamin performa aktual sebuah video di YouTube.

## Tujuan Proyek

Proyek ini bertujuan untuk:

1. Mengembangkan model machine learning untuk mengklasifikasikan apakah video yang saat ini berada dalam Top 50 trending YouTube Indonesia akan tetap berada dalam Top 50 dalam 24 jam ke depan.
2. Menggunakan data YouTube seperti jumlah views, likes, comments, waktu upload, kategori video, serta perubahan statistik video dari beberapa snapshot waktu.
3. Mengumpulkan data trending secara berkala menggunakan YouTube Data API untuk membangun dataset historis dan mensimulasikan kondisi data yang dinamis dalam lingkungan production.
4. Membangun pipeline MLOps yang mencakup data ingestion, feature engineering, training, inference, evaluasi, monitoring, dan retraining model.
5. Menerapkan continual learning dengan melakukan evaluasi performa model secara berkala dan melakukan retraining apabila performa model berada di bawah threshold yang telah ditentukan.
6. Menyediakan insight mengenai kategori video dan waktu upload yang memiliki kecenderungan lebih baik berdasarkan video yang mampu bertahan dalam Top 50 trending.
7. Menyediakan lingkungan pengembangan yang reproducible menggunakan GitHub Codespaces.

Data trending diperoleh dari YouTube Data API menggunakan chart `mostPopular` dengan wilayah Indonesia (`regionCode=ID`). Data dikumpulkan dalam bentuk snapshot secara berkala sehingga video yang sama dapat diamati perubahan statistik dan posisi trending-nya dari waktu ke waktu.

## Struktur Direktori

Repository disusun menggunakan struktur proyek data science dan MLOps agar data, model, konfigurasi, eksperimen, machine learning pipeline, backend, dan frontend tersimpan secara terpisah.

```text
MLOps-YouTube-Trending/
├── .devcontainer/             Konfigurasi lingkungan GitHub Codespaces
│   └── devcontainer.json
├── backend/                   Backend API untuk mengakses model dan hasil prediksi
├── config/                    Konfigurasi proyek dan parameter pipeline
│   └── config.yaml
├── data/
│   ├── raw/                   Data mentah dari YouTube Data API
│   ├── processed/             Data yang telah diproses dan siap digunakan
│   └── external/              Dataset eksternal yang digunakan dalam proyek
├── frontend/                  Dashboard web untuk menampilkan hasil prediksi dan insight
├── models/                    Artifact model machine learning
├── notebooks/                 Notebook EDA dan eksperimen model
├── src/
│   ├── __init__.py
│   ├── data/                  Data ingestion dan pengolahan data
│   │   ├── __init__.py
│   │   └── ingestion.py
│   ├── features/              Feature engineering dan persiapan fitur
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                Training, inference, dan evaluasi model
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── monitoring/            Monitoring data, model, dan pipeline
│       ├── __init__.py
│       └── monitor.py
├── tests/                     Pengujian unit dan integrasi
├── .gitignore                 mengecualikan file yg tidak masuk git
├── LICENSE                    MIT License
├── README.md                  Dokumentasi utama proyek
└── requirements.txt           Dependency Python proyek
```

## Menjalankan Proyek dengan GitHub Codespaces

1. Buka repository ini di GitHub.

2. Klik tombol Code.

3. Pilih tab Codespaces.

4. Klik Create codespace on main.

5. Tunggu proses pembuatan container dan postCreateCommand selesai.

### Konfigurasi .devcontainer/devcontainer.json akan menyiapkan:

- Python 3.11;

- Git;

- dependency Python dari requirements.txt;

- ekstensi Python;

- Pylance;

- Jupyter;

- Ruff sebagai Python linter dan formatter;

- pandas;

- pyarrow;

- python-dotenv.

### Verifikasi Environment
Setelah Codespace selesai dibuat, buka terminal dan jalankan:

```
python --version
pip --version
```
Versi utama Python yang diharapkan:

```text
Python 3.11.x
```

Pastikan dependency utama dapat digunakan dengan menjalankan perintah berikut

```
python -c "import pandas, numpy, sklearn, requests; print('Environment OK')"
```

Apabila konfigurasi berhasil, terminal akan menampilkan:

```
Environment OK
```

### Persiapan Kredensial (Wajib)
Proyek ini membutuhkan kunci API YouTube Data v3 untuk menarik data.

**Untuk Eksekusi Manual (Lokal / Codespaces):**

1. Buat file baru bernama .env di root direktori proyek (sejajar dengan file README.md).

2. Masukkan API Key ke dalam file tersebut dengan format berikut:
    ```
    YOUTUBE_API_KEY=masukkan_api_key_anda_disini
    ```
    (Catatan: File .env sudah dimasukkan ke dalam .gitignore sehingga aman dan tidak akan ikut ter-push ke repository).

**Untuk Eksekusi Otomatis (Github Actions):**
Agar otomatisasi di server GitHub dapat berjalan, kamu harus memasukkan API Key ke dalam GitHub Secrets:

1. Masuk ke halaman repositori GitHub.

2. Buka **Settings > Secrets and variables > Actions**.

3. Klik **New repository secret**.

4. Isi kolom Name dengan YOUTUBE_API_KEY dan masukkan kuncinya di kolom Secret.

### Menjalankan Data Ingestion dan Preprocessing Otomatis
`A. Manual`
1. Data Ingestion 
    - Di terminal, masuk ke directory src/data/
    - Lalu ketikkan perintah
        ```
        python ingestion.py
        ```
    - Jika berhasil, maka akan muncul pesan 
        ```
        [SUCCESS] Tersimpan ke: data/raw/raw_trending_[time_stamp]_[hourandminutes].parquet
        ```
2. Preprocessing Data Raw 
    - Selanjutnya, untuk melakukan preprocessing terhadap data raw yang sudah diambil, masuk ke directory src/features
    - Lalu ketikkan perintah
        ```
        python build_features.py
        ```
    - Jika berhasil, maka akan muncul pesan 
        ```
        [SUCCESS] Fitur tersimpan ke: data/processed/processed_trending_[time_stamp]_[hourandminutes].parquet
        ```
`B. Otomatis dengan Github Action`

1. Pilih tab Actions di laman repository ini di github 

2. Pilih workflow Data Ingestion & Preprocessing Pipeline di menu sebelah kiri.

3. Klik Run workflow di sebelah kanan.

4. Pantau prosesnya. Jika indikatornya hijau (✅), maka pipeline akan secara otomatis  bekerja secara mandiri setiap 6 jam.

## Data Versioning dengan DVC

Proyek ini menggunakan [DVC](https://dvc.org/) untuk melacak versi dataset Top 50 YouTube Trending Indonesia. File Parquet **tidak disimpan di Git**. Git hanya menyimpan file penunjuk kecil (`.dvc`) berisi hash, ukuran, dan jumlah file. Data fisiknya disimpan di remote storage **DagsHub**.

### Struktur

```
data/
├── .gitignore        # dibuat otomatis oleh DVC, mengabaikan /raw dan /processed
├── raw.dvc           # pointer versi data mentah (dilacak Git)
├── processed.dvc     # pointer versi data hasil feature engineering (dilacak Git)
├── raw/              # *.parquet hasil src/data/ingestion.py (dilacak DVC)
└── processed/        # *.parquet hasil src/features/build_features.py (dilacak DVC)
```

Remote DVC: `https://dagshub.com/sulthanrps/MLOps-Youtube-Trending-Video-Prediction
.dvc` (nama remote: `origin`)

### Setup awal (sekali saja)

```bash
pip install dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <DAGSHUB_TOKEN>
dvc pull                       # unduh data sesuai pointer .dvc saat ini
```

Kredensial disimpan di `.dvc/config.local` dan tidak ikut ter-commit.

### Alur penambahan versi data (continual learning)

Setiap ingestion menambahkan snapshot baru ke `data/raw/`. Alur merekam versi barunya:

```bash
# 1. Ambil data terbaru dari YouTube Data API v3
python src/data/ingestion.py
python src/features/build_features.py

# 2. Cek perubahan yang belum direkam
dvc status

# 3. Rekam versi baru → hash di file .dvc berubah
dvc add data/raw data/processed

# 4. Commit pointer ke Git dan beri tag versi
git add data/raw.dvc data/processed.dvc
git commit -m "data: dataset vN"
git tag -a data-vN -m "Dataset vN"

# 5. Upload data fisik ke DagsHub
dvc push
```

### Audit dan perbandingan versi

```bash
dvc diff data-v1 data-v2              # file yang ditambah/diubah/dihapus antarversi
dvc diff --show-hash data-v1 data-v2  # beserta hash tiap file
git log --oneline -- data/raw.dvc     # riwayat (silsilah) versi data
dvc status -c                         # cek sinkronisasi cache lokal vs DagsHub
```

### Kembali ke versi data lama

```bash
git checkout data-v1 -- data/raw.dvc
dvc checkout                          # data/raw kembali ke isi versi v1

git checkout HEAD -- data/raw.dvc     # kembali ke versi terbaru
dvc checkout
```

### Otomatisasi (GitHub Actions)

Workflow `.github/workflows/ingestion-preprocessing_pipeline.yml` berjalan setiap 6 jam:

1. Checkout `main`, lalu tarik pointer data terakhir dari branch `data-updates` dan jalankan `dvc pull`.
2. Jalankan ingestion dan feature engineering (snapshot baru ditambahkan ke data lama).
3. `dvc add` + `dvc push` ke DagsHub.
4. Commit pointer `.dvc` ke branch **`data-updates`** (bukan `main`), supaya `main` tetap bersih dari bot-commit.

| Branch | Isi |
|---|---|
| `main` | Kode program (pipeline, training, dashboard, tests) dan versi data yang ditandai manual (`data-v1`, `data-v2`, ...) |
| `data-updates` | Riwayat otomatis pointer `.dvc` dari bot setiap 6 jam |

Saat retraining, pipeline training meminjam pointer data terbaru tanpa merge:

```bash
git checkout origin/data-updates -- data/raw.dvc data/processed.dvc
dvc pull
python src/models/train.py
git restore data/raw.dvc data/processed.dvc
```

### Riwayat versi dataset

| Tag | Keterangan | Jumlah file | Jumlah baris |
|---|---|---|---|
| `data-v1` | Dataset awal hasil LK-04 | 1190 | 24 |
| `data-v2` | Tambahan snapshot (simulasi continual learning, LK-05) | 1239 | 25 |