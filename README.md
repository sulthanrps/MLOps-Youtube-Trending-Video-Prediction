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

- Ruff sebagai Python linter dan formatter.

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