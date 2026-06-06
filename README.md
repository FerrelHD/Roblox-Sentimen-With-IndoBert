# 🎓 Analisis Sentimen Review Roblox (Skripsi)

Project ini adalah sistem end-to-end untuk analisis sentimen review Roblox dari Google Play Store (Bahasa Indonesia), meliputi:
- Scraping data → preprocessing teks → baseline ML (SVM) → evaluasi & visualisasi di Streamlit
- Deep Learning (IndoBERT) + **auto-labeling (Teacher HF) → fine-tuning otomatis** untuk memahami pola teks (tanpa bergantung rating)

## 📌 Ringkasan Cepat
- Entry point dashboard: `app.py`
- Baseline ML: `train_model.py` + `sentiment.py` (hasil: `data/processed/roblox_sentiment.csv`)
- IndoBERT:
  - Model aktif (hasil fine-tuning otomatis): `models/indobert_sentiment/best_model_annotated/`
  - Baseline fallback (jika tersedia): `models/indobert_sentiment/best_model/`
- Fokus utama dashboard: tren sentimen berbasis waktu + pola teks review (bukan sekadar rating)

## 📖 Daftar Isi
- [Panduan Penulisan Skripsi](#-panduan-penulisan-skripsi)
- [Tujuan](#-tujuan)
- [Fitur](#-fitur)
- [Struktur Folder](#-struktur-folder)
- [Teknologi](#-teknologi)
- [Instalasi](#-instalasi)
- [Menjalankan Dashboard](#-menjalankan-dashboard)
- [Alur Kerja](#-alur-kerja)
- [Model & Output](#-model--output)
- [Troubleshooting](#-troubleshooting)

## 📝 Panduan Penulisan Skripsi
Berikut adalah panduan struktur bab skripsi berdasarkan isi proyek ini.

### Bab 1: Pendahuluan
- Latar belakang: kebutuhan analisis sentimen review aplikasi, khususnya Roblox di Google Play Store.
- Rumusan masalah: bagaimana mengidentifikasi sentimen review menggunakan pendekatan machine learning dan deep learning.
- Tujuan penelitian: membangun sistem end-to-end untuk analisis sentimen, membandingkan SVM dan IndoBERT, serta membuat dashboard visualisasi.
- Manfaat penelitian: membantu pengembang aplikasi memahami opini pengguna, mendeteksi tren sentimen, dan meningkatkan kualitas aplikasi.
- Ruang lingkup: scraping review, preprocessing teks, baseline SVM, IndoBERT dengan auto-labeling, analisis perbandingan, dan dashboard.

### Bab 2: Tinjauan Pustaka
- Definisi analisis sentimen dan aplikasi dalam review aplikasi mobile.
- Teori NLP untuk Bahasa Indonesia, termasuk tokenisasi dan TF-IDF.
- Machine learning untuk klasifikasi teks: SVM, TF-IDF, dan perbandingan dengan model deep learning.
- Model Bahasa Indonesia: IndoBERT dan manfaat transfer learning untuk sentiment analysis.
- Konsep auto-labeling teacher-student: menggunakan model teacher untuk memberi label awal dan melatih model student.
- Studi sebelumnya tentang sentiment analysis untuk aplikasi atau game mobile (jika ada, bisa ditambahkan dari jurnal atau artikel).

### Bab 3: Metodologi Penelitian
- Dataset: sumber data dari Google Play Store, deskripsi `data/raw/roblox_raw.csv` dan file backup `data/raw/roblox_raw_backup_*.csv`.
- Scraping: proses `scraping.py` untuk mengumpulkan review.
- Preprocessing: proses `preprocessing.py` untuk membersihkan teks dan menyimpan `data/processed/roblox_cleaned.csv`.
- Baseline ML: pelatihan SVM + TF-IDF di `train_model.py`, lalu klasifikasi pada `sentiment.py` menghasilkan `data/processed/roblox_sentiment.csv`.
- Deep learning IndoBERT: pelatihan `train_indobert.py` dengan auto-labeling teacher, model tersimpan di `models/indobert_sentiment/best_model_annotated/`.
- Evaluasi: metrik model di `models/indobert_sentiment/best_model_annotated/metrics.json`, metode perbandingan rating di `sentiment_comparison_analysis.py`.
- Implementasi dashboard: `app.py` sebagai antarmuka visualisasi hasil, dan `predict_indobert.py` untuk prediksi teks real-time.
- Tools dan lingkungan: Python, Streamlit, Pandas, scikit-learn, PyTorch, HuggingFace Transformers.

### Bab 4: Hasil dan Pembahasan
- Hasil preprocessing: jumlah data bersih dan karakteristik teks setelah pembersihan.
- Hasil baseline SVM: akurasi, confusion matrix, distribusi sentimen pada `data/processed/roblox_sentiment.csv`.
- Hasil IndoBERT: metrik fine-tuning di `metrics.json`, perbandingan dengan model baseline.
- Analisis perbandingan rating: hasil `sentiment_comparison_full.csv` dan `sentiment_comparison_summary.csv` untuk membandingkan label model dengan rating asli.
- Visualisasi dan insight: tampilkan grafik dari dashboard `app.py` atau output `data/processed/sentiment_analysis/` jika tersedia.
- Diskusi: kelebihan dan kekurangan pendekatan, kasus disagreement (contoh dari `sentiment_disagreement_examples.csv`), serta rekomendasi perbaikan.
- KESIMPULAN awal: fokus pada efektivitas auto-labeling IndoBERT dibanding SVM dan nilai praktis dashboard.

## 🎯 Tujuan
- Mengklasifikasikan sentimen review Roblox: **positif / netral / negatif**
- Membandingkan beberapa pendekatan:
  - Baseline **SVM + TF‑IDF**
  - Deep Learning **IndoBERT**
- Menyajikan dashboard interaktif untuk filter waktu/versi aplikasi, visualisasi tren, evaluasi model, dan prediksi real-time
- Meningkatkan kualitas IndoBERT untuk memahami **pola teks** melalui **auto-labeling teacher → fine-tuning student** (tanpa label rating)

## ✨ Fitur
### Dashboard Streamlit
- Overview (ringkasan & kesimpulan)
- Analisis Sentimen (distribusi sentimen, per versi aplikasi, top kata/frasa dari data)
- Tren Temporal (harian/bulanan + puncak positif/negatif + pola teks penyebab)
- Model & Evaluasi:
  - Evaluasi SVM
  - IndoBERT (hasil fine-tuning otomatis + tombol auto fine-tuning)
  - Perbandingan rating (opsional, hanya pembanding)
- Scraping & Preprocessing (alur kerja project yang terbaru)

### Prediksi IndoBERT Real-time
- Input teks → prediksi sentimen + probabilitas
- Jika model fine-tuned otomatis tersedia, sistem otomatis memakai `best_model_annotated` (fallback ke `best_model`)

## 📁 Struktur Folder

```
Skripsi-Roblox-main/
├── app.py
├── scraping.py
├── preprocessing.py
├── analysis.py
├── train_model.py
├── sentiment.py
├── train_indobert.py
├── predict_indobert.py
├── sentiment_comparison_analysis.py
├── generate_report.py
├── presentation_narrative.py
├── requirements.txt
├── models/
│   ├── sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── indobert_sentiment/
│       ├── best_model/                 # baseline IndoBERT (checkpoint)
│       ├── best_model_annotated/       # hasil fine-tuning otomatis (auto-labeling)
│       ├── confusion_matrix.png        # (opsional) output training baseline
│       └── ...                         # tokenizer/config
└── data/
    ├── raw/
    │   ├── roblox_raw.csv
    │   └── roblox_raw_backup_*.csv
    └── processed/
        ├── roblox_cleaned.csv
        ├── roblox_sentiment.csv
        ├── sentiment_analysis_summary.csv   # ringkasan tren bulanan (opsional)
        ├── sentiment_crosstab.csv
        ├── sentiment_comparison_full.csv        # perbandingan rating vs model (opsional)
        ├── sentiment_comparison_summary.csv     # ringkasan perbandingan (opsional)
        ├── sentiment_disagreement_examples.csv  # contoh disagreement (opsional)
        └── sentiment_analysis/          # output laporan/grafik (opsional)
            └── ...
```

## 🧰 Teknologi
- Bahasa: Python
- Dashboard: Streamlit
- Visualisasi: Plotly, Matplotlib, Seaborn
- Data: Pandas, NumPy
- ML baseline: scikit-learn (TF-IDF + SVM), joblib
- Deep Learning: PyTorch, HuggingFace Transformers
- Utility: tqdm

## ⚙️ Instalasi
### Prasyarat
- Python 3.10+ (disarankan 3.11)
- Virtual environment (venv)

### Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Verifikasi GPU (opsional):
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 Menjalankan Dashboard
```bash
streamlit run app.py
```

## 🔄 Alur Kerja

### 1) Scraping
```bash
python scraping.py
```
Output: `data/raw/roblox_raw.csv`

### 2) Preprocessing
```bash
python preprocessing.py
```
Output: `data/processed/roblox_cleaned.csv`

### 3) Baseline ML (SVM + TF-IDF)
Training model:
```bash
python train_model.py
```
Output: `models/sentiment_model.pkl`, `models/tfidf_vectorizer.pkl`

Labeling/prediksi ke seluruh data:
```bash
python sentiment.py
```
Output: `data/processed/roblox_sentiment.csv`

Catatan: di implementasi saat ini, SVM menggunakan `SVC(kernel='linear')` dan TF‑IDF `max_features=5000`.

### 4) IndoBERT (Auto-labeling Teacher HF → Fine-tuning Otomatis)
Jalankan mode default (tanpa flag `--annotation_file`):
```bash
python train_indobert.py
```
Output disimpan ke:
- `models/indobert_sentiment/best_model_annotated/`
  - `metrics.json`
  - `confusion_matrix.png`
  - checkpoint model + tokenizer

Model teacher default:
- `w11wo/indonesian-roberta-base-sentiment-classifier`

Opsional (mempercepat dengan sampling):
```bash
python train_indobert.py --max_samples 20000
```

### 5) Analisis berbasis rating (Opsional / Pembanding)
Jika ingin menghasilkan file perbandingan rating untuk pembanding di dashboard:
```bash
python sentiment_comparison_analysis.py
```
Output utama: `data/processed/sentiment_comparison_full.csv` + visualisasi di `data/processed/sentiment_analysis/`

## 🧠 Model & Output
### SVM (Baseline)
- Model: `models/sentiment_model.pkl`
- Vectorizer: `models/tfidf_vectorizer.pkl`
- Dataset hasil prediksi: `data/processed/roblox_sentiment.csv`

### IndoBERT
- Baseline checkpoint: `models/indobert_sentiment/best_model/`
- Fine-tuned otomatis (auto-labeling): `models/indobert_sentiment/best_model_annotated/`
- Prediksi real-time otomatis memakai model fine-tuned jika tersedia (lihat: `predict_indobert.py`)

### Evaluasi
- Metrik IndoBERT fine-tuned dibaca dari `models/indobert_sentiment/best_model_annotated/metrics.json`
- Perbandingan rating disediakan sebagai pembanding opsional (bukan klaim utama pemahaman teks).

## 🧯 Troubleshooting
### 1) Error encoding Windows saat training
Sudah ditangani: output training tidak menggunakan emoji sehingga aman untuk console Windows.

### 2) Port Streamlit sudah dipakai
```bash
streamlit run app.py --server.port 8502
```

### 3) IndoBERT model tidak ter-load
- Pastikan salah satu folder ini ada:
  - `models/indobert_sentiment/best_model/`
  - `models/indobert_sentiment/best_model_annotated/`

## 🧩 Script & Output (Ringkas)
- `scraping.py` → hasil scraping: `data/raw/roblox_raw.csv`
- `preprocessing.py` → hasil preprocessing: `data/processed/roblox_cleaned.csv`
- `train_model.py` → model baseline: `models/sentiment_model.pkl`, `models/tfidf_vectorizer.pkl`
- `sentiment.py` → hasil prediksi SVM: `data/processed/roblox_sentiment.csv`
- `analysis.py` → ringkasan tren: `data/processed/sentiment_analysis_summary.csv`
- `sentiment_comparison_analysis.py` → perbandingan rating vs IndoBERT: `data/processed/sentiment_comparison_full.csv` + `data/processed/sentiment_analysis/*`
- `train_indobert.py`:
  - mode default → auto-labeling teacher + fine-tuning student → `models/indobert_sentiment/best_model_annotated/` + `metrics.json` + `confusion_matrix.png`
  - mode anotasi (`--annotation_file ...`) masih tersedia untuk kebutuhan supervised (opsional)
- `predict_indobert.py` → prediksi real-time (pakai model fine-tuned jika tersedia)
- `generate_report.py` → laporan otomatis (HTML/teks) di `data/processed/sentiment_analysis/`
- `presentation_narrative.py` → narasi presentasi/summary di `data/processed/sentiment_analysis/`
