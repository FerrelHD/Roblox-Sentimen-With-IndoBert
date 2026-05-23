# 📋 Skrip & Laporan Otomatis - Panduan Lengkap

## Daftar File yang Dibuat

### 1. **generate_report.py** - Skrip Laporan Otomatis
**Lokasi:** `Skripsi-Roblox-main/generate_report.py`

**Fungsi:**
- Generate laporan otomatis dari hasil analisis sentiment
- Menghasilkan dua format: HTML dan Markdown
- Mengekstrak statistik dari CSV files secara otomatis
- Membuat visualisasi tabel dan ringkasan

**Output:**
- `data/processed/sentiment_analysis/report.html` - Laporan interaktif (buka di browser)
- `data/processed/sentiment_analysis/report.md` - Laporan markdown (untuk dokumentasi)

**Cara Menggunakan:**
```bash
.\venv\Scripts\python.exe generate_report.py
```

**Fitur:**
✅ Automatic data loading dari CSV files
✅ Dynamic statistics calculation (agreement rate, disagreement, dll)
✅ Beautiful HTML layout dengan styling
✅ Markdown format untuk version control
✅ Cross-tabulation dan distribution analysis
✅ Image references untuk visualisasi

---

### 2. **presentation_narrative.py** - Skrip Narasi Presentasi
**Lokasi:** `Skripsi-Roblox-main/presentation_narrative.py`

**Fungsi:**
- Generate narasi presentasi yang siap digunakan
- Menyusun talking points untuk presenter
- Membuat executive summary yang komprehensif
- Struktur slide-by-slide dengan guidance

**Output:**
- `data/processed/sentiment_analysis/executive_summary.txt` - Ringkasan untuk eksekutif
- `data/processed/sentiment_analysis/slide_narratives.txt` - Narasi per slide presentasi
- `data/processed/sentiment_analysis/talking_points.txt` - Tips & trik untuk presenter

**Cara Menggunakan:**
```bash
.\venv\Scripts\python.exe presentation_narrative.py
```

**Fitur:**
✅ 8-slide narrative structure (dari title hingga conclusion)
✅ Talking points dengan timing guide
✅ Q&A preparation dengan contoh jawaban
✅ Business insights yang actionable
✅ Slide-by-slide guidance untuk presenter

---

## Output Files - Detail

### Laporan Otomatis

#### `report.html` 
- **Format:** Interactive HTML with styling
- **Penggunaan:** Buka di browser untuk presentasi atau sharing
- **Isi:**
  - Header dengan gradient styling
  - Statistics cards (Total, Agreement %, Disagreement %)
  - Distribution tables
  - Cross-tabulation analysis
  - Image gallery dengan visualization
  - Disagreement breakdown
  - Footer dengan metadata

#### `report.md`
- **Format:** Markdown (plain text)
- **Penggunaan:** Version control, dokumentasi, email
- **Isi:**
  - Executive summary dengan metrics
  - Distribution comparison
  - Cross-tabulation table
  - Disagreement analysis dengan interpretasi
  - Image references
  - Model information
  - Insights & recommendations

### Narasi Presentasi

#### `executive_summary.txt`
- **Ukuran:** 1 halaman ringkasan
- **Penggunaan:** Untuk eksekutif atau investor
- **Isi:**
  - Key findings dalam format visual
  - Agreement/disagreement rate
  - Sentiment distribution
  - Mismatch analysis dengan contoh
  - Technical metrics
  - Business implications
  - Recommendations dengan timeline

#### `slide_narratives.txt`
- **Struktur:** 8 slide dengan narasi lengkap
  1. Slide 1: Title Slide (opening hook)
  2. Slide 2: Project Overview (context)
  3. Slide 3: Agreement Analysis (main finding)
  4. Slide 4: Distribution Comparison (data)
  5. Slide 5: Mismatch Analysis (insights)
  6. Slide 6: Business Insights (value)
  7. Slide 7: Recommendations (action items)
  8. Slide 8: Conclusion (closing)

- **Penggunaan:** Copy-paste ke speaker notes, atau baca sebagai script

#### `talking_points.txt`
- **Isi Utama:**
  - Opening statements
  - Key data points untuk dihapal
  - Story angles (84.26% agreement, 15.74% disagreement)
  - Handling skepticism dengan Q&A
  - Slide-by-slide guidance
  - Timing notes
  - Closing statements

---

## Cara Menggunakan Semua Material

### Untuk Presentasi Live
1. **Buka `report.html`** di browser
2. **Baca `talking_points.txt`** untuk persiapan (3-5 menit membaca)
3. **Persiapkan slide PowerPoint** dengan visual dari `01_*.png` files
4. **Gunakan `slide_narratives.txt`** sebagai speaker notes
5. **Jawab Q&A menggunakan guidance** dari `talking_points.txt`

### Untuk Email/Sharing
- **Kirim `report.md`** ke non-technical stakeholders (atau `report.html`)
- **Kirim `executive_summary.txt`** ke C-level executives
- **Kirim grafik** `01_*.png`, `02_*.png`, dll untuk social media

### Untuk Internal Documentation
- **Commit ke git:** `report.md` (Markdown format)
- **Share link:** `report.html` (di internal wiki atau knowledge base)
- **Archive:** Seluruh folder `sentiment_analysis/` untuk future reference

---

## Customization

### Untuk Mengubah Report Template
Edit **`generate_report.py`**:
- Ubah styling di HTML section
- Ubah data rows di method `_generate_*_rows()`
- Tambah section baru sesuai kebutuhan

### Untuk Mengubah Presentation Narrative
Edit **`presentation_narrative.py`**:
- Ubah narratives di method `generate_slide_narratives()`
- Tambah/kurangi slide sesuai kebutuhan
- Sesuaikan talking points dengan audience

---

## Integration dengan Workflow

### Automated Pipeline
1. **Training Selesai** → Run `sentiment_comparison_analysis.py`
2. **Comparison Selesai** → Run `generate_report.py`
3. **Report Generated** → Run `presentation_narrative.py`
4. **Narasi Siap** → Share dengan stakeholders

### Scripted Execution
```bash
# Buat master script untuk semua
echo "Running all analysis and reporting..."
.\venv\Scripts\python.exe sentiment_comparison_analysis.py
.\venv\Scripts\python.exe generate_report.py
.\venv\Scripts\python.exe presentation_narrative.py
echo "Done! All reports generated."
```

---

## File Locations Summary

```
Skripsi-Roblox-main/
├── generate_report.py                    ← Script untuk laporan otomatis
├── presentation_narrative.py             ← Script untuk narasi presentasi
└── data/processed/sentiment_analysis/
    ├── 01_distribution_comparison.png    ← Visualisasi 1
    ├── 02_confusion_matrix.png           ← Visualisasi 2
    ├── 03_percentage_distribution.png    ← Visualisasi 3
    ├── 04_agreement_analysis.png         ← Visualisasi 4
    ├── report.html                       ← Laporan HTML (output)
    ├── report.md                         ← Laporan Markdown (output)
    ├── executive_summary.txt             ← Ringkasan eksekutif (output)
    ├── slide_narratives.txt              ← Narasi slide (output)
    └── talking_points.txt                ← Talking points (output)
```

---

## Tips & Best Practices

1. **Generate laporan regularly** - setiap kali model di-retrain
2. **Keep markdown reports** di version control untuk tracking changes
3. **Customize HTML** jika ingin brand colors/styling tertentu
4. **Update data points** di talking_points.txt sebelum presentasi live
5. **Test HTML report** di berbagai browser sebelum share
6. **Simpan presentations** dalam git dengan narrative files

---

**Dibuat:** May 13, 2026
**Project:** Skripsi Roblox Sentiment Analysis
**Status:** ✅ Ready for Production
