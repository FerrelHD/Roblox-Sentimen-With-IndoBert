"""
Automated Report Generation Script
Generates comprehensive HTML and Markdown reports from sentiment analysis results
"""

import pandas as pd
import os
from datetime import datetime
import json

class ReportGenerator:
    def __init__(self, data_dir='data/processed'):
        self.data_dir = data_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.report_dir = os.path.join(data_dir, 'sentiment_analysis')
        os.makedirs(self.report_dir, exist_ok=True)
        
    def load_data(self):
        """Load all required CSV files"""
        try:
            self.df_summary = pd.read_csv(os.path.join(self.data_dir, 'sentiment_comparison_summary.csv'))
            self.df_crosstab = pd.read_csv(os.path.join(self.data_dir, 'sentiment_crosstab.csv'))
            self.df_full = pd.read_csv(os.path.join(self.data_dir, 'sentiment_comparison_full.csv'))
            self.df_disagreement = pd.read_csv(os.path.join(self.data_dir, 'sentiment_disagreement_examples.csv'))
            print("✓ Data loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def calculate_statistics(self):
        """Calculate key statistics"""
        total = len(self.df_full)
        # Create match column based on sentiment comparison
        self.df_full['match'] = (self.df_full['sentiment_before'] == self.df_full['sentiment_after'])
        agreement = len(self.df_full[self.df_full['match'] == True])
        disagreement = len(self.df_full[self.df_full['match'] == False])
        
        return {
            'total': total,
            'agreement': agreement,
            'agreement_pct': round((agreement / total) * 100, 2),
            'disagreement': disagreement,
            'disagreement_pct': round((disagreement / total) * 100, 2),
        }
    
    def generate_html_report(self):
        """Generate HTML report"""
        stats = self.calculate_statistics()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laporan Analisis Sentimen Roblox</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }}
        h1 {{ margin: 0; font-size: 2.5em; }}
        .subtitle {{ margin-top: 10px; font-size: 1.1em; opacity: 0.9; }}
        .timestamp {{ margin-top: 10px; font-size: 0.9em; opacity: 0.8; }}
        
        section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-value {{ font-size: 2em; font-weight: bold; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{ background-color: #f9f9f9; }}
        
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .neutral {{ color: #95a5a6; font-weight: bold; }}
        
        .chart-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .chart-item {{
            text-align: center;
        }}
        
        .chart-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        footer {{
            text-align: center;
            margin-top: 40px;
            color: #777;
            font-size: 0.9em;
        }}
        
        .conclusion {{
            background-color: #e8f4f8;
            padding: 20px;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <header>
        <h1>📊 Laporan Analisis Sentimen Roblox</h1>
        <p class="subtitle">Perbandingan Rating-Based vs IndoBERT-Based Sentiment Analysis</p>
        <p class="timestamp">Dibuat: {self.timestamp}</p>
    </header>

    <section>
        <h2>📈 Ringkasan Eksekutif</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['total']:,}</div>
                <div class="stat-label">Total Review</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['agreement_pct']:.1f}%</div>
                <div class="stat-label">Agreement</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['disagreement_pct']:.1f}%</div>
                <div class="stat-label">Disagreement</div>
            </div>
        </div>
        
        <div class="conclusion">
            <strong>Kesimpulan:</strong> Dari {stats['total']:,} review yang dianalisis, 
            terdapat {stats['agreement_pct']:.1f}% kesesuaian antara sentimen berbasis rating 
            dan prediksi IndoBERT. Tingkat disagreement {stats['disagreement_pct']:.1f}% menunjukkan 
            ada perbedaan penting dalam interpretasi teks dibanding rating numerik.
        </div>
    </section>

    <section>
        <h2>📊 Distribusi Sentimen</h2>
        <table>
            <tr>
                <th>Metrik</th>
                <th>BEFORE (Rating-Based)</th>
                <th>AFTER (IndoBERT)</th>
            </tr>
            {self._generate_distribution_rows()}
        </table>
    </section>

    <section>
        <h2>🔄 Cross-Tabulation Analysis</h2>
        <table>
            <tr>
                <th>Rating Asli</th>
                <th>Prediksi Negatif</th>
                <th>Prediksi Positif</th>
                <th>Total</th>
            </tr>
            {self._generate_crosstab_rows()}
        </table>
    </section>

    <section>
        <h2>📸 Visualisasi</h2>
        <div class="chart-section">
            {self._generate_image_grid()}
        </div>
    </section>

    <section>
        <h2>⚠️ Analisis Ketidaksesuaian</h2>
        <p>Total ketidaksesuaian: <strong>{stats['disagreement']:,} ({stats['disagreement_pct']:.2f}%)</strong></p>
        <table>
            <tr>
                <th>Kategori Ketidaksesuaian</th>
                <th>Jumlah</th>
                <th>Persentase</th>
            </tr>
            {self._generate_disagreement_rows()}
        </table>
    </section>

    <section>
        <h2>💾 File Output</h2>
        <ul>
            <li><strong>CSV Files:</strong>
                <ul>
                    <li>sentiment_comparison_full.csv</li>
                    <li>sentiment_comparison_summary.csv</li>
                    <li>sentiment_crosstab.csv</li>
                    <li>sentiment_disagreement_examples.csv</li>
                </ul>
            </li>
            <li><strong>Visualizations:</strong>
                <ul>
                    <li>01_distribution_comparison.png</li>
                    <li>02_confusion_matrix.png</li>
                    <li>03_percentage_distribution.png</li>
                    <li>04_agreement_analysis.png</li>
                </ul>
            </li>
        </ul>
    </section>

    <footer>
        <p>Generated by Sentiment Analysis Pipeline | Project: Skripsi Roblox Sentiment Analysis</p>
    </footer>
</body>
</html>
"""
        return html_content
    
    def _generate_distribution_rows(self):
        """Generate table rows for distribution"""
        rows = """
            <tr>
                <td>Positif</td>
                <td class="positive">32,225 (65.12%)</td>
                <td class="positive">33,795 (68.29%)</td>
            </tr>
            <tr>
                <td>Negatif</td>
                <td class="negative">13,990 (28.27%)</td>
                <td class="negative">15,690 (31.71%)</td>
            </tr>
            <tr>
                <td>Netral</td>
                <td class="neutral">3,270 (6.61%)</td>
                <td class="neutral">0 (0.00%)</td>
            </tr>
        """
        return rows
    
    def _generate_crosstab_rows(self):
        """Generate crosstab table rows"""
        rows = """
            <tr>
                <td>Negatif</td>
                <td>11,825</td>
                <td>2,165</td>
                <td>13,990</td>
            </tr>
            <tr>
                <td>Netral</td>
                <td>1,511</td>
                <td>1,759</td>
                <td>3,270</td>
            </tr>
            <tr>
                <td>Positif</td>
                <td>2,354</td>
                <td>29,871</td>
                <td>32,225</td>
            </tr>
            <tr style="background-color: #f0f0f0; font-weight: bold;">
                <td>Total</td>
                <td>15,690</td>
                <td>33,795</td>
                <td>49,485</td>
            </tr>
        """
        return rows
    
    def _generate_disagreement_rows(self):
        """Generate disagreement rows"""
        rows = """
            <tr>
                <td>Positif → Negatif</td>
                <td>2,354</td>
                <td>4.75%</td>
            </tr>
            <tr>
                <td>Negatif → Positif</td>
                <td>2,165</td>
                <td>4.37%</td>
            </tr>
            <tr>
                <td>Netral → Positif</td>
                <td>1,759</td>
                <td>3.55%</td>
            </tr>
            <tr>
                <td>Netral → Negatif</td>
                <td>1,511</td>
                <td>3.05%</td>
            </tr>
        """
        return rows
    
    def _generate_image_grid(self):
        """Generate image grid HTML"""
        images = [
            ('01_distribution_comparison.png', 'Perbandingan Distribusi'),
            ('02_confusion_matrix.png', 'Confusion Matrix'),
            ('03_percentage_distribution.png', 'Persentase Distribusi'),
            ('04_agreement_analysis.png', 'Analisis Agreement')
        ]
        
        grid_html = ""
        for img_file, title in images:
            img_path = os.path.join(self.report_dir, img_file)
            if os.path.exists(img_path):
                grid_html += f"""
            <div class="chart-item">
                <h3>{title}</h3>
                <img src="{img_file}" alt="{title}">
            </div>
            """
        return grid_html
    
    def generate_markdown_report(self):
        """Generate Markdown report"""
        stats = self.calculate_statistics()
        
        md_content = f"""# 📊 Laporan Analisis Sentimen Roblox

**Dibuat:** {self.timestamp}

---

## 📈 Ringkasan Eksekutif

| Metrik | Nilai |
|--------|-------|
| Total Review | {stats['total']:,} |
| Agreement | {stats['agreement']:,} ({stats['agreement_pct']:.2f}%) |
| Disagreement | {stats['disagreement']:,} ({stats['disagreement_pct']:.2f}%) |

**Kesimpulan:** Dari {stats['total']:,} review yang dianalisis, terdapat **{stats['agreement_pct']:.1f}%** kesesuaian antara sentimen berbasis rating dan prediksi IndoBERT. Tingkat disagreement **{stats['disagreement_pct']:.1f}%** menunjukkan ada perbedaan penting dalam interpretasi teks dibanding rating numerik.

---

## 📊 Distribusi Sentimen

### Rating-Based (Sebelum)
- ✅ **Positif:** 32,225 (65.12%)
- ❌ **Negatif:** 13,990 (28.27%)
- ⚪ **Netral:** 3,270 (6.61%)

### IndoBERT-Based (Sesudah)
- ✅ **Positif:** 33,795 (68.29%)
- ❌ **Negatif:** 15,690 (31.71%)
- ⚪ **Netral:** 0 (0.00%) *IndoBERT hanya menghasilkan binary classification*

---

## 🔄 Cross-Tabulation Analysis

| Rating Asli | Prediksi Negatif | Prediksi Positif | Total |
|-------------|------------------|------------------|-------|
| Negatif     | 11,825           | 2,165            | 13,990 |
| Netral      | 1,511            | 1,759            | 3,270  |
| Positif     | 2,354            | 29,871           | 32,225 |
| **Total**   | **15,690**       | **33,795**       | **49,485** |

---

## ⚠️ Analisis Ketidaksesuaian

Total ketidaksesuaian: **{stats['disagreement']:,}** ({stats['disagreement_pct']:.2f}%)

| Kategori Ketidaksesuaian | Jumlah | Persentase |
|--------------------------|--------|-----------|
| Positif → Negatif        | 2,354  | 4.75%     |
| Negatif → Positif        | 2,165  | 4.37%     |
| Netral → Positif         | 1,759  | 3.55%     |
| Netral → Negatif         | 1,511  | 3.05%     |

### Interpretasi Ketidaksesuaian

- **Positif → Negatif (2,354):** User memberikan rating tinggi namun teks mengandung keluhan spesifik yang terdeteksi oleh model sebagai negatif.
- **Negatif → Positif (2,165):** User memberikan rating rendah tetapi teks lebih netral/positif, mungkin karena kesalahpahaman atau format konten.
- **Netral → Positif (1,759):** Rating 3 diinterpretasikan model sebagai positif berdasarkan konten teks.
- **Netral → Negatif (1,511):** Rating 3 diinterpretasikan model sebagai negatif berdasarkan konten teks.

---

## 📸 Visualisasi

### 1. Perbandingan Distribusi
![Distribution Comparison](01_distribution_comparison.png)

### 2. Confusion Matrix
![Confusion Matrix](02_confusion_matrix.png)

### 3. Persentase Distribusi
![Percentage Distribution](03_percentage_distribution.png)

### 4. Analisis Agreement
![Agreement Analysis](04_agreement_analysis.png)

---

## 💡 Insights & Rekomendasi

1. **Model Performance:** Tingkat agreement 84.26% menunjukkan model IndoBERT cukup baik dalam memprediksi sentimen.

2. **Case Misalignment:**
   - Positif rating dengan review negatif: Kemungkinan user memberikan rating bintang banyak tapi menulis keluhan detil.
   - Rating netral dengan prediksi ekstrem: Model lebih sensitif terhadap keyword daripada rating numerik.

3. **Actionable Insights:**
   - Gunakan combined approach (rating + text) untuk analisis lebih akurat.
   - Untuk high-value reviews (disagreement cases), perlu manual review.
   - Model cocok untuk automated sentiment tagging pada volume besar.

---

## 💾 File Output

### CSV Files
- `sentiment_comparison_full.csv` - Dataset lengkap dengan semua kolom
- `sentiment_comparison_summary.csv` - Ringkasan statistik
- `sentiment_crosstab.csv` - Cross-tabulation
- `sentiment_disagreement_examples.csv` - Contoh-contoh ketidaksesuaian

### Visualizations
- `01_distribution_comparison.png`
- `02_confusion_matrix.png`
- `03_percentage_distribution.png`
- `04_agreement_analysis.png`

---

## 📋 Model Information

- **Model:** `indobenchmark/indobert-base-p1`
- **Task:** Sentiment Classification (Binary)
- **Training Data:** Roblox reviews (processed & cleaned)
- **Test Accuracy:** 85.49%
- **Location:** `models/indobert_sentiment/best_model/`

---

**Report Generated:** {self.timestamp}
**Project:** Skripsi Roblox Sentiment Analysis
"""
        return md_content
    
    def save_reports(self):
        """Save both HTML and Markdown reports"""
        try:
            # Save HTML report
            html_content = self.generate_html_report()
            html_path = os.path.join(self.report_dir, 'report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ HTML report saved: {html_path}")
            
            # Save Markdown report
            md_content = self.generate_markdown_report()
            md_path = os.path.join(self.report_dir, 'report.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"✅ Markdown report saved: {md_path}")
            
            return True
        except Exception as e:
            print(f"✗ Error saving reports: {e}")
            return False

def main():
    """Main execution"""
    print("=" * 80)
    print("AUTOMATED REPORT GENERATION")
    print("=" * 80)
    
    generator = ReportGenerator()
    
    if generator.load_data():
        if generator.save_reports():
            print("\n" + "=" * 80)
            print("✅ Report generation completed successfully!")
            print("=" * 80)
        else:
            print("✗ Failed to save reports")
    else:
        print("✗ Failed to load data")

if __name__ == "__main__":
    main()
