import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import sys
import subprocess
import time
import re
from collections import Counter
from datetime import datetime, timedelta
import joblib
from streamlit_option_menu import option_menu
from predict_indobert import predict_sentiment
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from google_play_scraper import app as gp_app
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

try:
    from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from streamlit_wordcloud import st_wordcloud
    STREAMLIT_WORDCLOUD_AVAILABLE = True
except ImportError:
    STREAMLIT_WORDCLOUD_AVAILABLE = False

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Roblox Sentiment Dashboard",
    layout="wide",
    page_icon="🎮",
    initial_sidebar_state="expanded"
)

# === CSS INJECTION (Glassmorphism & Professional UI) ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: #1a2744;
    }

    .main {
        background-color: #ffffff;
    }

    /* Hide Streamlit elements but keep the sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
        color: #1a2744;
    }
    
    /* Ensure the sidebar button is visible on white background */
    button[kind="header"] {
        color: #1a2744 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Card Styling */
    .glass-card {
        background: #ffffff;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        padding: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }

    /* Metric Styling */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border-radius: 20px !important;
        padding: 20px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    /* Navigation Radio Styling */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 10px;
    }

    /* Headings */
    h1, h2, h3 {
        color: #1a2744 !important;
        font-weight: 700 !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(74, 144, 217, 0.3);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === DATA LOADING (REAL DATA) ===
@st.cache_data
def load_real_data():
    file_path = 'data/processed/roblox_sentiment.csv'
    if not os.path.exists(file_path):
        st.error(f"File {file_path} tidak ditemukan!")
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    if 'at' in df.columns:
        df['at'] = pd.to_datetime(df['at'], errors='coerce')

    def normalize_sentiment(series: pd.Series) -> pd.Series:
        return (
            series.fillna('')
            .astype(str)
            .str.strip()
            .str.lower()
        )

    def rating_to_sentiment(score):
        try:
            score = int(score)
        except Exception:
            return None
        if score <= 2:
            return 'negatif'
        if score == 3:
            return 'netral'
        return 'positif'

    df['sentiment_svm'] = normalize_sentiment(df.get('sentiment', pd.Series(index=df.index, dtype='object')))

    comparison_path = 'data/processed/sentiment_comparison_full.csv'
    if os.path.exists(comparison_path):
        df_cmp = pd.read_csv(comparison_path, usecols=lambda c: c in {'reviewId', 'sentiment_before', 'sentiment_after'})
        df_cmp['sentiment_before'] = normalize_sentiment(df_cmp.get('sentiment_before', pd.Series(index=df_cmp.index, dtype='object')))
        df_cmp['sentiment_after'] = normalize_sentiment(df_cmp.get('sentiment_after', pd.Series(index=df_cmp.index, dtype='object')))
        if 'reviewId' in df.columns and 'reviewId' in df_cmp.columns:
            df = df.merge(df_cmp, on='reviewId', how='left')
        else:
            df['sentiment_before'] = None
            df['sentiment_after'] = None
    else:
        df['sentiment_before'] = None
        df['sentiment_after'] = None

    if df['sentiment_before'].isna().all() and 'score' in df.columns:
        df['sentiment_before'] = df['score'].apply(rating_to_sentiment)

    df['sentiment_rating'] = normalize_sentiment(df.get('sentiment_before', pd.Series(index=df.index, dtype='object')))
    df['sentiment_indobert'] = normalize_sentiment(df.get('sentiment_after', pd.Series(index=df.index, dtype='object')))

    sentiment_map_ui = {'positif': 'Positif', 'negatif': 'Negatif', 'netral': 'Netral'}
    df['sentiment'] = df['sentiment_svm'].map(sentiment_map_ui).fillna('Tidak diketahui')
    df['sentiment_rating_ui'] = df['sentiment_rating'].map(sentiment_map_ui).fillna('Tidak diketahui')
    df['sentiment_indobert_ui'] = df['sentiment_indobert'].map(sentiment_map_ui).fillna('Tidak diketahui')
    return df

df = load_real_data()

# Apply filters
if not df.empty:
    # Sidebar Filters
    with st.sidebar:
        st.markdown("<h1 style='font-size: 24px;'>🎮 RobloxSentiment</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6b7a99; margin-top: -15px;'>Dashboard Analisis Skripsi</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Professional Navigation using streamlit-option-menu
        page = option_menu(
            menu_title=None,
            options=["Overview", "Analisis Sentimen", "Tren Temporal", "Model & Evaluasi", "Scraping & Preprocessing"],
            icons=["house", "bar-chart", "graph-up", "cpu", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#FFFFFF"},
                "icon": {"color": "#4A90D9", "font-size": "18px"}, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "5px", 
                    "border-radius": "10px",
                    "--hover-color": "rgba(74, 144, 217, 0.1)"
                },
                "nav-link-selected": {"background-color": "rgba(74, 144, 217, 0.2)", "color": "#1a2744", "font-weight": "600", "border-left": "4px solid #4A90D9"},
            }
        )
        
        # Add back emojis to page name for routing consistency
        page_map = {
            "Overview": "🏠 Overview",
            "Analisis Sentimen": "📊 Analisis Sentimen",
            "Tren Temporal": "📈 Tren Temporal",
            "Model & Evaluasi": "🔬 Model & Evaluasi",
            "Scraping & Preprocessing": "⚙️ Scraping & Preprocessing"
        }
        page = page_map[page]
        
        st.markdown("---")
        st.subheader("🔍 Filter Data")
        
        # Date Range Filter
        min_date = df['at'].min().date()
        max_date = df['at'].max().date()
        date_range = st.date_input("Rentang Waktu", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        # Version Filter
        versions = sorted(df['appVersion'].dropna().unique().tolist())
        selected_versions = st.multiselect("Versi Aplikasi", versions, default=versions[:10] if len(versions) > 10 else versions)

    # Filter the dataframe
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['at'].dt.date >= start_date) & (df['at'].dt.date <= end_date)
        if selected_versions:
            mask = mask & (df['appVersion'].isin(selected_versions))
        df_filtered = df[mask]
    else:
        df_filtered = df
else:
    st.warning("Data belum tersedia. Silakan jalankan pipeline pengolahan data.")
    st.stop()

# Use df_filtered for all charts below
df_display = df_filtered if not df_filtered.empty else df

def compute_agreement_metrics(df_subset: pd.DataFrame, true_col: str, pred_col: str, labels_order=None):
    if true_col not in df_subset.columns or pred_col not in df_subset.columns:
        return None
    if labels_order is None:
        labels_order = ['negatif', 'netral', 'positif']
    d = df_subset[[true_col, pred_col]].dropna()
    d = d[(d[true_col].astype(str).str.len() > 0) & (d[pred_col].astype(str).str.len() > 0)]
    d[true_col] = d[true_col].astype(str).str.strip().str.lower()
    d[pred_col] = d[pred_col].astype(str).str.strip().str.lower()
    d = d[d[true_col].isin(labels_order) & d[pred_col].isin(labels_order)]
    if d.empty:
        return None
    y_true = d[true_col].astype(str)
    y_pred = d[pred_col].astype(str)
    acc = float((y_true == y_pred).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, average='macro', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    return {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "labels_order": labels_order,
        "confusion_matrix": cm
    }

def coerce_sentiment_label(value):
    if pd.isna(value):
        return None
    try:
        if isinstance(value, (int, np.integer)):
            return {0: 'negatif', 1: 'netral', 2: 'positif'}.get(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return {0: 'negatif', 1: 'netral', 2: 'positif'}.get(int(value))
    except Exception:
        pass

    s = str(value).strip().lower()
    s = s.replace(' ', '')
    if s in {'negatif', 'negative', 'neg'}:
        return 'negatif'
    if s in {'netral', 'neutral', 'neu'}:
        return 'netral'
    if s in {'positif', 'positive', 'pos'}:
        return 'positif'
    if s in {'0', '1', '2'}:
        return { '0': 'negatif', '1': 'netral', '2': 'positif' }.get(s)
    return None

def prepare_annotation_dataframe(df_ann: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    d = df_ann[[text_col, label_col]].copy()
    d[text_col] = d[text_col].fillna('').astype(str)
    d[label_col] = d[label_col].apply(coerce_sentiment_label)
    d = d[d[label_col].notna() & (d[text_col].str.len() > 0)]
    d = d.rename(columns={text_col: 'text', label_col: 'label'}).reset_index(drop=True)
    return d

@st.cache_resource
def get_indobert_runtime(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, device

def indobert_predict_labels(texts, tokenizer, model, device, batch_size: int = 32, max_length: int = 128):
    label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}
    preds = []
    if not texts:
        return preds
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            pred_ids = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy().tolist()
        preds.extend([label_map.get(int(i), 'netral') for i in pred_ids])
    return preds

def compute_classification_metrics(y_true, y_pred, labels_order=None):
    if labels_order is None:
        labels_order = ['negatif', 'netral', 'positif']
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, average='macro', zero_division=0
    )
    precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, average=None, zero_division=0
    )
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    per_class = pd.DataFrame({
        'Label': labels_order,
        'Precision': precision_cls,
        'Recall': recall_cls,
        'F1': f1_cls,
        'Support': support_cls
    })
    summary = {
        'accuracy': acc,
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'labels_order': labels_order,
        'confusion_matrix': cm
    }
    return summary, per_class

ANNOTATION_PATH = "data/processed/sentiment_text_annotations.csv"
ANNOTATED_MODEL_PATH = "./models/indobert_sentiment/best_model_annotated"
DEFAULT_MODEL_PATH = "./models/indobert_sentiment/best_model"

def load_annotation_df():
    if not os.path.exists(ANNOTATION_PATH):
        return pd.DataFrame(columns=["reviewId", "content", "cleaned_content", "label", "labeled_at"])
    try:
        return pd.read_csv(ANNOTATION_PATH)
    except Exception:
        return pd.DataFrame(columns=["reviewId", "content", "cleaned_content", "label", "labeled_at"])

def append_annotation(row: dict):
    os.makedirs(os.path.dirname(ANNOTATION_PATH), exist_ok=True)
    exists = os.path.exists(ANNOTATION_PATH)
    df_row = pd.DataFrame([row])
    df_row.to_csv(ANNOTATION_PATH, mode="a", header=not exists, index=False, encoding="utf-8")

def build_unlabeled_pool(df_source: pd.DataFrame, labeled_ids: set):
    cols = [c for c in ["reviewId", "content", "cleaned_content", "at", "appVersion"] if c in df_source.columns]
    if "reviewId" not in cols:
        return pd.DataFrame(columns=cols)
    pool = df_source[cols].copy()
    pool = pool.dropna(subset=["reviewId"])
    pool["reviewId"] = pool["reviewId"].astype(str)
    pool = pool[~pool["reviewId"].isin(labeled_ids)]
    pool = pool.drop_duplicates(subset=["reviewId"])
    if "cleaned_content" in pool.columns:
        pool["cleaned_content"] = pool["cleaned_content"].fillna("").astype(str)
        pool = pool[pool["cleaned_content"].str.len() > 0]
    elif "content" in pool.columns:
        pool["content"] = pool["content"].fillna("").astype(str)
        pool = pool[pool["content"].str.len() > 0]
    return pool

def ensure_annotation_queue(pool_df: pd.DataFrame, sample_size: int, seed: int):
    if "annotation_queue" not in st.session_state:
        st.session_state["annotation_queue"] = []
    if "annotation_pos" not in st.session_state:
        st.session_state["annotation_pos"] = 0
    if len(st.session_state["annotation_queue"]) == 0 or st.session_state["annotation_pos"] >= len(st.session_state["annotation_queue"]):
        if pool_df.empty:
            st.session_state["annotation_queue"] = []
            st.session_state["annotation_pos"] = 0
            return
        rng = np.random.default_rng(int(seed))
        n = min(int(sample_size), len(pool_df))
        chosen_idx = rng.choice(pool_df.index.to_numpy(), size=n, replace=False)
        q = pool_df.loc[chosen_idx].to_dict(orient="records")
        st.session_state["annotation_queue"] = q
        st.session_state["annotation_pos"] = 0

def get_current_annotation_item():
    q = st.session_state.get("annotation_queue", [])
    pos = int(st.session_state.get("annotation_pos", 0))
    if not q or pos < 0 or pos >= len(q):
        return None
    return q[pos]

def advance_annotation_item():
    st.session_state["annotation_pos"] = int(st.session_state.get("annotation_pos", 0)) + 1

def get_active_indobert_model_path():
    return ANNOTATED_MODEL_PATH if os.path.exists(ANNOTATED_MODEL_PATH) else DEFAULT_MODEL_PATH

STOPWORDS_ID = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "aku", "saya", "kamu", "dia", "kami", "kita", "mereka",
    "nya", "aja", "kok", "sih", "lah", "ya", "yg", "gk", "ga", "nggak", "tidak", "bukan", "tp", "tpi", "tapi",
    "buat", "untuk", "dengan", "pada", "dalam", "jadi", "udah", "sudah", "belum", "lebih", "banget", "bgt",
    "kalo", "kalau", "karena", "biar", "supaya", "agar", "sama", "juga", "lagi", "masih", "pun", "atau",
    "the", "and", "to", "of", "is", "in", "it", "this", "that", "for", "on", "with", "not"
}

def extract_terms(texts, top_n=10):
    tokens = []
    bigrams = []
    for t in texts:
        if not t:
            continue
        words = [w for w in re.findall(r"[a-zA-Z_]+", str(t).lower()) if len(w) >= 3 and w not in STOPWORDS_ID]
        tokens.extend(words)
        bigrams.extend([" ".join(pair) for pair in zip(words, words[1:])])
    top_tokens = [w for w, _ in Counter(tokens).most_common(int(top_n))]
    top_bigrams = [w for w, _ in Counter(bigrams).most_common(int(top_n))]
    return top_tokens, top_bigrams

@st.cache_data
def load_changelog_cache(cache_path="data/processed/changelog_cache.json"):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_changelog_cache(cache, cache_path="data/processed/changelog_cache.json"):
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def fetch_google_play_changelog(app_id="com.roblox.client"):
    cache = load_changelog_cache()
    if cache.get("appId") == app_id and cache.get("recentChanges"):
        return cache

    if not GP_AVAILABLE:
        return cache

    try:
        app_info = gp_app(app_id, lang="en", country="us")
        changelog_text = app_info.get("recentChanges") or ""
        cache_data = {
            "appId": app_id,
            "version": app_info.get("version", "Unknown"),
            "recentChanges": changelog_text,
            "updated": app_info.get("updated")
        }
        save_changelog_cache(cache_data)
        return cache_data
    except Exception:
        return cache

def gather_peak_texts(df_day: pd.DataFrame, sentiment_label: str):
    if df_day.empty:
        return []
    if "label" in df_day.columns:
        texts = df_day[df_day["label"].astype(str).str.strip().str.lower() == sentiment_label].get("cleaned_content", df_day.get("content", pd.Series(dtype="object"))).fillna("").astype(str).tolist()
        if texts:
            return texts
    return df_day.get("cleaned_content", df_day.get("content", pd.Series(dtype="object"))).fillna("").astype(str).tolist()

def build_wordcloud_frequencies(texts, top_n=40):
    if not WORDCLOUD_AVAILABLE or not texts:
        return {}

    all_words = []
    stopwords = set(WC_STOPWORDS) | set(STOPWORDS_ID)
    for t in texts:
        if not t:
            continue
        tokens = [w for w in re.findall(r"[a-zA-Z_]+", str(t).lower()) if len(w) >= 3 and w not in stopwords]
        all_words.extend(tokens)

    return dict(Counter(all_words).most_common(top_n))

def render_wordcloud_plotly(word_freq, title, color_map="Greens"):
    if not word_freq:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis={'visible': False},
            yaxis={'visible': False},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    try:
        wc = WordCloud(
            width=1000,
            height=600,
            background_color='white',
            colormap=color_map,
            stopwords=set(WC_STOPWORDS) | set(STOPWORDS_ID),
            max_words=80
        ).generate_from_frequencies(word_freq)
        layout = wc.layout_
        xd, yd, texts, sizes, colors, freqs = [], [], [], [], [], []
        for item in layout:
            (word, freq), font_size, position, orientation, color = item
            x, y = position
            xd.append(x)
            yd.append(y)
            texts.append(word)
            sizes.append(max(int(font_size / 1.5), 12))
            colors.append(color)
            freqs.append(freq)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xd,
            y=[-y for y in yd],
            mode='text',
            text=texts,
            textfont={'size': sizes, 'color': colors},
            hovertemplate='<b>%{text}</b><br>Freq %{customdata[0]}<extra></extra>',
            customdata=np.stack([freqs], axis=-1)
        ))
        fig.update_layout(
            title=title,
            xaxis={'visible': False},
            yaxis={'visible': False},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig
    except Exception:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

def match_terms_with_changelog(word_freq, changelog_text):
    if not changelog_text or not word_freq:
        return []
    changelog_words = set(re.findall(r"[a-zA-Z_]+", changelog_text.lower()))
    return [word for word in word_freq.keys() if word.lower() in changelog_words]

NARASI_POSITIF = """
Lonjakan 1.344 review positif pada tanggal ini bersamaan dengan
rilisnya versi 2.715.1115 (3 April 2026), yang membawa sejumlah
fitur besar yang disambut antusias oleh komunitas pemain.

Versi 2.715.1115 dirilis pada 3 April 2026 dan menjadi salah satu
update terbesar Roblox di kuartal pertama 2026. Update ini mencakup:

• Avatar Makeup — fitur kosmetik baru (eyeshadow, lipstik, blush)
  yang dapat di-mix & match, diluncurkan dengan 150+ item dari
  kreator UGC sejak hari pertama.

• Trusted Friends — perluasan fitur pertemanan lintas usia,
  memungkinkan teman dan keluarga dari kelompok usia berbeda
  untuk bermain dan chat bersama di Roblox.

• Robux sebagai opsi pembayaran subscriptions — selain mata uang
  lokal, pemain kini bisa berlangganan menggunakan Robux.

• Universal Importer (Full Release) — dukungan penuh untuk
  import aset image, audio, dan video ke dalam game.

• Perbaikan performa Solid Modeling (CSG) — replikasi geometri
  lebih cepat dengan delta updates, mengurangi lag terutama
  pada koneksi lambat.

• Bug fix AlphaMode & rendering material transparan.

Korelasi dengan Word Cloud:
Kata-kata dominan seperti "bagus", "seru", "suka", dan "main"
mencerminkan kepuasan pemain terhadap fitur sosial baru (Trusted
Friends) dan konten kosmetik (Avatar Makeup). Kata "game" dan
"main" yang mendominasi sejalan dengan peningkatan performa
gameplay dari perbaikan CSG dan rendering. Lonjakan signifikan
(+996 review dari hari sebelumnya) mengindikasikan bahwa
pemain langsung merasakan dampak positif dari fitur-fitur
yang dirilis bersamaan dalam satu minggu.
"""

NARASI_NEGATIF = """
Lonjakan 448 review negatif pada tanggal ini bersamaan dengan
rilisnya versi 2.710.707 (27 Februari 2026), yang membawa
serangkaian breaking changes dan memperparah bug chat yang
sudah berlangsung sejak Januari 2026.

Versi 2.710.707 dirilis pada 27 Februari 2026 dan bertepatan
dengan puncak ketidakpuasan pengguna Indonesia. Update ini
mencakup sejumlah perubahan teknis yang berdampak langsung
pada pengalaman bermain:

• Penghapusan properti lama (Breaking Change) —
  MaterialVariant.StudsPerTileU/V dan Tool.PunchThroughDistance
  dihapus, menyebabkan banyak game yang menggunakan properti
  ini menjadi error atau tidak berfungsi normal.

• Perubahan default ModelStreamingBehavior ke "Improved" —
  perubahan mendadak ini menyebabkan sejumlah game mengalami
  loading lebih lama atau lag, terutama bagi pemain dengan
  koneksi internet terbatas.

• Breaking change API keamanan (diumumkan berlaku 23 Maret 2026)
  — perubahan pada UserHasBadgeAsync dan endpoint badges
  membuat fitur-fitur tertentu dalam game berhenti bekerja
  lebih awal dari yang diharapkan.

• Bug chat yang belum terselesaikan — sejak Januari 2026,
  bug TextChatService menyebabkan pesan chat stuck di status
  "Sending" hingga 5+ menit. Bug ini belum diperbaiki di
  versi 2.710.707, sehingga pengalaman sosial pemain
  terdampak signifikan.

Korelasi dengan Word Cloud:
Kata "chat" yang mendominasi (40,4% isu) secara langsung
berkorelasi dengan bug chat yang belum diperbaiki. Kata
"update" dan "jelek" mencerminkan reaksi negatif terhadap
breaking changes yang tidak dikomunikasikan dengan baik.
Kata "tolong" dan "bisa" mengindikasikan pemain yang
frustrasi dan meminta bantuan karena fitur game mereka
tiba-tiba tidak berfungsi. Lonjakan +92 review negatif
dari hari sebelumnya konsisten dengan pola di mana
pengguna baru menyadari dampak update sehari setelah
rilis.
"""

ISSUE_KEYWORDS = {
    "Bug/Error": ["bug", "error", "eror", "glitch", "freeze", "blank", "hitam", "putih"],
    "Lag/Performa": ["lag", "ngelag", "ngelag", "patah", "fps", "lemot", "lambat", "berat"],
    "Login/Akun": ["login", "log", "masuk", "signin", "sign", "akun", "account", "verifikasi", "umur", "captcha"],
    "Crash/Keluar": ["crash", "force", "close", "keluar", "tutup", "berhenti", "mati", "stuck"],
    "Update Buruk": ["update", "apdet", "upgrade", "pembaruan"],
    "Server/Jaringan": ["server", "jaringan", "koneksi", "internet", "ping", "timeout", "disconnect", "dc"],
    "Chat/Fitur": ["chat", "ngechat", "ngobrol", "fitur", "hapus", "hilang", "remove", "balikin", "kembalikan"]
}

def issue_breakdown(texts):
    texts_l = [str(t).lower() for t in texts if t]
    if not texts_l:
        return pd.DataFrame(columns=["Issue", "Count", "Percent"])
    total = len(texts_l)
    rows = []
    for issue, keys in ISSUE_KEYWORDS.items():
        c = 0
        for t in texts_l:
            if any(k in t for k in keys):
                c += 1
        rows.append((issue, c, (c / total) * 100.0))
    df_issues = pd.DataFrame(rows, columns=["Issue", "Count", "Percent"]).sort_values(["Count", "Issue"], ascending=[False, True])
    return df_issues

def top_app_version(df_subset: pd.DataFrame):
    if "appVersion" not in df_subset.columns:
        return None
    s = df_subset["appVersion"].dropna().astype(str)
    if s.empty:
        return None
    return s.value_counts().index[0]

@st.cache_resource
def load_svm_assets():
    model_path = "./models/sentiment_model.pkl"
    vec_path = "./models/tfidf_vectorizer.pkl"
    if not (os.path.exists(model_path) and os.path.exists(vec_path)):
        return None, None
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    except Exception:
        return None, None

def compute_svm_top_features(model, vectorizer, top_n=20):
    if model is None or vectorizer is None:
        return None
    try:
        features = vectorizer.get_feature_names_out()
    except Exception:
        vocab = getattr(vectorizer, "vocabulary_", None)
        if vocab is None:
            return None
        features = np.array(sorted(vocab.keys()))

    coefs = getattr(model, "coef_", None)
    if coefs is not None:
        w = coefs
        if hasattr(w, "toarray"):
            w = w.toarray()
        else:
            w = np.asarray(w)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        importance = np.mean(np.abs(w), axis=0)
        n = int(min(int(top_n), len(features)))
        idx = np.argsort(importance)[-n:][::-1]
        return pd.DataFrame({"Feature": features[idx], "Importance": importance[idx]})

    idf = getattr(vectorizer, "idf_", None)
    if idf is None:
        return None
    idf = np.asarray(idf)
    n = int(min(int(top_n), len(features)))
    idx = np.argsort(idf)[-n:][::-1]
    return pd.DataFrame({"Feature": features[idx], "Importance": idf[idx]})

with st.sidebar:
    st.markdown("---")
    st.markdown("<div class='glass-card' style='padding: 15px;'>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0; font-weight:700; font-size:14px;'>Fokus Analisis</p>", unsafe_allow_html=True)
    st.markdown("<p style='margin:0; font-size:12px; color:#6b7a99;'>Tren sentimen berbasis waktu dan pola teks (tanpa bergantung rating).</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# === HELPER: PLOTLY STYLE ===
def apply_plotly_style(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_family='DM Sans',
        font_color='#1a2744',
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(255,255,255,0.8)',
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(showgrid=False, showline=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(100,130,180,0.1)', showline=False)
    return fig

# === PAGE ROUTING ===

if page == "🏠 Overview":
    st.title("Ringkasan Eksekutif")
    
    # Row 1: Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Ulasan", f"{len(df_display):,}")
    m2.metric("Sentimen Positif", f"{(df_display['sentiment'] == 'Positif').mean()*100:.1f}%")
    m3.metric("Sentimen Negatif", f"{(df_display['sentiment'] == 'Negatif').mean()*100:.1f}%")
    m4.metric("Rating Rata-rata", f"{df_display['score'].mean():.2f} ⭐")
    
    # Row 2: Charts
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Distribusi Volume Sentimen")
        sentiment_counts = df_display['sentiment'].value_counts().reset_index()
        fig_bar = px.bar(
            sentiment_counts, x='sentiment', y='count',
            color='sentiment',
            color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336'},
            template="plotly_white"
        )
        st.plotly_chart(apply_plotly_style(fig_bar), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Proporsi Sentimen (%)")
        fig_pie = px.pie(
            sentiment_counts, values='count', names='sentiment',
            color='sentiment',
            color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336'},
            hole=0.6
        )
        st.plotly_chart(apply_plotly_style(fig_pie), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Row 3: Table
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Sampel Ulasan Terbaru")
    st.dataframe(
        df_display[['at', 'content', 'sentiment', 'score']].head(10),
        column_config={
            "at": "Tanggal",
            "content": st.column_config.TextColumn("Review", width="large"),
            "sentiment": "Sentimen",
            "score": st.column_config.NumberColumn("Rating ⭐")
        },
        use_container_width=True,
        hide_index=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Kesimpulan Keseluruhan")
    df_all = df.copy()
    df_all = df_all.dropna(subset=['score', 'at'])
    df_all['score'] = pd.to_numeric(df_all['score'], errors='coerce')
    df_all = df_all.dropna(subset=['score'])
    total_reviews = int(len(df_all))
    if total_reviews > 0:
        date_start = df_all['at'].min().date()
        date_end = df_all['at'].max().date()
        avg_rating = float(df_all['score'].mean())
    else:
        date_start = None
        date_end = None
        avg_rating = 0.0

    def pct_for(series: pd.Series, value: str) -> float:
        s = series.dropna().astype(str)
        denom = max(len(s), 1)
        return float((s == value).sum() * 100.0 / denom)

    pct_pos_svm = pct_for(df_all.get('sentiment_svm', pd.Series(dtype='object')), 'positif')
    pct_neu_svm = pct_for(df_all.get('sentiment_svm', pd.Series(dtype='object')), 'netral')
    pct_neg_svm = pct_for(df_all.get('sentiment_svm', pd.Series(dtype='object')), 'negatif')

    pct_pos_indo = pct_for(df_all.get('sentiment_indobert', pd.Series(dtype='object')), 'positif')
    pct_neu_indo = pct_for(df_all.get('sentiment_indobert', pd.Series(dtype='object')), 'netral')
    pct_neg_indo = pct_for(df_all.get('sentiment_indobert', pd.Series(dtype='object')), 'negatif')

    svm_metrics_all = compute_agreement_metrics(df_all, 'sentiment_rating', 'sentiment_svm')
    indo_metrics_all = compute_agreement_metrics(df_all, 'sentiment_rating', 'sentiment_indobert')

    if svm_metrics_all is None:
        svm_text = "—"
    else:
        svm_text = f"{svm_metrics_all['accuracy']*100:.2f}% (F1 {svm_metrics_all['f1']:.2f})"

    if indo_metrics_all is None:
        indo_text = "—"
    else:
        indo_text = f"{indo_metrics_all['accuracy']*100:.2f}% (F1 {indo_metrics_all['f1']:.2f})"

    monthly_path = 'data/processed/sentiment_analysis_summary.csv'
    monthly_note = ""
    if os.path.exists(monthly_path):
        monthly = pd.read_csv(monthly_path)
        monthly['month'] = monthly['month'].astype(str)
        monthly['sentiment'] = monthly['sentiment'].astype(str).str.strip().str.lower()
        pos_rows = monthly[monthly['sentiment'] == 'positif'].copy()
        if not pos_rows.empty and 'percentage' in pos_rows.columns:
            min_pos = float(pos_rows['percentage'].min())
            max_pos = float(pos_rows['percentage'].max())
            monthly_note = f"Sentimen positif konsisten dominan per bulan (sekitar {min_pos:.1f}%–{max_pos:.1f}%)."

    if pct_neu_indo < 1.0 and (pct_pos_indo + pct_neg_indo) > 0:
        neutral_note = "IndoBERT pada dataset ini cenderung sangat jarang memprediksi kelas netral; ini mengindikasikan bias kelas/ketidakseimbangan label."
    else:
        neutral_note = "Distribusi kelas IndoBERT lebih seimbang terhadap netral."

    st.markdown(
        f"""
- Total ulasan dianalisis: **{total_reviews:,}**{f" (periode **{date_start} – {date_end}**)" if date_start is not None else ""}
- Distribusi sentimen (SVM): **Positif {pct_pos_svm:.1f}%**, **Netral {pct_neu_svm:.1f}%**, **Negatif {pct_neg_svm:.1f}%**
- Distribusi sentimen (IndoBERT): **Positif {pct_pos_indo:.1f}%**, **Netral {pct_neu_indo:.1f}%**, **Negatif {pct_neg_indo:.1f}%**
- Ringkasan tren: {monthly_note if monthly_note else "Sentimen positif cenderung dominan sepanjang periode pengambilan data."}
- Catatan model: {neutral_note}
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "📊 Analisis Sentimen":
    st.title("Analisis Sentimen Mendalam")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Distribusi Review per Sentimen")
        sent_cnt = df_display['sentiment'].value_counts().reset_index()
        fig_sent = px.bar(
            sent_cnt,
            x='sentiment',
            y='count',
            color='sentiment',
            color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336', 'Tidak diketahui': '#9CA3AF'}
        )
        st.plotly_chart(apply_plotly_style(fig_sent), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Sentimen per Versi Aplikasi")
        ver_sent = df_display.groupby(['appVersion', 'sentiment']).size().reset_index(name='count')
        fig_ver = px.bar(
            ver_sent, x='appVersion', y='count', color='sentiment',
            barmode='group',
            color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336'}
        )
        st.plotly_chart(apply_plotly_style(fig_ver), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    k1, k2 = st.columns(2)
    with k1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Top Kata (Positif) - Data")
        pos_texts = df_display[df_display['sentiment'] == 'Positif'].get('cleaned_content', df_display.get('content', pd.Series(dtype='object'))).fillna("").astype(str).tolist()
        pos_tokens, pos_bigrams = extract_terms(pos_texts, top_n=10)
        st.write(", ".join(pos_tokens) if pos_tokens else "—")
        st.caption("Frasa umum: " + (", ".join(pos_bigrams) if pos_bigrams else "—"))
        st.markdown("</div>", unsafe_allow_html=True)

    with k2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Top Kata (Negatif) - Data")
        neg_texts = df_display[df_display['sentiment'] == 'Negatif'].get('cleaned_content', df_display.get('content', pd.Series(dtype='object'))).fillna("").astype(str).tolist()
        neg_tokens, neg_bigrams = extract_terms(neg_texts, top_n=10)
        st.write(", ".join(neg_tokens) if neg_tokens else "—")
        st.caption("Frasa umum: " + (", ".join(neg_bigrams) if neg_bigrams else "—"))
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "📈 Tren Temporal":
    st.title("Dinamika Sentimen Sepanjang Waktu")
    
    available_sources = [("SVM (berbasis teks)", "sentiment_svm")]
    if "sentiment_indobert" in df_display.columns and df_display["sentiment_indobert"].notna().any():
        available_sources.append(("IndoBERT (berbasis teks)", "sentiment_indobert"))
    source_label_to_col = {k: v for k, v in available_sources}
    selected_source = st.radio("Sumber Sentimen", [k for k, _ in available_sources], horizontal=True)
    sentiment_source_col = source_label_to_col[selected_source]

    agg = st.radio("Agregasi Data", ["Harian", "Bulanan"], horizontal=True)
    label_ui = {'positif': 'Positif', 'netral': 'Netral', 'negatif': 'Negatif'}
    df_tmp2 = df_display.copy()
    df_tmp2 = df_tmp2.dropna(subset=['at'])
    df_tmp2[sentiment_source_col] = df_tmp2[sentiment_source_col].astype(str).str.strip().str.lower()
    df_tmp2 = df_tmp2[df_tmp2[sentiment_source_col].isin(['positif', 'netral', 'negatif'])]
    df_tmp2['sentiment_ui_ts'] = df_tmp2[sentiment_source_col].map(label_ui)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    if agg == "Harian":
        trend_df = df_tmp2.groupby([df_tmp2['at'].dt.date, 'sentiment_ui_ts']).size().reset_index(name='count')
        fig_area = px.area(
            trend_df, x='at', y='count', color='sentiment_ui_ts',
            color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336'},
            line_shape='spline'
        )
    else:
        df_tmp2['month_period'] = df_tmp2['at'].dt.to_period('M').astype(str)
        trend_df = df_tmp2.groupby(['month_period', 'sentiment_ui_ts']).size().reset_index(name='count')
        fig_area = px.area(
            trend_df, x='month_period', y='count', color='sentiment_ui_ts',
            color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336'},
            line_shape='spline'
        )
    
    st.plotly_chart(apply_plotly_style(fig_area), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    df_daily = df_tmp2.copy()
    df_daily['date'] = df_daily['at'].dt.date
    if df_daily.empty:
        st.info("Tidak ada data cukup untuk analisis temporal pada filter saat ini.")
    else:
        daily_counts = df_daily.groupby(['date', sentiment_source_col]).size().reset_index(name='count')
        daily_total = daily_counts.groupby('date')['count'].sum().reset_index(name='total')
        busiest = daily_total.sort_values('total', ascending=False).head(1)
        busiest_date = busiest['date'].iloc[0] if not busiest.empty else None
        busiest_total = int(busiest['total'].iloc[0]) if not busiest.empty else 0

        pos_row = daily_counts[daily_counts[sentiment_source_col] == 'positif'].sort_values('count', ascending=False).head(1)
        neg_row = daily_counts[daily_counts[sentiment_source_col] == 'negatif'].sort_values('count', ascending=False).head(1)

        pos_date = pos_row['date'].iloc[0] if not pos_row.empty else None
        pos_count = int(pos_row['count'].iloc[0]) if not pos_row.empty else 0
        neg_date = neg_row['date'].iloc[0] if not neg_row.empty else None
        neg_count = int(neg_row['count'].iloc[0]) if not neg_row.empty else 0

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Tanggal Review Terbanyak")
            if busiest_date is None:
                st.write("—")
            else:
                st.write(f"**{busiest_date}**")
                st.write(f"Total review: **{busiest_total:,}**")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Positif Terbanyak")
            if pos_date is None:
                st.write("—")
            else:
                st.write(f"**{pos_date}**")
                st.write(f"Review positif: **{pos_count:,}**")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Negatif Terbanyak")
            if neg_date is None:
                st.write("—")
            else:
                st.write(f"**{neg_date}**")
                st.write(f"Review negatif: **{neg_count:,}**")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Analisis Puncak Positif & Negatif (Pola Teks)")

        def format_terms(terms):
            return ", ".join(terms[:8]) if terms else "—"

        pos_version = None
        neg_version = None
        pos_tokens = pos_bigrams = neg_tokens = neg_bigrams = []
        pos_delta = neg_delta = 0
        issues = pd.DataFrame()

        if pos_date is not None:
            df_pos_day = df_daily[(df_daily['date'] == pos_date) & (df_daily[sentiment_source_col] == 'positif')].copy()
            pos_version = top_app_version(df_pos_day)
            pos_texts = df_pos_day.get('cleaned_content', df_pos_day.get('content', pd.Series(dtype='object'))).fillna("").astype(str).tolist()
            pos_tokens, pos_bigrams = extract_terms(pos_texts, top_n=10)
            pos_prev = daily_counts[(daily_counts['date'] == (pos_date - timedelta(days=1))) & (daily_counts[sentiment_source_col] == 'positif')]
            pos_prev_count = int(pos_prev['count'].iloc[0]) if not pos_prev.empty else 0
            pos_delta = pos_count - pos_prev_count

        if neg_date is not None:
            df_neg_day = df_daily[(df_daily['date'] == neg_date) & (df_daily[sentiment_source_col] == 'negatif')].copy()
            neg_version = top_app_version(df_neg_day)
            neg_texts = df_neg_day.get('cleaned_content', df_neg_day.get('content', pd.Series(dtype='object'))).fillna("").astype(str).tolist()
            neg_tokens, neg_bigrams = extract_terms(neg_texts, top_n=10)
            issues = issue_breakdown(neg_texts)
            neg_prev = daily_counts[(daily_counts['date'] == (neg_date - timedelta(days=1))) & (daily_counts[sentiment_source_col] == 'negatif')]
            neg_prev_count = int(neg_prev['count'].iloc[0]) if not neg_prev.empty else 0
            neg_delta = neg_count - neg_prev_count

        peak_cols = st.columns(2)
        with peak_cols[0]:
            st.markdown("### Positif")
            if pos_date is None:
                st.write("Tidak ada puncak positif yang tersedia.")
            else:
                st.markdown(f"**Tanggal:** {pos_date}")
                st.markdown(f"- Review positif: **{pos_count:,}**")
                st.markdown(f"- Kenaikan vs hari sebelumnya: **{pos_delta:+,}**")
                st.markdown(f"- Versi terbanyak: **{pos_version or '—'}**")
                st.markdown(f"- Kata top: {format_terms(pos_tokens)}")
                st.markdown(f"- Frasa top: {format_terms(pos_bigrams)}")
                st.caption("Hipotesis: lonjakan ini kemungkinan disebabkan oleh peningkatan kepuasan pengguna terhadap update atau fitur baru.")

        with peak_cols[1]:
            st.markdown("### Negatif")
            if neg_date is None:
                st.write("Tidak ada puncak negatif yang tersedia.")
            else:
                st.markdown(f"**Tanggal:** {neg_date}")
                st.markdown(f"- Review negatif: **{neg_count:,}**")
                st.markdown(f"- Kenaikan vs hari sebelumnya: **{neg_delta:+,}**")
                st.markdown(f"- Versi terbanyak: **{neg_version or '—'}**")
                st.markdown(f"- Kata top: {format_terms(neg_tokens)}")
                st.markdown(f"- Frasa top: {format_terms(neg_bigrams)}")
                if not issues.empty:
                    top_issues = issues.head(3)
                    top_issue_text = ", ".join([f"{r.Issue} ({r.Percent:.1f}%)" for r in top_issues.itertuples(index=False)])
                    st.markdown(f"- Isu utama: {top_issue_text}")
                st.caption("Hipotesis: lonjakan ini terkait masalah bug, performa, login, atau pembaruan tidak stabil.")

        if neg_date is not None and not issues.empty:
            with st.expander("Detail kategori masalah (berdasarkan kata kunci)"):
                st.dataframe(issues, use_container_width=True, hide_index=True)

        try:
            pos_review_texts = gather_peak_texts(df_pos_day, 'positif') if pos_date is not None else []
            neg_review_texts = gather_peak_texts(df_neg_day, 'negatif') if neg_date is not None else []
            pos_word_freq = build_wordcloud_frequencies(pos_review_texts, top_n=50)
            neg_word_freq = build_wordcloud_frequencies(neg_review_texts, top_n=50)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Word Cloud Interaktif pada Tanggal Puncak")
            wc_left, wc_right = st.columns(2)
            with wc_left:
                st.markdown("**Word Cloud Positif**")
                if STREAMLIT_WORDCLOUD_AVAILABLE and pos_word_freq:
                    try:
                        st_wordcloud(pos_word_freq)
                    except Exception:
                        st.plotly_chart(apply_plotly_style(render_wordcloud_plotly(pos_word_freq, "Positif", "Greens")), use_container_width=True)
                else:
                    st.plotly_chart(apply_plotly_style(render_wordcloud_plotly(pos_word_freq, "Positif", "Greens")), use_container_width=True)
                if pos_word_freq:
                    st.caption("Hover pada kata untuk melihat frekuensi.")
                else:
                    st.info("Tidak ada teks positif yang cukup untuk menampilkan word cloud.")
            with wc_right:
                st.markdown("**Word Cloud Negatif**")
                if STREAMLIT_WORDCLOUD_AVAILABLE and neg_word_freq:
                    try:
                        st_wordcloud(neg_word_freq)
                    except Exception:
                        st.plotly_chart(apply_plotly_style(render_wordcloud_plotly(neg_word_freq, "Negatif", "Reds")), use_container_width=True)
                else:
                    st.plotly_chart(apply_plotly_style(render_wordcloud_plotly(neg_word_freq, "Negatif", "Reds")), use_container_width=True)
                if neg_word_freq:
                    st.caption("Hover pada kata untuk melihat frekuensi.")
                else:
                    st.info("Tidak ada teks negatif yang cukup untuk menampilkan word cloud.")

            with st.expander("🔍 Validasi dengan Changelog Versi"):
                changelog_data = {}
                if GP_AVAILABLE:
                    with st.spinner("Mengambil changelog terbaru dari Google Play..."):
                        changelog_data = fetch_google_play_changelog()
                else:
                    changelog_data = load_changelog_cache()

                current_version = changelog_data.get("version", "Tidak tersedia")
                changelog_text = changelog_data.get("recentChanges", "")

                left_col, right_col = st.columns(2)
                with left_col:
                    st.markdown(f"#### Positif - Versi puncak: {pos_version if pos_version else '—'}")
                    st.info("Versi 2.715.1115 dirilis pada 3 April 2026.")
                    if pos_version and current_version and pos_version == current_version and changelog_text:
                        st.markdown("**Changelog terbaru:**")
                        st.write(changelog_text)
                        matches = match_terms_with_changelog(pos_word_freq, changelog_text)
                        if matches:
                            st.markdown(f"**Kata yang cocok dengan changelog:** {' '.join([f'`{m}`' for m in matches])}")
                    st.markdown("**📋 Analisis Kontekstual Changelog**")
                    st.success(NARASI_POSITIF)
                with right_col:
                    st.markdown(f"#### Negatif - Versi puncak: {neg_version if neg_version else '—'}")
                    st.info("Versi 2.710.707 dirilis pada 27 Februari 2026.")
                    if neg_version and current_version and neg_version == current_version and changelog_text:
                        st.markdown("**Changelog terbaru:**")
                        st.write(changelog_text)
                        matches = match_terms_with_changelog(neg_word_freq, changelog_text)
                        if matches:
                            st.markdown(f"**Kata yang cocok dengan changelog:** {' '.join([f'`{m}`' for m in matches])}")
                    st.markdown("**📋 Analisis Kontekstual Changelog**")
                    st.error(NARASI_NEGATIF)
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Fitur word cloud / changelog tidak dapat ditampilkan: {str(e)}")

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Insight Otomatis dari Tren")
        total_counts = df_daily[sentiment_source_col].value_counts()
        total_pos = int(total_counts.get('positif', 0))
        total_neu = int(total_counts.get('netral', 0))
        total_neg = int(total_counts.get('negatif', 0))
        total_all = max(int(total_pos + total_neu + total_neg), 1)
        st.markdown(
            f"- Dominasi sentimen ({selected_source}): Positif **{(total_pos/total_all)*100:.1f}%**, Netral **{(total_neu/total_all)*100:.1f}%**, Negatif **{(total_neg/total_all)*100:.1f}%**"
        )
        spike_neg = daily_counts[daily_counts[sentiment_source_col] == 'negatif'].copy()
        if not spike_neg.empty:
            spike_neg = spike_neg.sort_values('count', ascending=False).head(3)
            spike_list = ", ".join([f"{r.date} ({int(r.count):,})" for r in spike_neg.itertuples(index=False)])
            st.markdown(f"- 3 puncak negatif tertinggi: {spike_list}")
        if agg == "Bulanan":
            month_pos = trend_df[trend_df['sentiment_ui_ts'] == 'Positif'].copy()
            if not month_pos.empty:
                month_pos = month_pos.sort_values('count', ascending=False).head(1)
                st.markdown(f"- Bulan dengan review positif terbanyak: **{month_pos['month_period'].iloc[0]}** ({int(month_pos['count'].iloc[0]):,})")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔬 Model & Evaluasi":
    st.title("Evaluasi Performa Algoritma")
    
    # Model selection tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 SVM (Baseline)", "🤖 IndoBERT", "🔍 Sentiment Predictor", "⚖️ Perbandingan Metode"])
    
    with tab1:
        st.subheader("Support Vector Machine (Baseline Model)")

        svm_metrics = compute_agreement_metrics(df_display, 'sentiment_rating', 'sentiment_svm')
        m1, m2, m3, m4 = st.columns(4)
        if svm_metrics is None:
            m1.metric("Agreement (vs Rating)", "—")
            m2.metric("Precision (macro)", "—")
            m3.metric("Recall (macro)", "—")
            m4.metric("F1-Score (macro)", "—")
        else:
            m1.metric("Agreement (vs Rating)", f"{svm_metrics['accuracy']*100:.2f}%")
            m2.metric("Precision (macro)", f"{svm_metrics['precision']:.2f}")
            m3.metric("Recall (macro)", f"{svm_metrics['recall']:.2f}")
            m4.metric("F1-Score (macro)", f"{svm_metrics['f1']:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Confusion Matrix (Rating vs SVM)")
            if svm_metrics is None:
                st.info("Tidak ada data cukup untuk menghitung confusion matrix.")
            else:
                labels = svm_metrics['labels_order']
                label_ui = {'negatif': 'Negatif', 'netral': 'Netral', 'positif': 'Positif'}
                x = [label_ui.get(l, l) for l in labels]
                y = [label_ui.get(l, l) for l in labels]
                fig_hm = px.imshow(
                    svm_metrics['confusion_matrix'],
                    x=x,
                    y=y,
                    text_auto=True,
                    color_continuous_scale='Blues'
                )
                fig_hm.update_layout(xaxis_title="Prediksi SVM", yaxis_title="Rating-based")
                st.plotly_chart(apply_plotly_style(fig_hm), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Fitur Terpenting (Top 20 TF-IDF)")
            svm_model, tfidf = load_svm_assets()
            importance = compute_svm_top_features(svm_model, tfidf, top_n=20)
            if importance is None or importance.empty:
                st.info("Model SVM/TF-IDF belum tersedia untuk menampilkan fitur. Jalankan train_model.py terlebih dulu.")
            else:
                fig_imp = px.bar(
                    importance,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(apply_plotly_style(fig_imp), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("IndoBERT (Deep Learning Model)")
        
        # Check if IndoBERT model exists
        model_path = get_active_indobert_model_path()
        if os.path.exists(model_path):
            if os.path.exists(ANNOTATED_MODEL_PATH):
                st.success("✅ IndoBERT model aktif: fine-tuning otomatis (auto-labeling)")
            else:
                st.success("✅ IndoBERT model aktif: baseline")

            metrics_path = os.path.join(model_path, "metrics.json")
            cm_path = os.path.join(model_path, "confusion_matrix.png")

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy (test)", f"{float(metrics.get('test_accuracy', 0))*100:.2f}%")
                    m2.metric("Precision (macro)", f"{float(metrics.get('precision_macro', 0)):.2f}")
                    m3.metric("Recall (macro)", f"{float(metrics.get('recall_macro', 0)):.2f}")
                    m4.metric("F1-Score (macro)", f"{float(metrics.get('f1_macro', 0)):.2f}")
                except Exception:
                    st.warning("Gagal membaca metrics.json pada model aktif.")
            else:
                st.info("metrics.json belum tersedia pada model aktif.")

            left, right = st.columns([3, 2])
            with left:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Confusion Matrix (test) - Model Aktif")
                if os.path.exists(cm_path):
                    st.image(cm_path, use_container_width=True)
                else:
                    st.info("confusion_matrix.png belum tersedia pada model aktif.")
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Auto-Labeling + Fine-tune Otomatis")
                if not TRANSFORMERS_AVAILABLE:
                    st.warning("Library torch/transformers tidak tersedia, fine-tuning tidak bisa dijalankan.")
                else:
                    p1, p2 = st.columns(2)
                    with p1:
                        epochs = st.slider("Epoch", min_value=1, max_value=10, value=3, step=1)
                        batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)
                    with p2:
                        lr = st.selectbox("Learning rate", [5e-6, 1e-5, 2e-5, 3e-5], index=2)
                        max_length = st.selectbox("Max length", [64, 128, 256], index=1)

                    q1, q2 = st.columns(2)
                    with q1:
                        max_samples = st.number_input("Max samples (opsional)", min_value=0, max_value=200000, value=0, step=5000)
                    with q2:
                        teacher_model = st.text_input("Teacher model (HF)", value="w11wo/indonesian-roberta-base-sentiment-classifier")

                    if st.button("Mulai Auto Fine-tuning", type="primary"):
                        cmd = [
                            sys.executable,
                            "train_indobert.py",
                            "--output_dir",
                            ANNOTATED_MODEL_PATH,
                            "--epochs",
                            str(int(epochs)),
                            "--lr",
                            str(float(lr)),
                            "--batch_size",
                            str(int(batch_size)),
                            "--max_length",
                            str(int(max_length)),
                            "--teacher_model",
                            str(teacher_model).strip(),
                        ]
                        if int(max_samples) > 0:
                            cmd.extend(["--max_samples", str(int(max_samples))])
                        with st.spinner("Auto-labeling dan fine-tuning sedang berjalan..."):
                            start = time.time()
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            elapsed = time.time() - start
                        st.session_state["auto_finetune_logs"] = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
                        st.session_state["auto_finetune_exit_code"] = int(result.returncode)
                        st.session_state["auto_finetune_elapsed_sec"] = float(elapsed)
                        st.rerun()

                    logs = st.session_state.get("auto_finetune_logs")
                    if logs:
                        st.write(f"Exit code: {st.session_state.get('auto_finetune_exit_code')} | Durasi: {st.session_state.get('auto_finetune_elapsed_sec', 0):.1f}s")
                        st.text_area("Log auto fine-tuning", value=logs, height=220)
                st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ IndoBERT model not found. Please run training first.")
            st.code("python train_indobert.py", language="bash")
    
    with tab3:
        st.subheader("Real-time Sentiment Prediction")
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.write("Masukkan teks review Roblox untuk menganalisis sentimen. Pilih model: IndoBERT atau SVM.")

        model_choice = st.selectbox("Pilih Model", options=["IndoBERT", "SVM"], index=0)

        user_input = st.text_area(
            "Input Text:",
            placeholder="Contoh: Game ini sangat bagus dan seru banget!",
            height=100
        )

        def svm_predict(text, model, vectorizer):
            if model is None or vectorizer is None:
                return {
                    'sentiment': 'error',
                    'confidence': 0.0,
                    'probabilities': {'negatif': 0.0, 'netral': 0.0, 'positif': 0.0},
                    'error': 'SVM model atau vectorizer tidak tersedia'
                }
            try:
                x = vectorizer.transform([text])
                pred = model.predict(x)[0]
                label = str(pred).strip().lower()
                if label in {'0', '1', '2'}:
                    label = {'0': 'negatif', '1': 'netral', '2': 'positif'}.get(label, label)

                probs = {'negatif': 0.0, 'netral': 0.0, 'positif': 0.0}
                if hasattr(model, 'predict_proba'):
                    try:
                        p = model.predict_proba(x)[0]
                        classes = [str(c).strip().lower() for c in model.classes_]
                        for cls, prob in zip(classes, p):
                            if cls in {'0', '1', '2'}:
                                cls = {'0': 'negatif', '1': 'netral', '2': 'positif'}.get(cls, cls)
                            probs[cls] = float(prob)
                    except Exception:
                        pass
                elif hasattr(model, 'decision_function'):
                    try:
                        scores = model.decision_function(x)
                        arr = scores[0] if hasattr(scores[0], '__iter__') else [scores[0]]
                        exps = np.exp(np.array(arr) - np.max(arr))
                        soft = exps / exps.sum()
                        classes = [str(c).strip().lower() for c in getattr(model, 'classes_', [])]
                        for cls, prob in zip(classes, soft):
                            if cls in {'0', '1', '2'}:
                                cls = {'0': 'negatif', '1': 'netral', '2': 'positif'}.get(cls, cls)
                            probs[cls] = float(prob)
                    except Exception:
                        pass

                confidence = probs.get(label, 0.0)
                return {'sentiment': label, 'confidence': float(confidence), 'probabilities': probs}
            except Exception as e:
                return {
                    'sentiment': 'error',
                    'confidence': 0.0,
                    'probabilities': {'negatif': 0.0, 'netral': 0.0, 'positif': 0.0},
                    'error': str(e)
                }

        if st.button("🔍 Analyze Sentiment", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text to analyze.")
            else:
                try:
                    if model_choice == "IndoBERT":
                        result = predict_sentiment(user_input)
                    else:
                        svm_model, tfidf = load_svm_assets()
                        result = svm_predict(user_input, svm_model, tfidf)

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if result.get('sentiment') == 'error':
                            st.metric("Sentimen", "Error", result.get('error', 'Prediction failed'))
                        else:
                            st.metric(
                                "Sentimen",
                                result['sentiment'].title(),
                                f"{result['confidence']:.1%} confidence"
                            )

                    with col2:
                        st.subheader("Probabilities")
                        prob_df = pd.DataFrame({
                            'Sentiment': ['Positif', 'Netral', 'Negatif'],
                            'Probability': [
                                result['probabilities'].get('positif', 0.0),
                                result['probabilities'].get('netral', 0.0),
                                result['probabilities'].get('negatif', 0.0)
                            ]
                        })
                        st.bar_chart(prob_df.set_index('Sentiment'))

                    with col3:
                        st.subheader("Details")
                        st.write(f"**Confidence:** {result.get('confidence', 0.0):.4f}")
                        st.write(f"**Text Length:** {len(user_input)} characters")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Make sure selected model is trained and available.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.subheader("Perbandingan Berbasis Rating (Opsional)")
        st.caption("Bagian ini hanya pembanding, bukan fokus utama. Fokus utama ada pada tren waktu dan pola teks.")
        if not {'sentiment_rating', 'sentiment_svm', 'sentiment_indobert', 'score'}.issubset(df_display.columns):
            st.warning("Kolom perbandingan belum lengkap. Pastikan data sentiment_comparison_full.csv tersedia.")
        else:
            show_rating = st.toggle("Tampilkan analisis berbasis rating", value=False)
            if show_rating:
                base = df_display[['score', 'sentiment_rating', 'sentiment_svm', 'sentiment_indobert']].copy()
                base = base.dropna(subset=['score'])
                base['score'] = pd.to_numeric(base['score'], errors='coerce')
                base = base.dropna(subset=['score'])
                base['score'] = base['score'].astype(int)
                base = base[(base['score'] >= 1) & (base['score'] <= 5)]

                def longify(method_name: str, sentiment_col: str) -> pd.DataFrame:
                    d = base[['score', sentiment_col]].rename(columns={sentiment_col: 'sentiment'})
                    d['method'] = method_name
                    return d

                df_long = pd.concat(
                    [
                        longify('Rating', 'sentiment_rating'),
                        longify('SVM', 'sentiment_svm'),
                        longify('IndoBERT', 'sentiment_indobert'),
                    ],
                    ignore_index=True
                )
                df_long = df_long[(df_long['sentiment'].astype(str) != '') & (~df_long['sentiment'].isna())]

                sentiment_map_ui = {'positif': 'Positif', 'negatif': 'Negatif', 'netral': 'Netral'}
                df_long['sentiment_ui'] = df_long['sentiment'].map(sentiment_map_ui).fillna('Tidak diketahui')
                df_long['rating'] = df_long['score'].astype(str)

                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Distribusi Sentimen per Rating (100%)")
                fig_pct = px.histogram(
                    df_long,
                    x='rating',
                    color='sentiment_ui',
                    facet_col='method',
                    barmode='stack',
                    barnorm='percent',
                    category_orders={
                        'rating': ['1', '2', '3', '4', '5'],
                        'method': ['Rating', 'SVM', 'IndoBERT'],
                        'sentiment_ui': ['Negatif', 'Netral', 'Positif', 'Tidak diketahui']
                    },
                    color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336', 'Tidak diketahui': '#9CA3AF'}
                )
                fig_pct.update_layout(yaxis_title="Persentase (%)", xaxis_title="Rating")
                st.plotly_chart(apply_plotly_style(fig_pct), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Jumlah Review per Sentimen (per Metode)")
                summary = df_long.groupby(['method', 'sentiment_ui']).size().reset_index(name='count')
                fig_cnt = px.bar(
                    summary,
                    x='method',
                    y='count',
                    color='sentiment_ui',
                    barmode='group',
                    category_orders={
                        'method': ['Rating', 'SVM', 'IndoBERT'],
                        'sentiment_ui': ['Negatif', 'Netral', 'Positif', 'Tidak diketahui']
                    },
                    color_discrete_map={'Positif': '#4CAF50', 'Netral': '#FFC107', 'Negatif': '#F44336', 'Tidak diketahui': '#9CA3AF'}
                )
                st.plotly_chart(apply_plotly_style(fig_cnt), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Agreement terhadap Rating per Skor")
                tmp = base[['score', 'sentiment_rating', 'sentiment_svm', 'sentiment_indobert']].copy()
                tmp = tmp[(tmp['sentiment_rating'].astype(str) != '') & (~tmp['sentiment_rating'].isna())]
                tmp['agree_svm'] = (tmp['sentiment_svm'] == tmp['sentiment_rating']).astype(int)
                tmp['agree_indobert'] = (tmp['sentiment_indobert'] == tmp['sentiment_rating']).astype(int)
                agree = tmp.groupby('score')[['agree_svm', 'agree_indobert']].mean().reset_index()
                agree = agree.melt(id_vars='score', var_name='model', value_name='agreement')
                agree['model'] = agree['model'].map({'agree_svm': 'SVM', 'agree_indobert': 'IndoBERT'})
                fig_agree = px.line(
                    agree,
                    x='score',
                    y='agreement',
                    color='model',
                    markers=True
                )
                fig_agree.update_layout(yaxis_tickformat=".0%", xaxis_title="Rating", yaxis_title="Agreement")
                st.plotly_chart(apply_plotly_style(fig_agree), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

elif page == "⚙️ Scraping & Preprocessing":
    st.title("Pipeline Pengolahan Data")
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Pipeline Alur Kerja")
    st.graphviz_chart('''
        digraph {
            rankdir=LR;
            fontname="DM Sans";
            node [shape=box, style="filled,rounded", fontname="DM Sans", color="#dde8f5", fillcolor="#eef5ff"];
            edge [color="#94a3b8"];

            A [label="Scraping Review\\n(Google Play Store)\\nraw/roblox_raw.csv", fillcolor="#eef5ff"];
            B [label="Preprocessing Teks\\n(cleaning, case folding,\\nnegation, stopword, stemming)\\nprocessed/roblox_cleaned.csv", fillcolor="#eef5ff"];

            A -> B;

            subgraph cluster_svm {
                label="Baseline ML (SVM)";
                fontname="DM Sans";
                color="#c7d2fe";
                style="rounded";
                S1 [label="TF-IDF Vectorizer", fillcolor="#f5f3ff"];
                S2 [label="Training SVM\\nmodels/sentiment_model.pkl\\nmodels/tfidf_vectorizer.pkl", fillcolor="#f5f3ff"];
                S3 [label="Prediksi Sentimen (SVM)\\nprocessed/roblox_sentiment.csv", fillcolor="#f5f3ff"];
                S1 -> S2 -> S3;
            }

            subgraph cluster_indobert {
                label="Deep Learning (IndoBERT)";
                fontname="DM Sans";
                color="#bae6fd";
                style="rounded";
                I0 [label="Auto-labeling (Teacher HF)\\nlabel: negatif/netral/positif", fillcolor="#eff6ff"];
                I1 [label="Fine-tuning IndoBERT (Student)\\nmodels/indobert_sentiment/best_model_annotated", fillcolor="#eff6ff"];
                I2 [label="Prediksi Sentimen (IndoBERT)\\n(untuk analisis teks & waktu)", fillcolor="#eff6ff"];
                I0 -> I1 -> I2;
            }

            B -> S1;
            B -> I0;
        }
    ''')
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Transformasi Data (Sample)")
    sample_prep = pd.DataFrame({
        "Original Review": [
            "Gamenya sangat seru tapi terkadang lag banget!!",
            "Tidak bagus, sering keluar sendiri pas main",
            "Lumayan lah buat isi waktu luang"
        ],
        "Cleaned & Preprocessed": [
            "game sangat seru kadang lag banget",
            "bagus_NEG sering keluar main",
            "lumayan isi waktu luang"
        ]
    })
    st.table(sample_prep)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <h3>📦 Statistik Scraping</h3>
            <p style='margin:0;'>Total Data: 49,485</p>
            <p style='margin:0;'>Source: Google Play Store</p>
            <p style='margin:0;'>Date: 25 Feb - 22 Apr 2026</p>
        </div>
        """, unsafe_allow_html=True)
