import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_sentiment_model():
    input_file = 'data/processed/roblox_cleaned.csv'
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan.")
        return

    print("--- Memulai Training Machine Learning (Random Forest) ---")
    df = pd.read_csv(input_file)
    
    def get_label(score):
        if score >= 4: return 'positif'
        elif score <= 2: return 'negatif'
        else: return 'netral'
    
    df['label'] = df['score'].apply(get_label)
    df = df[['cleaned_content', 'label']].dropna()

    # --- Feature Extraction (TF-IDF) ---
    print("Mengekstrak fitur teks (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_content'])
    y = df['label']

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Build & Train Model ---
    print("Melatih model SVM... (lebih baik untuk teks)")
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    print(f"\nAkurasi Model: {accuracy_score(y_test, y_pred):.2%}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    # --- Save Model & Assets ---
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
        
    print("\n--- Training Selesai! ---")
    print("Model disimpan di: models/sentiment_model.pkl")

if __name__ == "__main__":
    train_sentiment_model()
