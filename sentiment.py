import pandas as pd
import numpy as np
import os
import joblib

def process_sentiment_ml(input_file='data/processed/roblox_cleaned.csv', output_file='data/processed/roblox_sentiment.csv'):
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan.")
        return

    model_path = 'models/sentiment_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'

    if not os.path.exists(model_path):
        print("Error: Model belum dilatih. Jalankan python train_model.py dahulu.")
        return

    print("--- Memulai Pelabelan dengan Machine Learning (Random Forest) ---")
    
    # Load Model & Vectorizer
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)

    df = pd.read_csv(input_file)
    print(f"Memproses {len(df)} data...")

    # Handle NaN values in cleaned_content
    df = df.dropna(subset=['cleaned_content'])
    df['cleaned_content'] = df['cleaned_content'].astype(str)
    print(f"Data setelah dihapus NaN: {len(df)}")

    # Transform & Predict
    X = tfidf.transform(df['cleaned_content'])
    df['sentiment'] = model.predict(X)
    
    # Save result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Pelabelan selesai. Data tersimpan di {output_file}")
    
    print("\nDistribusi Sentimen (Random Forest):")
    print(df['sentiment'].value_counts())

if __name__ == "__main__":
    process_sentiment_ml()
