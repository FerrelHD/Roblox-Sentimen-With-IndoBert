import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Indonesian stopwords
list_stopwords = set(stopwords.words('indonesian'))

def clean_text(text):
    """
    Perform text preprocessing:
    1. Case folding
    2. Remove punctuation
    3. Tokenization
    4. Stopword removal
    5. Stemming
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Case folding
    text = text.lower()
    
    # 2. Remove punctuation and numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'\d+', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Negation handling (NEW)
    negation_words = ["tidak", "bukan", "jangan", "belum", "tak", "enggak", "nggak", "ga", "gak"]
    
    temp_tokens = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word in negation_words:
            if i + 1 < len(tokens):
                temp_tokens.append(tokens[i+1] + "_NEG")
                i += 2 
            else:
                temp_tokens.append(word)
                i += 1
        else:
            temp_tokens.append(word)
            i += 1
    tokens = temp_tokens
    
    # 5. Stopword removal & Token length filtering
    tokens = [word for word in tokens if word not in list_stopwords and len(word) > 2]
    
    # 6. Stemming
    # Note: Stemming can be slow for large datasets.
    # For Skripsi purposes, it's often required.
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(stemmed_tokens)

def preprocess_data(input_file='data/raw/roblox_raw.csv', output_file='data/processed/roblox_cleaned.csv'):
    """
    Load raw data, clean it, and save to processed folder.
    """
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} tidak ditemukan.")
        return
    
    print(f"Memproses data dari {input_file}...")
    df = pd.DataFrame()
    
    # Read CSV (using chunksize if file is very large, but 50k should fit in memory)
    df = pd.read_csv(input_file)
    
    # Check if 'content' column exists (standard name from google-play-scraper)
    if 'content' not in df.columns:
        print("Error: Kolom 'content' tidak ditemukan di CSV.")
        return
    
    # Apply preprocessing
    print("Sedang melakukan preprocessing (ini mungkin memakan waktu)...")
    df['cleaned_content'] = df['content'].apply(clean_text)
    
    # Save result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Preprocessing selesai. Data tersimpan di {output_file}")

if __name__ == "__main__":
    preprocess_data()