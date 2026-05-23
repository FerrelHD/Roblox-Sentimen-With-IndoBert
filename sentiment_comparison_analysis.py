import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ====================
# 1. RATING-BASED SENTIMENT (BEFORE)
# ====================
def create_rating_based_sentiment(score):
    """
    Convert rating score to sentiment label (BEFORE approach)
    1-2: negatif
    3: netral
    4-5: positif
    """
    if score <= 2:
        return 'negatif'
    elif score == 3:
        return 'netral'
    else:  # score >= 4
        return 'positif'

# ====================
# 2. INDOBERT-BASED SENTIMENT (AFTER)
# ====================
def preprocess_text(text):
    """Clean text for IndoBERT"""
    import re
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text.lower()

def get_indobert_sentiment(texts, batch_size=32):
    """
    Predict sentiment using IndoBERT model
    Returns: list of sentiment labels
    """
    print("Loading IndoBERT model...")
    model_name = "indobenchmark/indobert-base-p1"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure to run: pip install transformers torch")
        return None
    
    model = model.to(device)
    model.eval()
    
    # Check if there's a fine-tuned model
    finetuned_path = './models/indobert_sentiment/best_model'
    if os.path.exists(finetuned_path):
        print(f"Loading fine-tuned model from {finetuned_path}...")
        model = AutoModelForSequenceClassification.from_pretrained(finetuned_path)
        model = model.to(device)
    
    sentiment_map = {0: 'negatif', 1: 'netral', 2: 'positif'}
    predictions = []
    probabilities = []
    
    print(f"Processing {len(texts)} texts with IndoBERT...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize
                encoding = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors='pt'
                )
                
                # Move to device
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Predict
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=1)
                pred_labels = torch.argmax(logits, dim=1)
                
                predictions.extend([sentiment_map[label.item()] for label in pred_labels])
                probabilities.extend(probs.cpu().numpy())
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                predictions.extend(['netral'] * len(batch_texts))
                probabilities.extend([[0.33, 0.34, 0.33]] * len(batch_texts))
    
    return predictions, probabilities

# ====================
# 3. MAIN ANALYSIS
# ====================
def run_sentiment_comparison_analysis():
    """
    Main function to load data, create both sentiment labels, and compare
    """
    
    # Load data
    input_file = 'data/processed/roblox_cleaned.csv'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} tidak ditemukan!")
        return
    
    print("=" * 80)
    print("SENTIMENT ANALYSIS: RATING-BASED vs TEXT-BASED (IndoBERT)")
    print("=" * 80)
    
    df = pd.read_csv(input_file)
    print(f"\nTotal data: {len(df)}")
    
    # Remove rows with missing content
    df = df.dropna(subset=['cleaned_content'])
    df = df[df['cleaned_content'].str.len() > 0]
    print(f"After removing empty content: {len(df)}")
    
    # ====================
    # BEFORE: Rating-based sentiment
    # ====================
    print("\n" + "="*80)
    print("STEP 1: Creating Rating-Based Sentiment (BEFORE)")
    print("="*80)
    df['sentiment_before'] = df['score'].apply(create_rating_based_sentiment)
    
    print("\nDistribusi Sentimen BEFORE (Rating-based):")
    print(df['sentiment_before'].value_counts())
    print("\nPersentase Sentimen BEFORE:")
    print(df['sentiment_before'].value_counts(normalize=True) * 100)
    
    # ====================
    # AFTER: IndoBERT-based sentiment
    # ====================
    print("\n" + "="*80)
    print("STEP 2: Creating IndoBERT-Based Sentiment (AFTER)")
    print("="*80)
    
    texts = df['cleaned_content'].tolist()
    sentiments_after, probabilities = get_indobert_sentiment(texts)
    
    if sentiments_after is not None:
        df['sentiment_after'] = sentiments_after
        
        print("\nDistribusi Sentimen AFTER (IndoBERT-based):")
        print(df['sentiment_after'].value_counts())
        print("\nPersentase Sentimen AFTER:")
        print(df['sentiment_after'].value_counts(normalize=True) * 100)
    else:
        print("Skipping IndoBERT prediction due to errors")
        return
    
    # ====================
    # COMPARISON ANALYSIS
    # ====================
    print("\n" + "="*80)
    print("STEP 3: Comparison Analysis")
    print("="*80)
    
    # Calculate agreement
    agreement = (df['sentiment_before'] == df['sentiment_after']).sum()
    agreement_pct = (agreement / len(df)) * 100
    print(f"\nAgreement between BEFORE and AFTER: {agreement}/{len(df)} ({agreement_pct:.2f}%)")
    print(f"Disagreement: {len(df) - agreement}/{len(df)} ({100-agreement_pct:.2f}%)")
    
    # Confusion matrix of BEFORE vs AFTER
    print("\nCross-tabulation: BEFORE vs AFTER")
    crosstab = pd.crosstab(df['sentiment_before'], df['sentiment_after'], margins=True)
    print(crosstab)
    
    # Create detailed comparison DataFrame
    comparison_summary = pd.DataFrame({
        'Metric': ['Total Reviews', 'Negatif', 'Netral', 'Positif'],
        'BEFORE (Rating)': [
            len(df),
            (df['sentiment_before'] == 'negatif').sum(),
            (df['sentiment_before'] == 'netral').sum(),
            (df['sentiment_before'] == 'positif').sum(),
        ],
        'AFTER (IndoBERT)': [
            len(df),
            (df['sentiment_after'] == 'negatif').sum(),
            (df['sentiment_after'] == 'netral').sum(),
            (df['sentiment_after'] == 'positif').sum(),
        ]
    })
    
    print("\nComparison Summary:")
    print(comparison_summary)
    
    # ====================
    # EXAMPLES OF DISAGREEMENT
    # ====================
    print("\n" + "="*80)
    print("STEP 4: Examples of Disagreement (Where BEFORE ≠ AFTER)")
    print("="*80)
    
    disagreement_mask = df['sentiment_before'] != df['sentiment_after']
    disagreement_df = df[disagreement_mask].copy()
    
    if len(disagreement_df) > 0:
        print(f"\nTotal disagreements: {len(disagreement_df)} ({(len(disagreement_df)/len(df)*100):.2f}%)")
        
        # Show examples for each type of disagreement
        sentiment_labels = ['negatif', 'netral', 'positif']
        for before_sent in sentiment_labels:
            for after_sent in sentiment_labels:
                if before_sent == after_sent:
                    continue
                examples = disagreement_df[
                    (disagreement_df['sentiment_before'] == before_sent) &
                    (disagreement_df['sentiment_after'] == after_sent)
                ]
                if len(examples) > 0:
                    print(f"\n📌 {before_sent.upper()} → {after_sent.upper()} ({len(examples)} examples)")
                    # Show up to 3 examples
                    for idx, (i, row) in enumerate(examples.head(3).iterrows()):
                        print(f"\n  Example {idx+1}:")
                        print(f"  Rating: {row['score']}")
                        print(f"  Text: {row['cleaned_content'][:100]}...")
    
    # ====================
    # VISUALIZATIONS
    # ====================
    print("\n" + "="*80)
    print("STEP 5: Generating Visualizations")
    print("="*80)
    
    os.makedirs('data/processed/sentiment_analysis', exist_ok=True)
    
    # 1. Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sentiments = ['negatif', 'netral', 'positif']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    before_counts = [
        (df['sentiment_before'] == s).sum() for s in sentiments
    ]
    after_counts = [
        (df['sentiment_after'] == s).sum() for s in sentiments
    ]
    
    axes[0].bar(sentiments, before_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_title('Distribusi Sentimen: Rating-Based (BEFORE)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Jumlah Review')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(before_counts):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    axes[1].bar(sentiments, after_counts, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_title('Distribusi Sentimen: IndoBERT (AFTER)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Jumlah Review')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(after_counts):
        axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_analysis/01_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 01_distribution_comparison.png")
    plt.close()
    
    # 2. Confusion matrix (BEFORE vs AFTER)
    sentiment_to_num = {'negatif': 0, 'netral': 1, 'positif': 2}
    y_before = df['sentiment_before'].map(sentiment_to_num).values
    y_after = df['sentiment_after'].map(sentiment_to_num).values
    
    cm = confusion_matrix(y_before, y_after)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sentiments, yticklabels=sentiments,
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_title('Confusion Matrix: BEFORE (Rating) vs AFTER (IndoBERT)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rating-Based (BEFORE)')
    ax.set_xlabel('IndoBERT-Based (AFTER)')
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_analysis/02_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 02_confusion_matrix.png")
    plt.close()
    
    # 3. Percentage distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    before_pct = [(c/sum(before_counts)*100) for c in before_counts]
    after_pct = [(c/sum(after_counts)*100) for c in after_counts]
    
    axes[0].pie(before_counts, labels=sentiments, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0].set_title('Persentase Sentimen: Rating-Based (BEFORE)', fontsize=12, fontweight='bold')
    
    axes[1].pie(after_counts, labels=sentiments, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1].set_title('Persentase Sentimen: IndoBERT (AFTER)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_analysis/03_percentage_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 03_percentage_distribution.png")
    plt.close()
    
    # 4. Agreement visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    agreement_data = [agreement, len(df) - agreement]
    axes[0].bar(['Agreement', 'Disagreement'], agreement_data, color=['#27ae60', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_title('Agreement between BEFORE and AFTER', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(agreement_data):
        axes[0].text(i, v + 50, f'{v}\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')
    
    axes[1].pie(agreement_data, labels=['Agreement', 'Disagreement'], 
                autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'], startangle=90)
    axes[1].set_title('Rasio Agreement vs Disagreement', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('data/processed/sentiment_analysis/04_agreement_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: 04_agreement_analysis.png")
    plt.close()
    
    # ====================
    # SAVE RESULTS
    # ====================
    print("\n" + "="*80)
    print("STEP 6: Saving Results")
    print("="*80)
    
    # Save full comparison dataset
    output_file = 'data/processed/sentiment_comparison_full.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Full comparison saved to: {output_file}")
    
    # Save summary
    summary_file = 'data/processed/sentiment_comparison_summary.csv'
    comparison_summary.to_csv(summary_file, index=False)
    print(f"✅ Summary saved to: {summary_file}")
    
    # Save crosstab
    crosstab_file = 'data/processed/sentiment_crosstab.csv'
    crosstab.to_csv(crosstab_file)
    print(f"✅ Cross-tabulation saved to: {crosstab_file}")
    
    # Save disagreement examples
    if len(disagreement_df) > 0:
        disagreement_file = 'data/processed/sentiment_disagreement_examples.csv'
        disagreement_df[['score', 'cleaned_content', 'sentiment_before', 'sentiment_after']].to_csv(
            disagreement_file, index=False
        )
        print(f"✅ Disagreement examples saved to: {disagreement_file}")
    
    # ====================
    # FINAL SUMMARY
    # ====================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal Reviews Analyzed: {len(df):,}")
    print(f"Agreement: {agreement:,} ({agreement_pct:.2f}%)")
    print(f"Disagreement: {len(df) - agreement:,} ({100-agreement_pct:.2f}%)")
    print("\nSentiment Distribution Comparison:")
    print(comparison_summary.to_string(index=False))
    print("\n✅ All files saved to: data/processed/sentiment_analysis/")
    print("="*80)

if __name__ == "__main__":
    run_sentiment_comparison_analysis()
