import pandas as pd
import numpy as np
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import pickle

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load & Explore Dataset
def load_and_explore_data():
    print("Loading dataset...")
    # Try both possible file locations
    files_to_try = [
        'data/processed/sentiment_comparison_full.csv',
        'data/processed/roblox_sentiment.csv'
    ]
    
    df = None
    for file_path in files_to_try:
        if os.path.exists(file_path):
            print(f"Found: {file_path}")
            df = pd.read_csv(file_path)
            break
    
    if df is None:
        print("Error: No suitable dataset found!")
        print(f"Tried: {files_to_try}")
        return None

    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Determine which sentiment column to use
    if 'sentiment_before' in df.columns and 'sentiment_after' not in df.columns:
        print("\n[INFO] Using sentiment_before column (rating-based)")
        df['sentiment'] = df['sentiment_before']
    elif 'sentiment' not in df.columns and 'sentiment_before' in df.columns:
        print("\n[INFO] No 'sentiment' column found. Using 'sentiment_before' as ground truth")
        df['sentiment'] = df['sentiment_before']

    print("\nSentiment distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    print(sentiment_counts)

    print("\nPercentage distribution:")
    sentiment_percentages = df['sentiment'].value_counts(normalize=True) * 100
    print(sentiment_percentages)

    # Check for class imbalance
    print("\nClass imbalance analysis:")
    min_class = sentiment_counts.min()
    max_class = sentiment_counts.max()
    imbalance_ratio = max_class / min_class
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2:
        print("[WARN] Significant class imbalance detected!")
    else:
        print("[OK] Classes are relatively balanced.")

    return df

# 2. Preprocessing for IndoBERT
def preprocess_text(text):
    """Clean text for IndoBERT"""
    import re
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text.lower()

def prepare_data(df):
    print("\nPreprocessing data...")

    # Clean text
    df['cleaned_content'] = df['cleaned_content'].fillna('').apply(preprocess_text)

    # Map sentiments to labels
    sentiment_map = {'negatif': 0, 'netral': 1, 'positif': 2}
    df['label'] = df['sentiment'].map(sentiment_map)

    # Remove any rows with NaN labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"After preprocessing: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts()}")

    return df

def coerce_sentiment_label(value):
    if pd.isna(value):
        return None
    try:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return int(value)
    except Exception:
        pass
    s = str(value).strip().lower()
    if s in {'negatif', 'negative', 'neg'}:
        return 0
    if s in {'netral', 'neutral', 'neu'}:
        return 1
    if s in {'positif', 'positive', 'pos'}:
        return 2
    if s in {'0', '1', '2'}:
        return int(s)
    return None

def load_annotated_data(file_path, text_col='cleaned_content', label_col='label'):
    # Loads user-provided annotations (manual labels) for supervised fine-tuning.
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    df = pd.read_csv(file_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Kolom tidak ditemukan. Teks: {text_col}, Label: {label_col}")
    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].fillna('').astype(str).apply(preprocess_text)
    df[label_col] = df[label_col].apply(coerce_sentiment_label)
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    df = df[df[text_col].str.len() > 0]
    df = df.rename(columns={text_col: 'cleaned_content', label_col: 'label'})
    df['sentiment'] = df['label'].map({0: 'negatif', 1: 'netral', 2: 'positif'})
    print(f"Annotated dataset shape: {df.shape}")
    print("Label distribution:")
    print(df['label'].value_counts())
    return df

def infer_teacher_label_mapping(id2label):
    # Maps teacher model label names to this project's label ids:
    # 0=negatif, 1=netral, 2=positif
    def norm(s):
        return str(s).strip().lower()
    mapped = {}
    for idx, name in (id2label or {}).items():
        n = norm(name)
        if "neg" in n:
            mapped[int(idx)] = 0
        elif "neu" in n or "net" in n:
            mapped[int(idx)] = 1
        elif "pos" in n:
            mapped[int(idx)] = 2
    if len(mapped) == 3:
        return mapped

    # Fallback for common ordering (negative, neutral, positive).
    return {0: 0, 1: 1, 2: 2}

def auto_label_with_teacher(texts, teacher_model_name, batch_size=64, max_length=128):
    # Uses a pretrained HF sentiment classifier as a "teacher" to label texts.
    print(f"Teacher model: {teacher_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(teacher_model_name)
    model.to(device)
    model.eval()

    teacher_map = infer_teacher_label_mapping(getattr(model.config, "id2label", None))
    print(f"Teacher label mapping (teacher_id -> label): {teacher_map}")

    preds = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Auto-labeling", leave=False):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            batch_pred_ids = torch.argmax(out.logits, dim=1).detach().cpu().numpy().tolist()
            preds.extend([teacher_map.get(int(i), 1) for i in batch_pred_ids])
    return preds

def load_and_autolabel_dataset(
    input_file="data/processed/roblox_cleaned.csv",
    text_col="cleaned_content",
    teacher_model_name="w11wo/indonesian-roberta-base-sentiment-classifier",
    max_samples=None,
    seed=42,
    teacher_batch_size=64,
    max_length=128
):
    # Auto-labeling mode (no rating and no manual annotation).
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)
    df = pd.read_csv(input_file)
    if text_col not in df.columns:
        raise ValueError(f"Kolom teks tidak ditemukan: {text_col}")

    df = df[[text_col]].copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df = df[df[text_col].str.len() > 0]

    if max_samples is not None and len(df) > int(max_samples):
        df = df.sample(n=int(max_samples), random_state=int(seed)).reset_index(drop=True)

    # Keep preprocessing consistent with student IndoBERT.
    df["cleaned_content"] = df[text_col].apply(preprocess_text)
    texts = df["cleaned_content"].tolist()

    labels = auto_label_with_teacher(
        texts,
        teacher_model_name=teacher_model_name,
        batch_size=int(teacher_batch_size),
        max_length=int(max_length)
    )
    df["label"] = labels
    df["sentiment"] = df["label"].map({0: "negatif", 1: "netral", 2: "positif"})
    print(f"Auto-labeled dataset shape: {df.shape}")
    print("Label distribution:")
    print(df["label"].value_counts())
    return df

# 3. Custom Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 4. Handle Class Imbalance
def compute_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

# 5. Fine-tune IndoBERT
def train_indobert_model(df, output_dir, num_epochs=3, lr=2e-5, batch_size=16, max_length=128, test_size=0.2, val_size_in_train=0.1):
    print("\nInitializing IndoBERT tokenizer and model...")

    # Load tokenizer and model
    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # Prepare data
    texts = df['cleaned_content'].tolist()
    labels = df['label'].tolist()

    # Split data (requested: 80/20 train/test). Validation is taken from the train split.
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=float(test_size), random_state=42, stratify=labels
    )
    if val_size_in_train and float(val_size_in_train) > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=float(val_size_in_train),
            random_state=42,
            stratify=train_labels
        )
    else:
        val_texts, val_labels = [], []

    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}, Test size: {len(test_texts)}")

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length=max_length) if len(val_texts) > 0 else None
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length=max_length)

    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    print(f"Class weights: {class_weights}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset is not None else None

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Loss with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = model.to(device)
    model.train()

    best_val_accuracy = 0.0
    best_model_path = output_dir
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        epoch_loss = 0.0
        step = 0

        for batch in tqdm(train_loader, desc='Training', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            if loss is None:
                logits = outputs.logits
                loss = loss_fn(logits, labels_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        epoch_loss /= max(step, 1)
        print(f"Epoch {epoch} training loss: {epoch_loss:.4f}")

        # Evaluate on validation set (if available)
        if val_loader is not None:
            model.eval()
            val_preds = []
            val_labels_eval = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation', leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels_batch = batch['labels'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1)

                    val_preds.extend(preds.cpu().numpy())
                    val_labels_eval.extend(labels_batch.cpu().numpy())

            val_accuracy = accuracy_score(val_labels_eval, val_preds)
            print(f"Epoch {epoch} validation accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"Saving best model (val accuracy improved to {best_val_accuracy:.4f})")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
        else:
            # If no validation split is configured, always save the latest checkpoint.
            print("No validation split. Saving current checkpoint.")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)

        model.train()

    print("Model training completed.")

    return model, test_dataset, tokenizer

# 6. Evaluate Model
def evaluate_model(model, test_dataset, tokenizer, output_dir):
    print("\nEvaluating model on test set...")
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())

    accuracy = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    target_names = ['negatif', 'netral', 'positif']
    report = classification_report(labels, preds, target_names=target_names)
    print("\nClassification Report:")
    print(report)

    cm = confusion_matrix(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2], average='macro', zero_division=0
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - IndoBERT Sentiment Analysis')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    metrics = {
        "test_accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "classification_report": report
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return accuracy, report, cm, metrics

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str, default=None)
    parser.add_argument("--text_col", type=str, default="cleaned_content")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--teacher_model", type=str, default="w11wo/indonesian-roberta-base-sentiment-classifier")
    parser.add_argument("--teacher_batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs('./models/indobert_sentiment', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    print("=" * 80)
    print("INDOBERT SENTIMENT ANALYSIS - FINE-TUNING")
    print("=" * 80)
    print(f"Device: {device}")

    # Select output directory default per mode.
    if args.output_dir is None:
        if args.annotation_file:
            args.output_dir = "./models/indobert_sentiment/best_model"
        else:
            args.output_dir = "./models/indobert_sentiment/best_model_annotated"
    print(f"Output dir: {args.output_dir}")

    if args.annotation_file:
        # Mode lama: supervised fine-tuning dari anotasi manual (file).
        df = load_annotated_data(args.annotation_file, text_col=args.text_col, label_col=args.label_col)
    else:
        # Mode baru: auto-labeling dari teks menggunakan model teacher, lalu fine-tune student IndoBERT.
        print("Mode: auto-labeling + fine-tuning (no rating, no manual annotation)")
        print("Teacher rationale: uses a pretrained Indonesian sentiment classifier to provide text-based labels for bootstrapping.")
        df = load_and_autolabel_dataset(
            input_file="data/processed/roblox_cleaned.csv",
            text_col=args.text_col,
            teacher_model_name=args.teacher_model,
            max_samples=args.max_samples,
            seed=42,
            teacher_batch_size=args.teacher_batch_size,
            max_length=args.max_length
        )
    
    if df is None:
        print("\n[ERROR] Failed to load data. Exiting...")
        exit(1)

    # Auto-labeling mode already produces 'cleaned_content' and 'label'.
    if args.annotation_file:
        pass
    
    if df is None or len(df) == 0:
        print("\n[ERROR] Failed to prepare data. Exiting...")
        exit(1)

    # Step 3-5: Train model
    print("\n" + "=" * 80)
    print("TRAINING INDOBERT MODEL")
    print("=" * 80)
    try:
        model, test_dataset, tokenizer = train_indobert_model(
            df,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            test_size=0.2,
            val_size_in_train=0.1
        )
        
        # Step 6: Evaluate
        print("\n" + "=" * 80)
        print("EVALUATING MODEL")
        print("=" * 80)
        accuracy, report, cm, metrics = evaluate_model(model, test_dataset, tokenizer, output_dir=args.output_dir)

        print("\n" + "=" * 80)
        print("TRAINING AND EVALUATION COMPLETED!")
        print("=" * 80)
        print(f"Final Test Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")

        if accuracy < 0.80:
            print("\nSuggestions to improve accuracy:")
            print("1. Increase training epochs (current: 5)")
            print("2. Try different learning rates")
            print("3. Use data augmentation for minority classes")
            print("4. Fine-tune hyperparameters")
            print("5. Consider ensemble methods")
            print("6. Add more training data if possible")
        else:
            print("\n[OK] Model achieved good accuracy! Consider using it for inference.")
            
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
