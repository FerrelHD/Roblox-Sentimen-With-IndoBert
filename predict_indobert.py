import re
import os

# Try to import torch and transformers, handle if not available
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/transformers not available. IndoBERT prediction will not work.")

class IndoBERTSentimentPredictor:
    def __init__(self, model_path='./models/indobert_sentiment/best_model'):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for IndoBERT prediction. Please install torch and transformers.")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        annotated_path = './models/indobert_sentiment/best_model_annotated'
        selected_path = annotated_path if os.path.exists(annotated_path) else model_path
        print(f"Loading IndoBERT model from {selected_path} on {self.device}")

        if not os.path.exists(selected_path):
            raise FileNotFoundError(f"Model path {selected_path} not found. Please run training first.")

        self.tokenizer = AutoTokenizer.from_pretrained(selected_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(selected_path)
        self.model.to(self.device)
        self.model.eval()

        self.label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}

    def preprocess_text(self, text):
        """Clean text for IndoBERT"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text.lower()

    def predict(self, text):
        """Predict sentiment for a single text"""
        # Preprocess
        cleaned_text = self.preprocess_text(text)

        # Tokenize
        inputs = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        sentiment = self.label_map[predicted_class]

        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negatif': probabilities[0][0].item(),
                'netral': probabilities[0][1].item(),
                'positif': probabilities[0][2].item()
            }
        }

    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        try:
            _predictor = IndoBERTSentimentPredictor()
        except (ImportError, FileNotFoundError) as e:
            _predictor = None
            print(f"IndoBERT predictor not available: {e}")
    return _predictor

def predict_sentiment(text):
    """Convenience function for single prediction"""
    if not TORCH_AVAILABLE:
        return {
            'sentiment': 'error',
            'confidence': 0.0,
            'probabilities': {'negatif': 0.0, 'netral': 0.0, 'positif': 0.0},
            'error': 'PyTorch not available'
        }
    
    predictor = get_predictor()
    if predictor is None:
        return {
            'sentiment': 'error',
            'confidence': 0.0,
            'probabilities': {'negatif': 0.0, 'netral': 0.0, 'positif': 0.0},
            'error': 'IndoBERT model not found'
        }
    
    try:
        return predictor.predict(text)
    except Exception as e:
        return {
            'sentiment': 'error',
            'confidence': 0.0,
            'probabilities': {'negatif': 0.0, 'netral': 0.0, 'positif': 0.0},
            'error': str(e)
        }

def predict_sentiments(texts):
    """Convenience function for batch prediction"""
    predictor = get_predictor()
    return predictor.predict_batch(texts)

if __name__ == "__main__":
    # Test the predictor
    predictor = IndoBERTSentimentPredictor()

    test_texts = [
        "Game ini sangat bagus dan seru!",
        "Roblox sering lag dan error",
        "Game nya biasa saja sih"
    ]

    print("Testing IndoBERT Sentiment Predictor:")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")
