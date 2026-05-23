from transformers import AutoTokenizer, AutoModelForSequenceClassification

print('transformers import OK')
model_name = 'indobenchmark/indobert-base-p1'

tokenizer = AutoTokenizer.from_pretrained(model_name)
print('tokenizer loaded')

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    ignore_mismatched_sizes=True,
)
print('model loaded')
