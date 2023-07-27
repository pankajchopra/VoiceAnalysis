# Import libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

# Prepare input texts
texts = ["I'm so happy today!", "This is so frustrating.", "I don't care about anything."]

# Tokenize input texts
inputs = tokenizer(texts, padding=True, return_tensors="pt")

# Get model outputs
outputs = model(**inputs)

# Get probabilities for each label
probs = torch.sigmoid(outputs.logits)

# Apply threshold to get binary predictions
preds = (probs > 0.5).int()

# Get label names
labels = model.config.id2label.values()

# Print results
for i, text in enumerate(texts):
    print(f"Text: {text}")
    print(f"Probabilities: {probs[i]}")
    print(f"Predictions: {preds[i]}")
    print(f"Labels: {[label for j, label in enumerate(labels) if preds[i][j] == 1]}")
    print()
