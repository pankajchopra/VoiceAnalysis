from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Define the model and tokenizer names
# model_name = "SamLowe/roberta-base-go_emotions"
model_name ="bhadresh-savani/bert-base-uncased-emotion"
# Directory to save the model
# model_directory = "roberta-base-go_emotions"
model_directory = "bhadresh-savani-bert-base-uncased-emotion"

# Create the model directory if it doesn't exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Create a path to the file you want to reach
# current_directory = os.getcwd()
folder_path = "\\Users\\panka\\projects\\VoiceAnalysis\\roberta-base-go_emotions"
# folder_path = "\\Users\\panka\\projects\\VoiceAnalysis\\bhadresh-savani-bert-base-uncased-emotion"
print(folder_path)

model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions", cache_dir=folder_path)
# model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion", cache_dir=folder_path)

model.save_pretrained(folder_path)
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
# tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")

tokenizer.save_pretrained(folder_path)
# model = AutoModelForSequenceClassification.from_pretrained(folder_path)
# tokenizer = AutoTokenizer.from_pretrained(folder_path)
# Classify a text sequence
# text = "I love this product! It's amazing."
# inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
# outputs = model(**inputs)
#
# sentiment_scores = outputs.logits
# Example labels for the model
# labels = ["anger", "joy", "optimism", "sadness", "surprise", "neutral"]
#
# Print the predicted scores for each emotion
# for label, score in zip(labels, sentiment_scores.tolist()[0]):
#     print(f"{label}: {score:.4f}")