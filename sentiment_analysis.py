from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define the model and tokenizer names
# model_name = "SamLowe/roberta-base-go_emotions"
model_name ="bhadresh-savani/bert-base-uncased-emotion"

# Directory to save the model
# model_directory = "roberta-base-go_emotions"
model_directory = "bhadresh-savani-bert-base-uncased-emotion"
text = "I love you"

# Now you can use the model for sentiment analysis
# Load the model and tokenizer from the folder
model = AutoModelForSequenceClassification.from_pretrained(model_directory)
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# Tokenize the text and obtain the sentiment scores
inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
outputs = model(**inputs)

# Extract the predicted sentiment scores
sentiment_scores = outputs.logits

# The sentiment scores represent the probabilities of different emotions
# You can interpret the scores based on the model's labels

# Example labels for the model
emotion_labels = tokenizer.model_input_names
print(emotion_labels)
labels = ["anger", "joy", "optimism", "sadness", "surprise", "neutral", "positive"]
# # Print the index and label of each emotion
# for index, emotion_label in enumerate(emotion_labels):
#     labels = labels.append(emotion_label)
#     print(index, emotion_label)

# Print the predicted scores for each emotion
all_emotions_scores = sentiment_scores.tolist()[0]
print(all_emotions_scores)
for label, score in zip(labels, all_emotions_scores):
    print(f"{label}: {score:.4f}")

most_prominent_emotion = all_emotions_scores.index(max(all_emotions_scores))
print(most_prominent_emotion)
