from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

roberta_labels = ["admiration", "amusement", "anger", "annoyance", "approval",
                  "caring", "confusion", "curiosity", "desire", "disappointment",
                  "disapproval", "disgust", "embarrassment", "excitement", "fear",
                  "gratitude", "grief", "joy", "love", "nervousness",
                  "optimism", "pride", "realization", "relief", "remorse", "sadness",
                  "surprise", "neutral"]

savani_labels = ["anger", "joy", "optimism", "sadness", "surprise", "neutral", "positive"]

def sentiment_analysis(model_name, model_directory, text, emotions_labels):
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
    # print(emotion_labels)

    # Print the predicted scores for each emotion
    all_emotions_scores = sentiment_scores.tolist()[0]
    # print(all_emotions_scores)
    # for label, score in zip(emotions_labels, all_emotions_scores):
    #     print(f"{label}: {score:.4f}")

    most_prominent_emotion = all_emotions_scores.index(max(all_emotions_scores))
    print("Sentiment is:" + emotions_labels[most_prominent_emotion] + ":" + str(all_emotions_scores[most_prominent_emotion]))


if __name__ == '__main__':
    # Define the model and tokenizer names
    model_name = "SamLowe/roberta-base-go_emotions"
    # Directory to save the model
    model_directory = "roberta-base-go_emotions"
    text = "I really love you, I can do anything for you."
    sentiment_analysis(model_name, model_directory, text, roberta_labels)

    # Define the model and tokenizer names
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    # Directory to save the model
    model_directory = "bhadresh-savani-bert-base-uncased-emotion"
    text = "I really love you, I can do anything for you."
    sentiment_analysis(model_name, model_directory, text, savani_labels)
