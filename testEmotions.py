import streamlit as st
from textblob import TextBlob


# Function to classify emotions based on sentiment scores
def classify_emotion(sentiment):
    if sentiment.polarity > 0.2:
        return "Happy"
    elif sentiment.polarity < -0.2:
        return "Angry"
    elif sentiment.subjectivity < 0.5:
        return "Fear"
    else:
        return "Sad"


# Streamlit app
def main():
    st.title("Sentiment Analysis and Emotion Classification")

    # Input text from user
    user_input = st.text_input("Enter text:")

    if user_input:
        # Perform sentiment analysis
        blob = TextBlob(user_input)
        sentiment = blob.sentiment

        # Classify emotion
        emotion = classify_emotion(sentiment)

        # Display sentiment scores and emotion
        st.write("Sentiment Polarity:", sentiment.polarity)
        st.write("Sentiment Subjectivity:", sentiment.subjectivity)
        st.write("Emotion:", emotion)


# Run the app
if __name__ == "__main__":
    main()
