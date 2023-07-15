import time
import streamlit as st
from voiceAnalysisServices import perform_sentiment_analysis, perform_text_classification_using_bhadresh_savani, perform_sentiment_analysis

def main():
    text = [
        "I love this product",
        "This is a terrible product",
        "It works fine",
        "I hate this product",
        "This is an amazing product",
        "I am happy",
        "I am so greatful"
    ]
    resultDict = dict()
    resultDict = perform_sentiment_analysis(text, False, 'flair')
    print(resultDict)



if __name__ == "__main__":
    main()