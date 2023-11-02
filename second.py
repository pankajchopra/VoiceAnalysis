import time
import streamlit as st
from voiceAnalysisServices import VoiceAnalysisServices

def main():
    voiceAnalysisServices = VoiceAnalysisServices()
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
    resultDict = voiceAnalysisServices.perform_sentiment_analysis(text, False, 'flair')
    print(f"result:{resultDict}")



if __name__ == "__main__":
    main()