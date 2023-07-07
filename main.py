import os
import traceback
import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from os import path


def perform_sentiment_analysis(text):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)


def perform_sentiment_analysis(text):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)
    results = sentiment_analysis(text)
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']
    return sentiment_label, sentiment_score


def transcribe_audio_file(audio_file):
    with sr.WavFile(audio_file) as source:
        r = sr.Recognizer()
        audio = r.record(source)
        try:
            transcribed_text1 = r.recognize_google(audio)
            print(r.recognize_google(audio, language='en-US'))
        except sr.UnknownValueError:
            print("Google could not understand audio")
        except sr.RequestError as e:
            print("Google error; {0}".format(e))
        return transcribed_text1


def transcribe_audio_data(audio_data):
    recognizer = sr.Recognizer()
    # with sr.AudioData(audio_data) as source:
    #     audio = recognizer.record(source)
    try:
        # transcribed_text1 = recognizer.recognize_google(audio_data)
        print("Google thinks you said " + recognizer.recognize_google(audio_data))
    except sr.UnknownValueError:
        print("Google could not understand audio")
    except sr.RequestError as e:
        print("Google error; {0}".format(e))


st.set_page_config(layout="wide")
# col1, col2 = st.columns([1,1])

st.markdown(' # AI to Detect Sentiment in the Audio!')
st.markdown('Pre-recorded audio and live audio analysis')
#  app.py
st.sidebar.title("Audio Analysis")
st.sidebar.write("""The Audio Analysis app is a powerful tool that allows you to analyze audio files 
                 and gain valuable insights from them. It combines speech recognition 
                 and sentiment analysis techniques to transcribe the audio 
                 and determine the sentiment expressed within it.""")


# AUDIO_FILE1 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0061_8k.wav")
# print(AUDIO_FILE1)
# AUDIO_FILE2 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0040_8k.wav")
# print(AUDIO_FILE2)
AUDIO_FILE3 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0019_8k.wav")
st.write(AUDIO_FILE3)
transcribed_text = transcribe_audio_file(AUDIO_FILE3)
sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text)
st.header("Transcribed Text")
st.text_area("Transcribed Text", transcribed_text, height=200)
st.header("Sentiment Analysis")
negative_icon = "👎"
neutral_icon = "😐"
positive_icon = "👍"
if sentiment_label == "NEGATIVE":
  st.write(f"{negative_icon} Negative (Score: {sentiment_score})", unsafe_allow_html=True)
else:
  st.empty()

if sentiment_label == "NEUTRAL":
  st.write(f"{neutral_icon} Neutral (Score: {sentiment_score})", unsafe_allow_html=True)
else:
  st.empty()

if sentiment_label == "POSITIVE":
  st.write(f"{positive_icon} Positive (Score: {sentiment_score})", unsafe_allow_html=True)
else:
  st.empty()
