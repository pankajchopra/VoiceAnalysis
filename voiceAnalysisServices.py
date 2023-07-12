import speech_recognition as sr
import streamlit as st
# import streamlit
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import flair

# import gradio as gr
# import whisper

# Twitter-roberta-base-sentiment is a roBERTa model trained on ~58M tweets and fine-tuned for sentiment analysis. Fine-tuning is the process of taking a pre-trained large language model (e.g. roBERTa in this case) and then tweaking it with additional training data to make it perform a second similar task (e.g. sentiment analysis).
# Bert-base-multilingual-uncased-sentiment is a model fine-tuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian.
# Distilbert-base-uncased-emotion is a model fine-tuned for detecting emotions in texts, including sadness, joy, love, anger, fear and surprise.
#DistilBERT is a smaller, faster and cheaper version of BERT. It has 40% smaller than BERT and runs 60% faster while preserving over 95% of BERT’s performance.
# bhadresh-savani/bert-base-uncased-emotion gives better all emotions levels

distilbert_base_uncased_model="distilbert-base-uncased-finetuned-sst-2-english"
bhadresh_savani_bert_base_uncased_emotion_model="bhadresh-savani/bert-base-uncased-emotion"




def perform_sentiment_analysis(text, return_all, model):
    if 'current_model' not in st.session_state:
        st.session_state['current_model'] = 'distilbert'
    print(f' Model parameter is {model}')
    if model=='distilbert':
        st.session_state['current_model'] = model
        print(f' Using Model {model}')
        return perform_sentiment_analysis_using_distilbert(text, return_all)
    elif model=='vader':
        st.session_state['current_model'] = model
        print(f' Using Model {model}')
        return sentiment_analysis_vader(text)
    elif model=='roberta':
        st.session_state['current_model'] = model
        print(f' Using Model {model}')
        return perform_sentiment_analysis_using_sam_lowe(text, return_all)
    elif model=='flair':
        print(f' Using Model {model}')
        st.session_state['current_model'] = model
        print(f' Using Model {model}')
        return perform_sentiment_analysis_using_flair(text, return_all)
    else:
        return perform_sentiment_analysis_using_distilbert(text, return_all)


def perform_sentiment_analysis_using_flair(text, return_all):
    try:
        #download mode
        flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
        s = flair.data.Sentence(text)
        flair_sentiment.predict(s)
        print(s.score)
        print(s.get_label().value)
        total_sentiment = s.labels
        model = st.session_state['current_model']
        print(f'Sentiment analysis {model} results are {total_sentiment}')
        if return_all:
            return s
        else:
            if s:
                sentiment_label = s.get_label().value
                sentiment_score = s.score
                return sentiment_label, sentiment_score
            else:
                return 'bad_data', 'Not Enough or Bad Data'
    except Exception as ex:
        print("Error occurred during .. perform_sentiment_analysis_using_distilbert")
        print(str(ex))
        return "error", str(ex)


def perform_sentiment_analysis_using_distilbert(text, return_all):
    current_model = st.session_state['current_model']
    try:
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        sentiment_analysis = pipeline("sentiment-analysis", model=model_name, return_all_scores=return_all)
        results = sentiment_analysis(text)
        print(f'Sentiment analysis {current_model} results are {results}')
        if return_all:
            return results[0]
        else:
            if(results[0]):
                sentiment_label = results[0]['label']
                sentiment_score = results[0]['score']
                return sentiment_label, sentiment_score
            else:
                return 'bad_data', 'Not Enough or Bad Data'
    except Exception as ex:
        print("Error occurred during .. perform_sentiment_analysis_using_distilbert")
        print(str(ex))
        return "error", str(ex)


def perform_text_classification_using_bhadresh_savani(text, return_all):
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    classification = pipeline("text-classification", model=model_name, return_all_scores=return_all)
    results = classification(text)
    print(f'Text Classification Analysis results are {results}')
    if return_all:
        return results[0]
    else:
        if results[0]:
            sentiment_label = results[0]['label']
            sentiment_score = results[0]['score']
            return sentiment_label, sentiment_score
        else:
            return 'bad_data', 'Not Enough or Bad Data'


def perform_sentiment_analysis_using_sam_lowe(text, return_all):
    current_model = st.session_state['current_model']
    model_name = "SamLowe/roberta-base-go_emotions"
    classification = pipeline("sentiment-analysis", model=model_name, return_all_scores=return_all)
    results = classification(text)
    print(f'Sentimental Analysis  {current_model} results are {results}')
    if return_all:
        return results[0]
    else:
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
    try:
        r = sr.Recognizer()
        transcribed_text1 = r.recognize_google(audio_data)
        print(r.recognize_google(audio_data, language='en-US'))
    except sr.UnknownValueError:
        print("Google could not understand audio")
    except sr.RequestError as e:
        print("Google error; {0}".format(e))
    return transcribed_text1


def text_blob_sentiments(text):
    # Create a TextBlob object
    output = TextBlob(text)
    if output:
        sentiments = output.sentiment
        # Extract words from object
        word_count = output.word_counts
        # print(sentiments, word_count)
        sentiment_results = dict()
        i = 1
        for sentence in output.sentences:
            print(sentence)
            # print(sentence.string)
            sentiment_results['sentence{}'.format(i)] = sentence.string
            sentiment_results[f'sentence_assessment{i}'.format(i)] = sentence.sentiment_assessments
            i = i+1
        # print(sentiment_results)
        return sentiments, word_count, sentiment_results


# st.write(f'Sentence Polarity:{sentence.sentiment["polarity"]}')
def analyze_sentiment(text):
    sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")
    results = sentiment_analysis(text)
    sentiment_results = {result['label']: result['score'] for result in results}
    return sentiment_results


# def inference(audio, sentiment_option):
#     model = whisper.load_model("base")
#
#     audio = whisper.load_audio(audio)
#     audio = whisper.pad_or_trim(audio)
#
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
#     _, probs = model.detect_language(mel)
#     lang = max(probs, key=probs.get)
#
#     options = whisper.DecodingOptions(fp16=False)
#     result = whisper.decode(model, mel, options)
#
#     sentiment_results = analyze_sentiment(result.text)
#     return sentiment_results
#
#     sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)
#
#     return lang.upper(), result.text, sentiment_output


# function to print sentiments
# of the sentence.
def sentiment_analysis_vader(sentence):
    current_model = st.session_state['current_model']
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    print(f'Semtimental Analysis  {current_model} results are {sentiment_dict}')
    if sentiment_dict:
        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            sentiment_label = 'Positive'
            sentiment_score = sentiment_dict['pos']
        elif sentiment_dict['compound'] <= - 0.05:
            sentiment_label = 'Negative'
            sentiment_score = sentiment_dict['neg']
        else:
            sentiment_label = 'Neutral'
            sentiment_score = sentiment_dict['neu']
        return sentiment_label, sentiment_score
    else:
        return 'bad_data', 'Bad Data or Insufficient Data'

