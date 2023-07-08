import speech_recognition as sr
from transformers import pipeline
# import gradio as gr
# import whisper

# Twitter-roberta-base-sentiment is a roBERTa model trained on ~58M tweets and fine-tuned for sentiment analysis. Fine-tuning is the process of taking a pre-trained large language model (e.g. roBERTa in this case) and then tweaking it with additional training data to make it perform a second similar task (e.g. sentiment analysis).
# Bert-base-multilingual-uncased-sentiment is a model fine-tuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian.
# Distilbert-base-uncased-emotion is a model fine-tuned for detecting emotions in texts, including sadness, joy, love, anger, fear and surprise.
#DistilBERT is a smaller, faster and cheaper version of BERT. It has 40% smaller than BERT and runs 60% faster while preserving over 95% of BERT’s performance.
# bhadresh-savani/bert-base-uncased-emotion gives better all emotions levels

distilbert_base_uncased_model="distilbert-base-uncased-finetuned-sst-2-english"
bhadresh_savani_bert_base_uncased_emotion_model="bhadresh-savani/bert-base-uncased-emotion"

def perform_sentiment_analysis_using_distilbert(text, return_all):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name, return_all_scores=return_all)
    results = sentiment_analysis(text)
    print(f'Sentiment analysis results are {results}')
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']
    return sentiment_label, sentiment_score


def perform_text_classification_using_bhadresh_savani(text, return_all):
    model_name = "bhadresh-savani/bert-base-uncased-emotion"
    classification = pipeline("text-classification", model=model_name, return_all_scores=return_all)
    results = classification(text)
    print(f'Text Classification Analysis results are {results}')
    if(return_all):
        return results[0]
    else:
        sentiment_label = results[0]['label']
        sentiment_score = results[0]['score']
        return sentiment_label, sentiment_score


def perform_sentiment_analysis_using_sam_lowe(text, return_all):
    model_name = "SamLowe/roberta-base-go_emotions"
    classification = pipeline("sentiment-analysis", model=model_name, return_all_scores=return_all)
    results = classification(text)
    print(f'Text Classification Analysis results are {results}')
    if(return_all):
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


def transcribe_audio_data(audio_data, r: sr.Recognizer):
    try:
        transcribed_text1 = r.recognize_google(audio_data)
        print(r.recognize_google(audio_data, language='en-US'))
    except sr.UnknownValueError:
        print("Google could not understand audio")
    except sr.RequestError as e:
        print("Google error; {0}".format(e))
    return transcribed_text1


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

