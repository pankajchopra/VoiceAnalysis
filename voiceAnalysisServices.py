import speech_recognition as sr
import streamlit as st
from transformers import pipeline
from textblob import TextBlob
import pandas as pd
import re
import gc
import flair
from flair.data import Sentence
from loadModules import LoadModules
from nltk.tokenize import sent_tokenize
import os

# from google.cloud import speech_v1p1beta1 as speech
# import traceback
# from punctuator import Punctuator

# import gradio as gr
# import whisper

# Twitter-roberta-base-sentiment is a roBERTa model trained on ~58M tweets and fine-tuned for sentiment analysis. Fine-tuning is the process of taking a pre-trained large language model (e.g. roBERTa in this case) and then tweaking it with additional training data to make it perform a second similar task (e.g. sentiment analysis).
# Bert-base-multilingual-uncased-sentiment is a model fine-tuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian.
# Distilbert-base-uncased-emotion is a model fine-tuned for detecting emotions in texts, including sadness, joy, love, anger, fear and surprise.
# DistilBERT is a smaller, faster and cheaper version of BERT. It has 40% smaller than BERT and runs 60% faster while preserving over 95% of BERT’s performance.
# bhadresh-savani/bert-base-uncased-emotion gives better all emotions levels

distilbert_base_uncased_model = "distilbert-base-uncased-finetuned-sst-2-english"
bhadresh_savani_bert_base_uncased_emotion_model = "bhadresh-savani/bert-base-uncased-emotion"

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./voiceanalysisproject-64b7cf2cc8dd.json"


class VoiceAnalysisServices(LoadModules):
    load_Modules = LoadModules(True)

    def __init__(self):
        print('in VoiceAnalysisServices constructor')
        if self.load_Modules is None:
            self.load_Modules = LoadModules(True)
        else:
            print('load_Modules already loaded')

    def perform_sentiment_analysis(self, text, return_all, model, isFileUpload=False):
        if 'current_model' not in st.session_state:
            print(f'Setting the current_model as {model}')
            st.session_state['current_model'] = model

        print(f' Model parameter is {model}')
        if model == 'distilbert':
            st.session_state['current_model'] = model
            print(f' Using Model {model}')
            return self.perform_sentiment_analysis_using_distilbert(text, return_all)
        elif model == 'vader':
            st.session_state['current_model'] = model
            print(f' Using Model {model}')
            return self.perform_sentiment_analysis_using_vader(text)
        elif model == 'roberta':
            st.session_state['current_model'] = model
            print(f' Using Model {model}')
            # set to all to get all emotions
            return_all = True
            return self.perform_sentiment_analysis_using_sam_lowe(text, return_all)
        elif model == 'flair':
            print(f' Using Model {model}')
            st.session_state['current_model'] = model
            return self.perform_sentiment_analysis_using_flair(text, return_all)
        elif model == 'savani':
            print(f' Using Model {model}')
            st.session_state['current_model'] = model
            if isinstance(text, str) or isinstance(text, list):
                return self.perform_text_classification_using_old_bhadresh_savani(text, return_all)
                # return result.iloc[0]['sentiment'], result.iloc[0]['polarity']
            elif isinstance(text, pd.DataFrame):
                return self.perform_text_classification_using_bhadresh_savani(text, False)
                # return result.iloc[0]['sentiment'], result.iloc[0]['polarity']
            else:
                print("Error!! wrong text type!!!!!!!!!!!!!!!!!!!!")
        elif model == 'textblob':
            print(f' Using Model {model}')
            st.session_state['current_model'] = model
            return self.perform_sentiment_analysis_using_textblob(text)
        elif model == 'All':
            print(f' Using Model {model}')
            st.session_state['current_model'] = model
            print(f' Using Model {model}')
            return self.perform_sentiment_analysis_all(text)
        else:
            return self.perform_sentiment_analysis_using_flair(text, return_all)

    def perform_sentiment_analysis_using_flair(self, text, return_all):
        try:
            flair_sentiment = None
            if self.all_modules and 'flair' in self.all_modules.keys():
                print('Found flair_sentiment in LoadModules.all_modules')
                flair_sentiment = self.all_modules['flair']
            else:
                print('Not found flair_sentiment LoadModules.all_modules, loading...')
                flair_sentiment = self.load_Modules.load_model_flair()
                # print(LoadModules.all_modules.keys())

            if type(text) == list:
                sentences = [Sentence(sent, use_tokenizer=False) for sent in text]
            else:
                sentences = [Sentence(text)]

            for sentence in sentences:
                flair_sentiment.predict(sentence)

            # Aggregate sentiment scores for the entire paragraph
            overall_sentiment_score = sum([sentence.labels[0].score for sentence in sentences]) / len(sentences)
            overall_sentiment_label = self.calcSentiment(overall_sentiment_score, -0.001, 0.001)
            print(overall_sentiment_score)
            print(overall_sentiment_label)
            # model = st.session_state['current_model']
            print(f'Sentiment analysis (flair) results are {overall_sentiment_score}')
            if return_all:
                return overall_sentiment_label, overall_sentiment_score
            elif overall_sentiment_score:
                return overall_sentiment_label, overall_sentiment_score
            else:
                return 'bad_data', 'Not Enough or Bad Data'
        except Exception as ex:
            print("Error occurred during .. perform_sentiment_analysis_using_flair")
            print(str(ex))
            return "error", str(ex)

    def calcSentiment(self, score, min, max):
        if score > min and score < max:
            return 'Neutral'
        elif score <= min:
            return 'Negative'
        elif score >= max:
            return 'Positive'

    def calcSentimentWithLabelAlso(self, result):
        negatives = sum(r['score'] for r in result if r['label']=='NEGATIVE')
        negatives_count = sum(1 for r in result if r['label'] == 'NEGATIVE')
        positives = sum(r['score'] for r in result if r['label'] == 'POSITIVE')
        positives_count = sum(1 for r in result if r['label'] == 'POSITIVE')
        if positives_count> negatives_count:
            return 'POSITIVE', positives / positives_count
        elif positives_count == negatives_count:
            return 'NEUTRAL', positives / positives_count
        else:
            return 'NEGATIVE', negatives / negatives_count

    def perform_sentiment_analysis_all(self, text):
        print('In perform_sentiment_analysis_all')
        sentiment_and_scores = dict()
        sentiment_label, sentiment_score = self.perform_sentiment_analysis_using_textblob(text)
        sentiment_and_scores['textblob'] = {'sentiment_label': sentiment_label, 'sentiment_score': sentiment_score}
        sentiment_label, sentiment_score = self.perform_sentiment_analysis_using_flair(text, False)
        sentiment_and_scores['flair'] = {'sentiment_label': sentiment_label, 'sentiment_score': sentiment_score}
        sentiment_label, sentiment_score = self.perform_sentiment_analysis_using_vader(text)
        sentiment_and_scores['vader'] = {'sentiment_label': sentiment_label, 'sentiment_score': sentiment_score}
        sentiment_label, sentiment_score = self.perform_sentiment_analysis_using_sam_lowe(text, False)
        sentiment_and_scores['roberta'] = {'sentiment_label': sentiment_label, 'sentiment_score': sentiment_score}
        sentiment_label, sentiment_score = self.perform_sentiment_analysis_using_distilbert(text, False)
        sentiment_and_scores['distilbert'] = {'sentiment_label': sentiment_label, 'sentiment_score': sentiment_score}
        return sentiment_and_scores

    def perform_sentiment_analysis_using_textblob(self, text):
        if isinstance(text, list):
            text = ' '.join(text)
            print(' text in perform_sentiment_analysis_using_textblob:' + text)
        try:
            print('Nothing to Load for textBlob')
            sentiment_label, sentiment_score = self.text_blob_sentiments(text)
            print(f'Sentiment analysis [TextBlob] results are {sentiment_label} ({sentiment_score})')
            return sentiment_label, sentiment_score
        except Exception as ex:
            print("Error occurred during .. perform_sentiment_analysis_using_textblob")
            print(str(ex))
            return "error", str(ex)

    def perform_sentiment_analysis_using_distilbert(self, paragraph, return_all):
        if type(paragraph) != list:
            paragraph = sent_tokenize(paragraph)

        print("distilbert - number lines in a paragraph" + str(len(paragraph)))
        # current_model = st.session_state['current_model']
        try:
            if self.all_modules and 'distilbert' in self.all_modules.all_modules.keys():
                print('Found distilbert_sentiment_analysis in global')
                distilbert_sentiment_analysis = self.all_modules['distilbert']
            else:
                print('Not found distilbert_sentiment_analysis global, loading...')
                distilbert_sentiment_analysis = self.load_Modules.load_model_distilbert()

            results = self.analyze_sentiment_per_sentence(paragraph, distilbert_sentiment_analysis)
            overall_sentiment_label, overall_sentiment_score = self.calcSentimentWithLabelAlso(results)
            print(f'Sentiment analysis (distilbert-base-uncased-finetuned-sst-2-english) results are {results}')
            if return_all:
                return overall_sentiment_label, overall_sentiment_score
            elif overall_sentiment_label:
                return overall_sentiment_label, overall_sentiment_score
            else:
                return 'bad_data', 'Not Enough or Bad Data'
        except Exception as ex:
            print("Error occurred during .. perform_sentiment_analysis_using_distilbert")
            print(str(ex))
            return "error", str(ex)

    def analyze_sentiment_per_sentence(self, sentences, distilbert_sentiment_analysis):

        # Analyze sentiment for each sentence
        sentence_sentiments = []
        for sentence in sentences:
            result = distilbert_sentiment_analysis(sentence)
            sentiment_label = result[0][0]['label']
            sentiment_score = result[0][0]['score']
            sentence_sentiments.append({'sentence': sentence, 'label': sentiment_label, 'score': sentiment_score})

        return sentence_sentiments

    # Text Classification input text has to be in the dataframe
    def perform_text_classification_using_bhadresh_savani(self, text_in_a_dataframe, return_all):
        # print("text_in_a_dataframe:"+text_in_a_dataframe)
        # if type(text_in_a_dataframe) == list or isinstance(text_in_a_dataframe, list):
        #     t_text = ' '.join(text_in_a_dataframe)
        #     text_in_a_dataframe = pd.DataFrame({"text": [t_text]})
        #     print("it is a list")
        # elif type(text_in_a_dataframe) == str:
        #     text_in_a_dataframe = pd.DataFrame({"text": [text_in_a_dataframe]})
        #     print("it is a str")

        # model_name = "bhadresh-savani/bert-base-uncased-emotion"
        # savani_classification = pipeline("text-classification", model=model_name, return_all_scores=return_all)
        if self.all_modules and 'savani' in self.all_modules.all_modules.keys():
            print('Found savani_classification in LoadModules.all_modules')
            savani_classification = self.all_modules.all_modules['savani']
        else:
            print('Not found savani_classification LoadModules.all_modules, loading')
            savani_classification = self.load_Modules.load_model_bhadresh_savani()

        text_in_a_dataframe['result'] = text_in_a_dataframe["text"].apply(savani_classification)

        text_in_a_dataframe['result'] = text_in_a_dataframe['result'].apply(lambda x: x[0])
        text_in_a_dataframe[['label', 'score']] = text_in_a_dataframe['result'].apply(pd.Series)
        text_in_a_dataframe['result'].apply(pd.Series)
        text_in_a_dataframe.drop('result', axis=1, inplace=True)
        # text_in_a_dataframe.rename(columns={'label':'sentiment'},inplace=True)
        # text_in_a_dataframe.rename(columns={'score': 'polarity'},inplace=True)

        result = pd.DataFrame()
        result['sentiment'] = text_in_a_dataframe['label']
        result['polarity'] = text_in_a_dataframe['score']
        print(f'Text Classification Analysis results are {result}')
        # if return_all:
        del text_in_a_dataframe
        del savani_classification
        gc.collect()

        return result

    # Text Classification
    def perform_text_classification_using_old_bhadresh_savani(self, text, return_all):
        if isinstance(text, list):
            text = ' '.join(text)
        # model_name = "bhadresh-savani/bert-base-uncased-emotion"
        # savani_classification = pipeline("text-classification", model=model_name, return_all_scores=return_all)
        if self.all_modules and 'savani' in self.all_modules.all_modules.keys():
            print('Found savani_classification in LoadModules.all_modules')
            savani_classification = self.all_modules['savani']
        else:
            print('Not found savani_classification LoadModules.all_modules, loading')
            savani_classification = self.load_Modules.load_model_bhadresh_savani()

        results = savani_classification(text)
        del savani_classification
        gc.collect()
        print(f'Text Classification Analysis results are {results}')
        if return_all:
            return results[0]
        else:
            if results[0]:
                # print(results[0])
                sentiment_label = results[0]['label']
                sentiment_score = results[0]['score']
                return sentiment_label, sentiment_score
            else:
                return 'bad_data', 'Not Enough or Bad Data'

    def perform_sentiment_analysis_using_sam_lowe(self, text, return_all):
        if isinstance(text, list):
            text = ' '.join(text)
        # current_model = st.session_state['current_model']
        sam_lowe_classification = None
        if self.all_modules and 'samLowe' in self.all_modules.keys():
            print('Found sam_lowe_classification in global')
            sam_lowe_classification = self.all_modules['samLowe']
        else:
            print('Not found sam_lowe_classification global, loading...')
            sam_lowe_classification = self.load_Modules.load_model_sam_lowe(False)

        results = sam_lowe_classification(text)
        del sam_lowe_classification
        gc.collect()
        print(f'Sentimental Analysis  (SamLowe/roberta-base-go_emotions) results are {results}')
        if return_all:
            return results[0]
        else:
            sentiment_label = results[0]['label']
            sentiment_score = results[0]['score']
            return sentiment_label, sentiment_score

    def transcribe_audio_file(self, audio_file):
        with sr.WavFile(audio_file) as source:
            r = sr.Recognizer()
            audio = r.record(source)
            try:
                transcribed_text1 = r.recognize_google(audio, language='en-US')
                print("un-punctuated transcribed text:{}".format(transcribed_text1))
                if LoadModules.all_modules and 'punctuation' in LoadModules.all_modules.keys():
                    print('Found punctuation model')
                    punctuation_model = LoadModules.all_modules['punctuation']
                else:
                    print('Not found punctuation model , loading...')
                    punctuation_model = self.load_Modules.load_punctuation_model()

                transcribed_text2 = punctuation_model.restore_punctuation(transcribed_text1)
                # punctuatorModel = Punctuator('model.pcl')
                # transcribed_text2 = punctuatorModel.punctuate(transcribed_text1)
                print("Punctuated transcribed text:{}".format(transcribed_text2))
                sentences = sent_tokenize(transcribed_text2)
                return sentences
            except sr.UnknownValueError:
                print("Google could not understand audio")
            except sr.RequestError as e:
                print("Google/PunctionalModel error; {0}".format(e))
            finally:
                del punctuation_model
                gc.collect()

    def transcribe_audio_data(self, audio_data):
        try:
            r = sr.Recognizer()
            transcribed_text1 = r.recognize_google(audio_data, language='en-US')
            print("un-punctuated transcribed text:{}".format(transcribed_text1))
            punctuation_model = LoadModules.all_modules['punctuation']
            transcribed_text2 = punctuation_model.restore_punctuation(transcribed_text1)
            print("Punctuated transcribed text:{}".format(transcribed_text2))
            sentences = sent_tokenize(transcribed_text2)
        except sr.UnknownValueError:
            print("Google could not understand audio")
        except sr.RequestError as e:
            print("Google/PunctionalModel error; {0}".format(e))
        finally:
            del punctuation_model
            gc.collect()

        return sentences

    # def transcribe_audio_with_punctuation(self, audio_file):
    #     client = speech.SpeechClient()
    #     transcribed_text = []
    #     with open(audio_file, "rb") as audio_file:
    #         content = audio_file.read()

    #     audio = speech.RecognitionAudio(content=content)
    #     config = speech.RecognitionConfig(
    #         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    #         language_code="en-US",
    #         enable_automatic_punctuation=True,  # Enable automatic punctuation
    #     )

    #     response = client.recognize(config=config, audio=audio)

    #     for result in response.results:
    #         transcribed_text.append(result.alternatives[0].transcript)
    #         print("Transcript: {}".format(result.alternatives[0].transcript))

    #     transcribed_text = " ".join(transcribed_text)
    #     sentences = sent_tokenize(transcribed_text)
    #     return sentences

    def text_blob_sentiments(self, text):
        # Create a TextBlob object
        try:
            output = TextBlob(text)
            print('text_blob_sentiments output:')
            print(output)
            if output:
                polarity = output.sentiment.polarity
                subjectivity = output.subjectivity
                if -0.001 > polarity and subjectivity > 0:
                    return 'NEGATIVE', polarity
                elif -0.001 <= polarity <= 0.001:
                    return 'NEUTRAL', polarity
                else:
                    return 'POSITIVE', polarity
        except Exception as ex:
            print("Error occurred during .. text_blob_sentiments")
            print(str(ex))
            return "error", str(ex)

    # def text_blob_sentimentsOnly(self, text):
    #     # Create a TextBlob object
    #     try:
    #         output = TextBlob(text)
    #         if output:
    #             polarity = output.sentiment.polarity
    #             subjectivity = output.subjectivity
    #             # print(text, polarity, subjectivity)
    #             if -0.02 > polarity and subjectivity > 0:
    #                 return 'NEGATIVE'
    #             elif -0.2 < polarity < 0.2:
    #                 return 'NEUTRAL'
    #             elif polarity > 0.02:
    #                 return 'POSITIVE'
    #     except Exception as ex:
    #         print("Error occurred during .. text_blob_sentiments")
    #         print(str(ex))
    #         return "error", str(ex)
    #
    # def text_blob_polarityOnly(self, text):
    #     # Create a TextBlob object
    #     try:
    #         output = TextBlob(text)
    #         if output:
    #             polarity = output.sentiment.polarity
    #             # subjectivity = output.subjectivity
    #             # print(polarity, subjectivity)
    #             # if -0.02 > polarity and subjectivity > 0:
    #             #     return 'NEGATIVE'
    #             # elif -0.2 < polarity < 0.2:
    #             #     return 'NEUTRAL'
    #             # elif polarity > 0.02:
    #             #     return 'POSITIVE'
    #             return polarity
    #     except Exception as ex:
    #         print("Error occurred during .. text_blob_sentiments")
    #         print(str(ex))
    #         return "error", str(ex)

    # def analyze_sentiment(self, text):
    #     sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")
    #     results = sentiment_analysis(text)
    #     sentiment_results = {result['label']: result['score'] for result in results}
    #     return sentiment_results

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
    def perform_sentiment_analysis_using_vader(self, paragraph):
        if type(paragraph) != list:
            paragraph = sent_tokenize(paragraph)

        vader_obj = None
        if LoadModules.all_modules and 'vader' in LoadModules.all_modules.keys():
            print('Found vader_obj in LoadModules.all_modules')
            vader_obj = LoadModules.all_modules['vader']
        else:
            print('Not found vader_obj global, loading...')
            vader_obj = self.load_Modules.load_model_vader()

        # polarity_scores method of SentimentIntensityAnalyzer
        # object gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiments_dict = [vader_obj.polarity_scores(sentence)['compound'] for sentence in paragraph]
        print("Vader number of sentences:"+str(len(sentiments_dict)))
        # Calculate overall sentiment for the entire paragraph
        overall_sentiment = sum(sentiments_dict) / len(sentiments_dict)

        # sentiment_dict = vader_obj.polarity_scores(paragraph)
        print(f'Sentiment Analysis (Vader) results are {overall_sentiment}')
        del vader_obj
        gc.collect()
        if overall_sentiment:
            # decide sentiment as positive, negative and neutral
            if overall_sentiment >= 0.001:
                sentiment_label = 'Positive'
                sentiment_score = overall_sentiment
            elif overall_sentiment <= - 0.001:
                sentiment_label = 'Negative'
                sentiment_score = overall_sentiment
            else:
                sentiment_label = 'Neutral'
                sentiment_score = overall_sentiment
            return sentiment_label, sentiment_score
        else:
            return 'bad_data', 'Bad Data or Insufficient Data'

    def convertTextToSentences(self, text):
        # Tokenize the paragraph into sentences
        sentences = [Sentence(sent, use_tokenizer=True) for sent in text]
        return sentences

# if __name__ == "__main__":
#     try:
#         voiceAnalysisServices = VoiceAnalysisServices()
#         transcribed_text = voiceAnalysisServices.transcribe_audio_file('./voices/call_center_part-2b_sc.wav')
#         print( voiceAnalysisServices.perform_sentiment_analysis_using_sam_lowe(transcribed_text, True))
#         # voiceAnalysisServices.perform_text_classification_using_bhadresh_savani(transcribed_text, True)
#         # voiceAnalysisServices.perform_sentiment_analysis_using_textblob(transcribed_text)
#         # print(voiceAnalysisServices.perform_sentiment_analysis_using_distilbert(transcribed_text, True))
#         # voiceAnalysisServices.perform_sentiment_analysis_using_vader(transcribed_text)
#         # voiceAnalysisServices.perform_sentiment_analysis_using_flair(transcribed_text, True)
#         # main()
#     except Exception as ex:
#         st.error("Error occurred during sentiment/textual analysis.")
#         st.error(str(ex))
#         traceback.print_exc()
