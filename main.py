import time
import traceback
import streamlit as st
from voiceAnalysisServices import transcribe_audio_file, perform_text_classification_using_bhadresh_savani, perform_sentiment_analysis, text_blob_sentiments
from myUtilityDefs import convertToNewDictionary, print_sentiments, get_sentiment_emoji
from os import path
import audio_recorder_streamlit as ars

import nltk
nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
nltk.download('vader_lexicon')
# nltk.download('all-corpora')

st.set_page_config(layout="wide")
# col1, col2 = st.columns([1,1])
#-----
print("Setting session state")
actionRadioButtonState = st.session_state.get("enable_radio", {"value": True})
uploadButtonState = st.session_state.get("enable_upload", {"value": True})

# st.session_state["action_radio"] = "visible"
# st.session_state["action_radio"].disabled = False
# st.session_state['upload'].disabled = False

st.markdown(' # FA/Client Sentiment Analysis!')
# footer = st.footer('Author: *Pankaj Kumar Chopra*')
# st.markdown(
#     """
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: visible;}
#     </style>
#     """,
#     unsafe_allow_html=True
# )
# st.image('work-in-progress.png', width=100)

with st.sidebar:
    #  app.py
    st.title("Audio Analysis")
    st.write("""The Audio Analysis app is a powerful tool that allows you to analyze audio files 
                     and gain valuable insights from them. It combines speech recognition 
                     and sentiment analysis techniques to transcribe the audio 
                     and determine the sentiment expressed within it.""")
    # st.write(
    #     '''A brief description of Sematic Analysis and models. Check a presentation
    #     [link](https://docs.google.com/presentation/d/e/2PACX-1vTzSLasf4BF4oeAOi66N0fXYzICBlJA3_PyLZAOjqNhJ8GuTm5V2l5EJlknS7Xn2Z7PNkTYa1zNpPMz/pub?start=false&loop=false&delayms=3000)''')
    # pptx = path.join(path.dirname(path.realpath(__file__)), "Sentiments_Analysis.pptx")
    # with open(pptx, "rb") as file:
    #     st.download_button("Download",data=file, file_name='Sentiments_Analysis.pptx', mime='application/msword')
    # colm1, colm2 = st.columns([1,2])
    preds = {
        'TextBlob(PatternAnlyzer) Based Sentiment Analysis': "textblob",
        # 'TextBlob(NaiveBayesAnlyzer) Based Sentiment Analysis': "textblob",
        'SamLowe/roberta-base-go_emotions': 'roberta',
        'distilbert-base-uncased-finetuned': 'distilbert',
        'VADER Based Sentiment Analysis': 'vader',
        'FLAIR Based Sentiment Analysis': 'flair',
        "Use All and Compare": 'All'
        # ,'Whisper - MultiLingual(Audio)': "whisper"
    }
    model_select = st.selectbox(
        "Select the model to predict : ", list(preds.keys()))
    model_predict = preds.get(model_select)
    action_names = ['Sample Audio', 'Upload an Audio','Live Audio', 'Plain Text']
    action = st.radio('Actions',
                              action_names,
                              key='action_radio',
                              disabled= not actionRadioButtonState
                              )

    if action=='Sample Audio':
        process_sample1_button = st.button("Sample 1", key=1 )
        process_sample2_button = st.button("Call Center Sample", key=2)
        process_sample3_button = st.button("Sample 3", key=3)
    elif action == 'Upload an Audio':
        col1, col2 = st.columns([1,2])
        col1.markdown("**Upload an audio file (format = wav only) **")
        col2.markdown("*Do not upload music wav file it will give error(s).*")

    elif action=='Live Audio':
        st.markdown('*Audio Recorder*')
        recorded_audio_in_bytes = ars.audio_recorder(text="Click to Record ( 2 sec pause starts analysis)", pause_threshold=2.0, sample_rate=41_000)
    elif action=='Plain Text':
        st.markdown('*Plain Text*')
        text = st.text_area('Type or paste few sentences to Analyse(>10 char)', key=9, height=100)
        analyse = st.button('Analyse')


def write_current_status(status_area, text):
    with status_area:
        st.empty()
        st.markdown('**'+text+'**')


def process_and_show_semantic_analysis_results(audio_file, transcribed, transcribed_text, model):
    if not transcribed and audio_file:
        st.write(f'Processing {audio_file}...' )
        transcribed_text = transcribe_audio_file(audio_file)
        # st.header("Transcribed Text")
        st.text_area("Transcribed Text", transcribed_text, key=1, height=150)
    st.header("Transcribed Text")
    st.text_area("", transcribed_text, height=150)
    # st.markdown(" # Analysing...")
    return_all = False
    if model == 'All':
        st.write('''Sentiment analysis comparison for three NLP tools
                    Vader vs Flair vs TextBlob [Click Here](https://aashishmehta.com/sentiment-analysis-comparison/)''')
        result = perform_sentiment_analysis(transcribed_text, return_all, model)
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        # print(result)
        # st.write(result)
        mystyle = '''
            <style>
                p {
                    text-align: justify;
                }
            </style>
            '''
        st.markdown(mystyle, unsafe_allow_html=True)
        for k in list(result.keys()):
            sentiment_label = result.get(k)['sentiment_label']
            sentiment_score = result.get(k)['sentiment_score']
            if k == 'flair':
                col1.markdown("<font size='5' align='center'> Model: {} </font>".format(k.upper()), unsafe_allow_html=True)
                col1.markdown("<font size='4' > Score: {} </font>".format(sentiment_score), unsafe_allow_html=True)
                col1.subheader(f"{get_sentiment_emoji(sentiment_label.lower())} {sentiment_label}")
                col1.info("""Flair, employs pre-trained language models and transfer 
                learning to generate contextual string embeddings 
                for sentiment analysis""")
            if k == 'vader':
                col2.markdown("<font size='5' align='center'> Model: {} </font>".format(k.upper()), unsafe_allow_html=True)
                col2.markdown("<font size='4' > Score: {} </font>".format(sentiment_score), unsafe_allow_html=True)
                col2.subheader(f"{get_sentiment_emoji(sentiment_label.lower())} {sentiment_label}")
                col2.info("""VADER (Valence Aware Dictionary 
                           and sEntiment Reasoner) is a rule-based model that uses a 
                           sentiment lexicon and grammatical rules to determine 
                           the sentiment scores of the text. """)
            if k == 'textblob':
                col3.markdown("<font size='5' align='center'> Model: {} </font>".format(k.upper()), unsafe_allow_html=True)
                col3.markdown("<font size='4' > Score: {} </font>".format(sentiment_score), unsafe_allow_html=True)
                col3.subheader(f"{get_sentiment_emoji(sentiment_label.lower())} {sentiment_label}")
                col3.info("""TextBlob(default PatternAnalyzer) is a Python NLP library that uses a natural language toolkit (NLTK).  
                                           aTextblob it gives two outputs, which are polarity and subjectivity. 
                                           Polarity is the output that lies between [-1,1], where -1 refers to negative 
                                           sentiment and +1 refers to positive sentiment. Subjectivity is the output that 
                                           lies within [0,1] and refers to personal opinions and judgments 
                                           sentiment lexicon and grammatical rules to determine 
                                           the sentiment scores of the text. More details here[PatternAnalysis](https://phdservices.org/pattern-analysis-in-machine-learning/) [Naive Bayes](https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/)""")
            if k == 'roberta':
                col4.markdown("<font size='5' align='center'> Model: {} </font>".format(k.upper()), unsafe_allow_html=True)
                col4.markdown("<font size='4' > Score: {} </font>".format(sentiment_score), unsafe_allow_html=True)
                col4.subheader(f"{get_sentiment_emoji(sentiment_label.lower())} {sentiment_label}")
                col4.info("""Twitter-roberta-base-sentiment is a roBERTa model trained 
                on ~58M tweets and fine-tuned for sentiment analysis. 
                Fine-tuning is the process of taking a pre-trained
                 large language model (e.g. roBERTa in this case) 
                 and then tweaking it with additional training 
                data to make it perform a second similar task (e.g. sentiment analysis""")
            if k == 'distilbert':
                col5.markdown("<font size='5' align='center'> Model: {} </font>".format(k.upper()), unsafe_allow_html=True)
                col5.markdown("<font size='4' > Score: {} </font>".format(sentiment_score), unsafe_allow_html=True)
                col5.subheader(f"{get_sentiment_emoji(sentiment_label.lower())} {sentiment_label}")
                col5.info("""DistilBERT is a smaller, faster and cheaper version of BERT. 
                It has 40% smaller than BERT 
                and runs 60% faster while preserving over 95% of BERT’s performance""")
    else:
        sentiment_label, sentiment_score = perform_sentiment_analysis(transcribed_text, return_all, model)
        if(sentiment_label == 'error'):
            traceback.print_exc()
        else:
            st.header(f"Sentiment Analysis  ")
            st.markdown("<font size='5'>  Model: {} </font>".format( model_select), unsafe_allow_html=True)
            st.markdown("*" + print_sentiments(sentiment_label, sentiment_score) + "*")


def process_and_show_text_classification_results(audio_file, transcribed, transcribed_text):
    if not transcribed:
        st.write(f'Processing {audio_file}...' )
        transcribed_text = transcribe_audio_file(audio_file)
        # st.header("Transcribed Text")
        st.text_area("Transcribed Text", transcribed_text, key=2, height=200)

    # st.markdown(" # Text Classification...")
    st.header("Text Classification Results Analysis(Bert-base-uncased-emotion)")
    return_all=False
    if return_all:
        resultDict = perform_text_classification_using_bhadresh_savani(transcribed_text, return_all)
        rsultDictionary=convertToNewDictionary(resultDict)
        sentimental_results = []
        for key, value in rsultDictionary.items():
            sentimental_results.append(f'{key}({value}) ')

        st.markdown("*"+''.join(sentimental_results)+ "*")
    else:
        sentiment_label, sentiment_score = perform_text_classification_using_bhadresh_savani(transcribed_text, return_all)
        st.markdown("*"+print_sentiments(sentiment_label, sentiment_score)+ "*")


def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score}\n"
    return sentiment_text


def doActualthings(status_area,audio_file, model):
    with st.spinner('Processing...'):
        progressBar = st.progress(5,"Processing...")
        time.sleep(0.2)
        progressBar.progress(15, 'Transcribing...')

        # write_current_status(main_status, f'   *Selected Sample File: {audio_file}*')
        write_current_status(status_area, f'Transcribing audio of  {audio_file}...')
        transcribed_text = transcribe_audio_file(audio_file)
        progressBar.progress(40, 'Semantic Analysis..')
        write_current_status(status_area, f'''Semantic Analysis using {model_predict} 
                                            File Name: {audio_file}...''')

        process_and_show_semantic_analysis_results(None, True, transcribed_text, model)
        if model_predict != 'All':
            progressBar.progress(70, 'Textual Classification..')
            write_current_status(status_area, f'Text Classification of {audio_file}...')
            process_and_show_text_classification_results(audio_file, True, transcribed_text)
        progressBar.progress(90, 'Textual Classification Done...')
        # write_current_status(status_area, 'Finished Processing!! ')
        progressBar.progress(100, 'Done, Finished Processing!!')


def main():
    # progressBar = st.progress(0)
    # st.session_state['progressBar'] = progressBar
    status_area = st.markdown('')
    if action=='Sample Audio':
        audio_file1 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0061_8k.wav")
        audio_file2 = path.join(path.dirname(path.realpath(__file__)), "voices/call_center.wav")
        audio_file3 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0019_8k.wav")
        try:
            if process_sample1_button:
                doActualthings( status_area, audio_file1, model_predict)
            elif process_sample2_button:
                # actionRadioButtonState["value"] = False
                # st.session_state.actionRadioButtonState = actionRadioButtonState
                doActualthings(status_area,audio_file2, model_predict)
            elif process_sample3_button:
                # actionRadioButtonState["value"] = False
                # st.session_state.actionRadioButtonState = actionRadioButtonState
                doActualthings(status_area, audio_file3, model_predict)
        except Exception as ex:
            st.error("Error occurred during audio transcription and sentiment analysis.")
            st.error(str(ex))
            traceback.print_exc()
        finally:
            actionRadioButtonState["value"] = True
            st.session_state.actionRadioButtonState = actionRadioButtonState
    if action == 'Upload an Audio':
        audio_file = st.sidebar.file_uploader("Browse", type=["wav"])
        upload_button = st.sidebar.button("Upload & Process", key="upload", disabled=not uploadButtonState)
        if audio_file and upload_button:
            try:
                # uploadButtonState["value"] = False
                # st.session_state.uploadButtonState = uploadButtonState
                doActualthings(status_area, audio_file, model_predict)
            except Exception as ex:
                st.error("Error occurred during audio transcription and sentiment analysis.")
                st.error(str(ex))
                traceback.print_exc()
            finally:
                uploadButtonState["value"] = True
                st.session_state.uploadButtonState = uploadButtonState
        # Perform audio tr
    if action == 'Live Audio':
        # st.sidebar.markdown('*Audio Recorder*')
        # recorded_audio_in_bytes = ars.audio_recorder(text="Click to Record", pause_threshold=3.0, sample_rate=41_000)
        try:
            if recorded_audio_in_bytes is not None:
                if len(recorded_audio_in_bytes) > 0:
                    # convert to a wav file
                    wav_file = open("recorded.mp3", "wb")
                    wav_file.truncate()
                    wav_file.write(recorded_audio_in_bytes)
                    wav_file.close()
                    doActualthings(status_area, "recorded.mp3", model_predict)
        except Exception as ex:
            st.error("Error occurred during audio transcription and sentiment analysis.")
            st.error(str(ex))
            traceback.print_exc()
        finally:
            uploadButtonState["value"] = True
            st.session_state.uploadButtonState = uploadButtonState
    if action == 'Plain Text':
        if analyse and len(text)>10:
            # st.header("Seman Classification Results Analysis(Bert-base-uncased-emotion)")
            process_and_show_semantic_analysis_results(None, True, text, model_predict)
            print(f'model_predict:{model_predict}')
            if model_predict != 'All':
                process_and_show_text_classification_results(None, True, text)
            # st.markdown("*" + print_sentiments(sentiment_label, sentiment_score) + "*")
            # st.header("TextBlob(rule-based approach) Analysis")
            # sentiment, word_count, sentiment_results = text_blob_sentiments(text)
            # st.text_area("Transcribed Text", text, key=5, height=150)
            # if word_count :
            #     st.text_area("Word Count", word_count, key=6, height=150)
            # if sentiment:
            #     st.markdown("**Textblob Sentiment**")
            #     st.markdown(sentiment)
            # if sentiment_results :
            #     st.markdown("**Sentiment  per Statement**")
            #     for centence, assessment in sentiment_results.items():
            #         st.write(centence)
            #         st.write(assessment)
            #         st.write('--- - - - - - - - - - - - - - - - - - - - - - - --------------------------------')


if __name__ == "__main__":
    main()
