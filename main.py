import traceback
import streamlit as st
from voiceAnalysisServices import perform_sentiment_analysis_using_distilbert, transcribe_audio_file, perform_text_classification_using_bhadresh_savani, transcribe_audio_data
from myUtilityDefs import convertToNewDictionary, print_sentiments, get_sentiment_emoji
import pandas as pd
import numpy as np
from os import path
import audio_recorder_streamlit as ars
import time


st.set_page_config(layout="wide")
# col1, col2 = st.columns([1,1])
#-----
print("Setting session state")
actionRadioButtonState = st.session_state.get("enable_radio", {"value": True})
uploadButtonState = st.session_state.get("enable_upload", {"value": True})

# st.session_state["action_radio"] = "visible"
# st.session_state["action_radio"].disabled = False
# st.session_state['upload'].disabled = False

st.markdown(' # AI to Detect Sentiment in the Audio!')
st.markdown('*Pankaj Kumar Chopra*')
# st.image('work-in-progress.png', width=100)


#  app.py
st.sidebar.title("Audio Analysis")
st.sidebar.write("""The Audio Analysis app is a powerful tool that allows you to analyze audio files 
                 and gain valuable insights from them. It combines speech recognition 
                 and sentiment analysis techniques to transcribe the audio 
                 and determine the sentiment expressed within it.""")
action_names = ['Sample Audio', 'Upload an Audio','Live Audio']
action = st.sidebar.radio('Actions',
                          action_names,
                          key='action_radio',
                          disabled= not actionRadioButtonState
                          )

if action=='Sample Audio':
    process_sample1_button = st.sidebar.button("Process Sample 1", key=1 )
    process_sample2_button = st.sidebar.button("Process Sample 2", key=2)
    process_sample3_button = st.sidebar.button("Process Sample 3", key=3)
elif action=='Upload an Audio':
    st.markdown("*Upload an audio file(format=wav)*")
elif action=='Live Audio':
    st.sidebar.markdown('*Audio Recorder*')
    with st.sidebar:
        recorded_audio_in_bytes = ars.audio_recorder(text="Click to Record", pause_threshold=2.0, sample_rate=41_000)


def write_current_status(status_area, text):
    with status_area:
        st.empty()
        st.markdown('**'+text+'**');


def process_and_show_semantic_analysis_results(audio_file, transcribed, transcribed_text):
    if not transcribed:
        st.write(f'Processing {audio_file}...' )
        transcribed_text = transcribe_audio_file(audio_file)
        # st.header("Transcribed Text")
        st.text_area("Transcribed Text", transcribed_text, key=1, height=150)
    st.header("Transcribed Text")
    st.text_area("", transcribed_text, height=150)

    # st.markdown(" # Analysing...")
    return_all = False
    sentiment_label, sentiment_score = perform_sentiment_analysis_using_distilbert(transcribed_text, return_all)
    if(sentiment_label=='error'):
        traceback.print_exc()
    else:
        st.header("Sentiment Analysis")
        st.markdown("*" + print_sentiments(sentiment_label, sentiment_score) + "*")


def process_and_show_text_classification_results(audio_file, transcribed, transcribed_text):
    if not transcribed:
        st.write(f'Processing {audio_file}...' )
        transcribed_text = transcribe_audio_file(audio_file)
        # st.header("Transcribed Text")
        st.text_area("Transcribed Text", transcribed_text, key=2, height=200)

    # st.markdown(" # Text Classification...")
    st.header("Text Classification Results Analysis")
    return_all=False
    if return_all:
        resultDict = perform_text_classification_using_bhadresh_savani(transcribed_text, return_all)
        rsultDictionary=convertToNewDictionary(resultDict)
        # print(f' Keys:{list(rsultDictionary.keys())}')
        # print(f' Values {list(rsultDictionary.values())}')
        sentimental_results = []
        for key, value in rsultDictionary.items():
            sentimental_results.append(f'{key}({value}) ')

        st.markdown("*"+''.join(sentimental_results)+ "*")
    else:
        sentiment_label, sentiment_score = perform_text_classification_using_bhadresh_savani(transcribed_text, return_all)
        st.markdown("*"+print_sentiments(sentiment_label, sentiment_score)+ "*")
    # yy = list(rsultDictionary.values())
    # xx = list(rsultDictionary.keys())
    # print(xx, yy)
    # data = pd.DataFrame(yy, columns = xx)
    # st.line_chart(data)


def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score}\n"
    return sentiment_text


def doActualthings(main_status, status_area,audio_file):
    write_current_status(main_status, f'   *Selected Sample File: {audio_file}*')
    write_current_status(status_area, f'Transcribing audio of  {audio_file}...')
    transcribed_text = transcribe_audio_file(audio_file)
    write_current_status(status_area, f'Semantic Analysis of {audio_file}...')
    process_and_show_semantic_analysis_results(None, True, transcribed_text)
    write_current_status(status_area, f'Text Classification of {audio_file}...')
    process_and_show_text_classification_results(audio_file, True, transcribed_text)
    # write_current_status(main_status, '')
    write_current_status(status_area, 'Finished Processing!! ')

def main():
    main_status = st.markdown('')
    status_area = st.markdown('')
    if action=='Sample Audio':
        audio_file1 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0061_8k.wav")
        audio_file2 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0040_8k.wav")
        audio_file3 = path.join(path.dirname(path.realpath(__file__)), "voices/OSR_us_000_0019_8k.wav")
        try:
            # process_sample1_button = st.sidebar.button("Process Sample 1", key=1)
            # process_sample2_button = st.sidebar.button("Process Sample 2", key=2)
            # process_sample3_button = st.sidebar.button("Process Sample 3", key=3)
            if process_sample1_button:
                # actionRadioButtonState["value"] = False
                # st.session_state.actionRadioButtonState = actionRadioButtonState
                doActualthings(main_status, status_area, audio_file1)
            elif process_sample2_button:
                # actionRadioButtonState["value"] = False
                # st.session_state.actionRadioButtonState = actionRadioButtonState
                doActualthings(main_status, status_area,audio_file2)
            elif process_sample3_button:
                # actionRadioButtonState["value"] = False
                # st.session_state.actionRadioButtonState = actionRadioButtonState
                doActualthings(main_status, status_area, audio_file3)
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
                uploadButtonState["value"] = False
                st.session_state.uploadButtonState = uploadButtonState
                st.write(f'Processing {audio_file}...')
                write_current_status(status_area, f'Transcribing audio of  {audio_file}...')
                transcribed_text = transcribe_audio_file(audio_file)
                write_current_status(status_area, f'Semantic Analysis of {audio_file}...')
                process_and_show_semantic_analysis_results(None, True, transcribed_text)
                write_current_status(status_area, f'Text Classification of {audio_file}...')
                process_and_show_text_classification_results(audio_file, True, transcribed_text)
                write_current_status(status_area, f'Done!! Here are the results of {audio_file}...')
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
                    # wav_file = open("recorded.mp3", "r")
                    # wav_file.close()
                    # st.empty()
                    write_current_status(status_area, "Processing your recording...")
                    # transcribed_text = transcribe_audio_data(recorded_audio_in_bytes)
                    write_current_status(status_area, "Transcribing audio...")
                    transcribed_text = transcribe_audio_file('recorded.mp3')
                    write_current_status(status_area, "Semantic Analysis...")
                    process_and_show_semantic_analysis_results(None, True, transcribed_text)
                    write_current_status(status_area, "Text Classification...")
                    process_and_show_text_classification_results(None, True, transcribed_text)
                    write_current_status(status_area,"Done!! Here are the results")
        except Exception as ex:
            st.error("Error occurred during audio transcription and sentiment analysis.")
            st.error(str(ex))
            traceback.print_exc()
        finally:
            uploadButtonState["value"] = True
            st.session_state.uploadButtonState = uploadButtonState


if __name__ == "__main__":
    main()
