import os
import traceback

import audio_recorder_streamlit
import streamlit as st
import audio_recorder_streamlit as ars
import numpy as np

col1, col2, col3= st.columns([1,4,1])
col2.markdown(" # My First App with Streamlit")
col2.markdown("A Faster way to build and share data apps")

col2.markdown("---")

uploaded_audio = col2.file_uploader("*Upload an audio file to analyse*", type=[".wav", ".mp3"], accept_multiple_files=False)
if uploaded_audio is not None:
     audio_bytes = uploaded_audio.read()
if uploaded_audio is not None:
    if uploaded_audio.type=="mp3":
        col2.audio(audio_bytes, format='audio/mp3')
    else:
        col2.audio(audio_bytes, format='audio/wav')

col2.markdown("---")


col2.markdown('*Audio Recorder*')
recorded_audio_in_bytes = ars.audio_recorder(pause_threshold=3.0, sample_rate=41_000)

if recorded_audio_in_bytes is not None:
    if len(recorded_audio_in_bytes) > 0:
        # To play audio in frontend:
        st.audio(recorded_audio_in_bytes, start_time=0)
        # sample_rate = 44100  # 44100 samples per second
        # seconds = 2  # Note duration of 2 seconds
        # frequency_la = 440  # Our played note will be 440 Hz
        #  # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
        # t = np.linspace(0, seconds, seconds * sample_rate, False)
        #  # Generate a 440 Hz sine wave
        # note_la = np.sin(frequency_la * t * 2 * np.pi)
        # st.audio(note_la, sample_rate=sample_rate)
        # To save audio to a file:
        # wav_file = open("audio.mp3", "wb")
        # wav_file.write(recorded_audio_in_bytes)
