import time
import streamlit as st

def method1():
    time.sleep(1)
    # Method 1 logic

def method2():
    time.sleep(1)
    # Method 2 logic

def method3():
    time.sleep(1)
    # Method 3 logic

def method4():
    time.sleep(1)
    # Method 4 logic

def method5():
    time.sleep(1)
    # Method 5 logic

# Initialize the progress bar
progress_bar = st.progress(0)

# Method 1
method1()
progress_bar.progress(20)

# Method 2
method2()
progress_bar.progress(40)

# Method 3
method3()
progress_bar.progress(60)

# Method 4
method4()
progress_bar.progress(80)

# Method 5
method5()
progress_bar.progress(100)
