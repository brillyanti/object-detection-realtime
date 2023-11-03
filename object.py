import streamlit as st
from gtts import gTTS

# Streamlit app layout
st.title('Text-to-Speech with Streamlit and gTTS')

# Input teks dari pengguna
text = st.text_input('hallo apakabar?')

# Jika ada teks yang dimasukkan, konversi menjadi suara
if text:
    tts = gTTS(text)
    st.audio(tts.save("output.mp3"), format="audio/mp3")

