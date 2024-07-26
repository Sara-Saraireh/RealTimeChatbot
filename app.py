import streamlit as st
from transformers import pipeline
import spacy

st.write("Initializing spacy...")
try:
    nlp = spacy.load('en_core_web_sm')
    st.write("Spacy model loaded successfully.")
except OSError:
    st.write("Downloading spacy model...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
    st.write("Spacy model downloaded and loaded.")

st.write("Initializing chatbot pipeline...")
chatbot_pipeline = pipeline('conversational')
st.write("Chatbot pipeline initialized.")

def generate_response(user_input):
    st.write(f"Generating response for input: {user_input}")
    response = chatbot_pipeline(user_input)
    st.write(f"Response: {response[0]['generated_text']}")
    return response[0]['generated_text']

st.title("Real-time Chatbot Agent")
st.write("Interact with the chatbot below:")

if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Bot: {response}")

for message in st.session_state.history:
    st.write(message)
