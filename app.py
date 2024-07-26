import streamlit as st
from transformers import pipeline
import spacy

# Ensure you have downloaded the necessary spacy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Load the conversational pipeline from Hugging Face
chatbot_pipeline = pipeline('conversational')

# Function to generate a response
def generate_response(user_input):
    response = chatbot_pipeline(user_input)
    return response[0]['generated_text']

# Streamlit App
st.title("Real-time Chatbot Agent")
st.write("Interact with the chatbot below:")

# Chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response = generate_response(user_input)
        st.session_state.history.append(f"You: {user_input}")
        st.session_state.history.append(f"Bot: {response}")

# Display chat history
for message in st.session_state.history:
    st.write(message)
