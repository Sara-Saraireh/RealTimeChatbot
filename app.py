import streamlit as st
import os
from huggingface_hub import InferenceClient

# Get the API token from environment variable or Streamlit secrets
API_TOKEN = "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"

if not API_TOKEN:
    st.error("Please set the HF_API_TOKEN in your environment or Streamlit secrets.")
    st.stop()

# Initialize the InferenceClient
@st.cache_resource
def get_inference_client():
    return InferenceClient(
        "microsoft/DialoGPT-medium",
        token=API_TOKEN
    )

# Function to generate response
def generate_response(client, messages):
    response = ""
    try:
        for message in client.chat_completion(
            messages=messages,
            max_tokens=500,
            stream=True,
        ):
            response += message.choices[0].delta.content or ""
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        response = "I'm sorry, I encountered an error. Please try again."
    return response

# Streamlit app
st.title("Advanced AI Chatbot")
st.write("Hello! I'm an AI chatbot powered by DialoGPT. Let's chat!")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize InferenceClient
client = get_inference_client()

# User input
user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        # Add user input to messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate response
        bot_response = generate_response(client, st.session_state.messages)
        
        # Add bot response to messages
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.messages = []
