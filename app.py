import streamlit as st
import os
from huggingface_hub import InferenceClient

# Get the API token from environment variable or Streamlit secrets
API_TOKEN = os.environ.get("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN")

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
    try:
        response = client.text_generation(
            prompt="\n".join([f"{m['role']}: {m['content']}" for m in messages]),
            max_new_tokens=50,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True
        )
        return response[0]['generated_text'].split("\n")[-1].strip()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "I'm sorry, I encountered an error. Please try again."

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
