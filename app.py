import streamlit as st
from huggingface_hub import InferenceClient

# Initialize the InferenceClient
@st.cache_resource
def get_inference_client():
    return InferenceClient(
        "microsoft/DialoGPT-medium",
        token=st.secrets["HF_API_TOKEN"]
    )

# Function to generate response
def generate_response(client, messages):
    response = ""
    for message in client.chat_completion(
        messages=messages,
        max_tokens=500,
        stream=True,
    ):
        response += message.choices[0].delta.content or ""
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
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        # Add user input to messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate response
        bot_response = generate_response(client, st.session_state.messages)
        
        # Add bot response to messages
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Clear user input
        st.session_state.user_input = ""

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")
