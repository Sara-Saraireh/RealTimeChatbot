import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Hugging Face API token from Streamlit secrets
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]

# Load the DialoGPT model and tokenizer using the API token
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", use_auth_token=HF_API_TOKEN)
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", use_auth_token=HF_API_TOKEN)

# Function to generate a response
def generate_response(user_input, chat_history_ids=None):
    # Encode the new user input and add the EOS token
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the last output tokens from the bot
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response, chat_history_ids

# Streamlit App
st.title("Real-time Chatbot Agent")
st.write("Interact with the chatbot below:")

# Initialize chat history
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

# User input
user_input = st.text_input("You:", "")

if st.button("Send"):
    if user_input:
        response, chat_history_ids = generate_response(user_input, st.session_state.chat_history_ids)
        st.session_state.chat_history_ids = chat_history_ids
        
        st.write(f"You: {user_input}")
        st.write(f"Bot: {response}")
