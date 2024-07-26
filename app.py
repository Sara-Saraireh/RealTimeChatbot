import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

def generate_response(input_text, chat_history_ids):
    tokenizer, model = load_model()
    
    # Encode the new user input, add parameters and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )
    
    # Decode and return the model's response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

st.title("DialoGPT Chatbot")
st.write("Hello! I'm a chatbot powered by DialoGPT. Feel free to start a conversation!")

# Initialize chat history
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        # Generate response
        response, st.session_state.chat_history_ids = generate_response(user_input, st.session_state.chat_history_ids)
        
        # Update chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

        # Clear user input
        st.session_state.user_input = ""

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "You":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")
