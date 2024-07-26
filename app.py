import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

# Streamlit app
st.title("Advanced AI Chatbot")
st.write("Hello! I'm an AI chatbot powered by DialoGPT. Let's chat!")

# Initialize model and tokenizer
tokenizer, model = load_model()

# Initialize chat history
if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = None

if 'messages' not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        # Add user input to messages
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Tokenize the new user input
        new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        
        # Append the new user input tokens to the chat history
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_user_input_ids
        
        # Generate a response
        st.session_state.chat_history_ids = model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )
        
        # Decode the response
        response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Add bot response to messages
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    else:
        st.write(f"**Bot:** {message['content']}")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history_ids = None
    st.session_state.messages = []
