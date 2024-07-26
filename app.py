import streamlit as st
import random
import re

# Define some patterns and responses
patterns = [
    (r'\b(hi|hello|hey)\b', ['Hello!', 'Hi there!', 'Hey! How can I help you?']),
    (r'how are you', ['I'm doing well, thanks for asking!', 'I'm great! How about you?']),
    (r'your name', ['My name is ChatBot. Nice to meet you!', "I'm ChatBot, your friendly AI assistant."]),
    (r'(bye|goodbye)', ['Goodbye!', 'See you later!', 'Have a great day!']),
    (r'thank you', ['You're welcome!', 'Glad I could help!', 'My pleasure!']),
    (r'weather', ['I'm sorry, I don't have real-time weather information. You might want to check a weather website or app for that.']),
    (r'favorite (color|colour)', ['As an AI, I don't have personal preferences, but I find all colors fascinating!']),
    (r'tell me a joke', ['Why don't scientists trust atoms? Because they make up everything!', 'What do you call a fake noodle? An impasta!']),
    (r'what can you do', ['I can engage in conversation, answer questions, and help with various tasks. Feel free to ask me anything!']),
]

def get_response(user_input):
    user_input = user_input.lower()
    for pattern, responses in patterns:
        if re.search(pattern, user_input):
            return random.choice(responses)
    return "I'm not sure how to respond to that. Can you try rephrasing or asking something else?"

st.title("Rule-based Chatbot")
st.write("Hello! I'm a simple rule-based chatbot. Feel free to start a conversation!")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        # Generate response
        response = get_response(user_input)
        
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
