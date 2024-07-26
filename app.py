# Import langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Bring in streamlit for UI dev
import streamlit as st

# Bring in watsonx interface
from watsonxlangchain import LangChainInterface

creds = {
    'apikey': 'ZNPRw05jup0f2SFOCFkAtfZskeCPL6fESGaYAq1EWRp7',
    'url': 'https://us-south.ml.cloud.ibm.com'
}

# Create LLM using Langchain
llm = LangChainInterface(
    credentials=creds,
    model='meta-llama/llama-2-70b-chat',
    params={
        'decoding_method': 'sample',
        'max_new_tokens': 200,
        'temperature': 0.5
    },
    project_id='dd58941e-28cc-4abe-9189-0bebc2f2edec'
)

# This function loads a PDF of your choosing
@st.cache_resource
def load_pdf():
    # Update PDF name here to whatever you like
    pdf_name = 'what is generative ai.pdf'
    loaders = [PyPDFLoader(pdf_name)]
    # Create index - aka vector database - aka chromadb
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    # Return the vector database
    return index

# Load vector database
index = load_pdf()

# Create a Q&A chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever()
)

# Setup the app title
st.title('Ask watsonx')

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt here')

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # Send the prompt to the PDF Q&A CHAIN
    response = chain.run(prompt)
    # Show the LLM response
    st.chat_message('assistant').markdown(response)
    # Store the LLM response in state
    st.session_state.messages.append({'role': 'assistant', 'content': response})
