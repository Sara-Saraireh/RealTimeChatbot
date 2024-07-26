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
    'apikey': '1Ko0EUaMVmso8T4FEt1vYLhz-KSIdlztNTb2a-ykd7fR',
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
    project_id='1e2141f0-2107-4e64-97a2-c58e71976116'
)

# This function loads a PDF of your choosing
@st.cache_resource
def load_pdf(pdf_file):
    # Save uploaded file temporarily
    with open(pdf_file.name, "wb") as f:
        f.write(pdf_file.getbuffer())
    loaders = [PyPDFLoader(pdf_file.name)]
    # Create index - aka vector database - aka chromadb
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)
    # Return the vector database
    return index

# Setup the app title
st.title('Ask watsonx')

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Load vector database
    index = load_pdf(uploaded_file)

    # Create a Q&A chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=index.vectorstore.as_retriever(),
        input_key='question'
    )

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
else:
    st.write("Please upload a PDF file to proceed.")
