
import os
import tempfile
import shutil
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import uuid
import atexit

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set page configuration
st.set_page_config(page_title="Multi-Mode Chatbot", page_icon="💬", layout="wide")

# Title
st.title("💬 Multi-Mode Chatbot")

# Global variable to track temporary directory
TEMP_DIR = tempfile.mkdtemp()

# Cleanup function for temporary directory
def cleanup_temp_dir():
    try:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    except Exception as e:
        print(f"Error cleaning up temporary directory: {e}")

# Register cleanup at exit
atexit.register(cleanup_temp_dir)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_vectors' not in st.session_state:
    st.session_state.pdf_vectors = {}

# Initialize LLMs
@st.cache_resource
def initialize_llms():
    return {
        'pdf': ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it"),
        'internet': ChatGroq(groq_api_key=groq_api_key, model_name="mistral-saba-24b")
    }

llms = initialize_llms()

# PDF Prompt template with comprehensive answer instructions
pdf_prompt = ChatPromptTemplate.from_template(
    """
    Provide a detailed and comprehensive answer based ONLY on the provided context. 
    Ensure you extract and present all relevant information from the documents.
    If the answer requires multiple paragraphs or involves details from different parts of the context, include them fully.
    
    If no clear answer can be found, state "I cannot find a comprehensive answer in the uploaded document."
    
    Context:
    {context}
    
    Question: {input}
    
    Answer with full detail, using all available relevant information from the context.
    """
)

# Internet Chat Prompt
internet_prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful, respectful, and honest assistant. 
    Always answer to the best of your ability while being clear and precise.
    
    Question: {input}
    """
)

# Define vector embedding function for uploaded PDFs
def process_uploaded_pdfs(uploaded_files):
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Dictionary to store vectors for each PDF
    pdf_vectors = {}

    # Load all uploaded PDFs
    for uploaded_file in uploaded_files:
        # Generate a unique filename in the temporary directory
        unique_filename = os.path.join(
            TEMP_DIR, 
            f"temp_{str(uuid.uuid4())}_{uploaded_file.name}"
        )
        
        # Save the file to the temporary directory
        with open(unique_filename, "wb") as f:
            f.write(uploaded_file.read())  # Save the file locally
        
        # Load and process PDF
        loader = PyPDFLoader(unique_filename)
        documents = loader.load()

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(documents)

        # Create vector store for this specific PDF
        vectors = FAISS.from_documents(final_documents, embeddings)
        
        # Store vectors with original filename for reference
        pdf_vectors[uploaded_file.name] = {
            'vectors': vectors,
            'temp_filename': unique_filename
        }

    return pdf_vectors

# Sidebar for Mode Selection and PDF Upload
st.sidebar.header("Chatbot Settings")
chat_mode = st.sidebar.radio("Select Chat Mode", 
    ["Internet Chat", "PDF Q&A"], 
    index=0
)

# PDF Upload for PDF Q&A Mode
if chat_mode == "PDF Q&A":
    st.sidebar.subheader("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    # Process PDFs if uploaded
    if uploaded_files:
        with st.sidebar.status("Processing documents..."):
            st.session_state.pdf_vectors = process_uploaded_pdfs(uploaded_files)
        st.sidebar.success(f"{len(uploaded_files)} documents processed!")

# Chat Interface
if chat_mode == "PDF Q&A" and st.session_state.pdf_vectors:
    # PDF Selection for Querying
    pdf_options = list(st.session_state.pdf_vectors.keys())
    selected_pdf = st.sidebar.selectbox(
        "Select PDF to query", 
        pdf_options, 
        index=0
    )

    # Render existing chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])
        with st.chat_message("assistant"):
            st.write(chat['answer'])

    # Chat input with enter key support
    prompt_input = st.chat_input("Ask a question about the selected document")

    # Process the question
    if prompt_input and selected_pdf:
        try:
            # Get the vector store for the selected PDF
            current_vectors = st.session_state.pdf_vectors[selected_pdf]['vectors']

            # Create document retrieval and QA chain with enhanced retrieval
            document_chain = create_stuff_documents_chain(
                llms['pdf'], 
                ChatPromptTemplate.from_template(
                    """
                    Based on the provided context, give a comprehensive and detailed answer to the question.
                    If the answer spans multiple documents/chunks, ensure to include all relevant information.
                    
                    Context:
                    {context}
                    
                    Question: {input}
                    
                    Answer with as much detail as possible from the available context.
                    """
                )
            )

            # Enhanced retriever with more document retrieval
            retriever = current_vectors.as_retriever(
                search_kwargs={
                    "k": 10,  # Increase number of retrieved documents
                    "fetch_k": 20  # Increase initial fetch to have more documents to choose from
                }
            )

            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Generate answer
            with st.chat_message("user"):
                st.write(f"[From {selected_pdf}] {prompt_input}")
            
            with st.chat_message("assistant"):
                with st.spinner("Generating comprehensive answer..."):
                    response = retrieval_chain.invoke({
                        'input': prompt_input
                    })

                    # Extract and display answer
                    if response and 'answer' in response:
                        answer = response['answer']
                        st.write(answer)

                        # Save to chat history
                        st.session_state.chat_history.append({
                            'question': f"[From {selected_pdf}] {prompt_input}",
                            'answer': answer
                        })
                    else:
                        st.error("Could not generate an answer.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif chat_mode == "Internet Chat":
    # Render existing chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat['question'])
        with st.chat_message("assistant"):
            st.write(chat['answer'])

    # Chat input with enter key support
    prompt_input = st.chat_input("Ask me anything...")

    # Process the question
    if prompt_input:
        try:
            # Generate answer using Internet Chat mode
            with st.chat_message("user"):
                st.write(prompt_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    # Create chat chain
                    chat_chain = llms['internet'].invoke([
                        {"role": "system", "content": internet_prompt.format(input=prompt_input)},
                        {"role": "user", "content": prompt_input}
                    ])

                    # Extract and display answer
                    answer = chat_chain.content
                    st.write(answer)

                    # Save to chat history
                    st.session_state.chat_history.append({
                        'question': prompt_input,
                        'answer': answer
                    })
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    # Instruction for uploading documents in PDF mode
    st.info("Please upload PDF documents in the sidebar to start chatting.")

# Sidebar clear history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Additional information
st.sidebar.markdown("---")
st.sidebar.info(
    "**Modes:**\n"
    "- *Internet Chat*: General conversation\n"
    "- *PDF Q&A*: Ask questions about uploaded PDFs"
)
