# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS   # Vector store DB
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q&A")

# # Initialize LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

# # Prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question.
#     <context>
#     {context}
#     <context>
#     Questions: {input}
#     """
# )

# # Define vector embedding function for uploaded PDFs
# def process_uploaded_pdfs(uploaded_files):
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Load all uploaded PDFs
#     documents = []
#     for uploaded_file in uploaded_files:
#         with open(uploaded_file.name, "wb") as f:
#             f.write(uploaded_file.read())  # Save the file locally
#         loader = PyPDFLoader(uploaded_file.name)
#         documents.extend(loader.load())
    
#     # Split documents into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     final_documents = text_splitter.split_documents(documents)
    
#     # Create a vector store
#     st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
#     st.write("Vector Store DB is ready!")

# # File uploader for PDFs
# uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# # Automatically process PDFs when they are uploaded
# if uploaded_files:
#     st.write("Processing uploaded PDFs...")
#     process_uploaded_pdfs(uploaded_files)

# # Input field for question
# prompt1 = st.text_input("Enter Your Question From Documents", key="question")

# # Add a button for mouse users
# button_clicked = st.button("Submit Question")

# # If user presses Enter or clicks the button, process the input
# if prompt1 and (st.session_state.question or button_clicked):
#     if "vectors" not in st.session_state:
#         st.error("Please upload and process PDFs first!")
#     else:
#         # Create document retrieval and QA chain
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         # Measure response time and generate an answer
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         st.write("Response time:", time.process_time() - start)
#         st.write(response['answer'])

#         # Display relevant document chunks with expander
#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------------")
















# import os
# import tempfile
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# # Load environment variables
# load_dotenv()

# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.title("Gemma Model Document Q&A")

# # Initialize the LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

# # Initialize the chat history if it doesn't exist
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Initialize session state
# def initialize_session_state():
#     if "loader" not in st.session_state:
#         st.session_state.loader = None
#     if "final_documents" not in st.session_state:
#         st.session_state.final_documents = None
#     if "embeddings" not in st.session_state:
#         st.session_state.embeddings = None
#     if "vectors" not in st.session_state:
#         st.session_state.vectors = None
#     if "text_splitter" not in st.session_state:
#         st.session_state.text_splitter = None

# # Call initialization at the start
# initialize_session_state()

# # Function to handle file upload and process PDFs automatically
# def process_uploaded_pdf(uploaded_files):
#     """Process uploaded PDFs to create vector embeddings."""
#     if uploaded_files:
#         st.session_state.loader = []
#         for uploaded_file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                 tmp_file.write(uploaded_file.read())  # Write the content of the uploaded file
#                 tmp_file_path = tmp_file.name  # Get the path of the temporary file

#             # Use PyPDFLoader with the temporary file path
#             loader = PyPDFLoader(tmp_file_path)
#             st.session_state.loader.extend(loader.load())

#         # Perform text splitting and vector embedding
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.loader)
#         st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

#         st.write("Vector Store DB is ready!")
#     else:
#         st.warning("Please upload at least one PDF file.")

# # Automatically process PDFs when files are uploaded
# uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# # Trigger PDF processing automatically when files are uploaded
# if uploaded_files:
#     process_uploaded_pdf(uploaded_files)

# # Function to handle Q&A
# def ask_question(prompt1):
#     if prompt1 and st.session_state.vectors:
#         # Correcting the reference to use prompt1
#         prompt_template = ChatPromptTemplate.from_template(
#             """
#             Answer the questions based on the provided context only.
#             Please provide the most accurate response based on the question
#             <context>
#             {context}
#             </context>
#             Questions: {input}
#             """
#         )
        
#         document_chain = create_stuff_documents_chain(llm, prompt_template)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         response_time = time.process_time() - start

#         # Store question and answer in chat history
#         st.session_state.chat_history.append({"question": prompt1, "answer": response['answer'], "time": response_time})
        
#     elif prompt1:
#         st.warning("Please process the uploaded PDFs first!")

# # Display the chat-like interface
# for message in st.session_state.chat_history:
#     st.write(f"**User:** {message['question']}")
#     st.write(f"**Gemma:** {message['answer']}")
#     st.write(f"Response Time: {message['time']} seconds")
#     st.write("--------------------------------")

# # Question input box with key for proper handling
# prompt1 = st.text_input("Ask a question:", key="question_input")

# # Provide an option to submit the question using a button
# if st.button("Submit Question") and prompt1:
    # ask_question(prompt1)
    # # Reset the input field only after submission
    # st.session_state.question_input = ""  # Reset input after submitting

















# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# st.title("PDF Document Q&A")

# # Initialize session state for chat history
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'vectors' not in st.session_state:
#     st.session_state.vectors = None

# # Initialize LLM
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

# # Prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate response based on the question.
#     <context>
#     {context}
#     </context>
#     Question: {input}
#     """
# )

# # Define vector embedding function for uploaded PDFs
# def process_uploaded_pdfs(uploaded_files):
#     # Initialize embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Load all uploaded PDFs
#     documents = []
#     for uploaded_file in uploaded_files:
#         with open(uploaded_file.name, "wb") as f:
#             f.write(uploaded_file.read())  # Save the file locally
#         loader = PyPDFLoader(uploaded_file.name)
#         documents.extend(loader.load())

#     # Split documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     final_documents = text_splitter.split_documents(documents)

#     # Create the vector store with the embedded documents
#     vectors = FAISS.from_documents(final_documents, embeddings)
#     return vectors

# # File uploader for PDFs
# uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# # Create a central area for chat interface
# chat_container = st.container()

# # Input area for questions
# question_input = st.text_input("Ask a question about the uploaded documents")

# # Process button
# process_button = st.button("Get Answer")

# # Main processing logic
# if uploaded_files and process_button and question_input:
#     with st.spinner('Processing documents and generating answer...'):
#         try:
#             # Process PDFs and create vector store
#             st.session_state.vectors = process_uploaded_pdfs(uploaded_files)

#             # Create document retrieval and QA chain
#             document_chain = create_stuff_documents_chain(llm, prompt)
#             retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 most relevant docs

#             # Create retrieval chain
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)

#             # Generate answer
#             response = retrieval_chain.invoke({
#                 'input': question_input
#             })

#             # Update and display chat history
#             if response and 'answer' in response:
#                 st.session_state.chat_history.append({
#                     'question': question_input,
#                     'answer': response['answer']
#                 })
#             else:
#                 st.error("No answer could be generated.")
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
#             # Print full traceback for debugging
#             import traceback
#             st.error(traceback.format_exc())

# # Display chat history
# st.subheader("Chat History")
# for chat in st.session_state.chat_history:
#     st.markdown(f"**Q: {chat['question']}**")
#     st.markdown(f"A: {chat['answer']}")
#     st.divider()

# # Optional: Clear chat history
# if st.button("Clear Chat History"):
#     st.session_state.chat_history = []



#########################single pdf upload#######################################3333


# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# # Set page configuration
# st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="wide")

# # Title
# st.title("📄 PDF Document Q&A Chatbot")

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'vectors' not in st.session_state:
#     st.session_state.vectors = None

# # Initialize LLM
# @st.cache_resource
# def initialize_llm():
#     return ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

# llm = initialize_llm()

# # Prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based ONLY on the provided context. 
#     If the answer is not in the context, say "I cannot find the answer in the uploaded documents."
    
#     Context:
#     {context}
    
#     Question: {input}
#     """
# )

# # Define vector embedding function for uploaded PDFs
# def process_uploaded_pdfs(uploaded_files):
#     # Initialize embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Load all uploaded PDFs
#     documents = []
#     for uploaded_file in uploaded_files:
#         # Ensure unique filename to avoid overwriting
#         unique_filename = f"temp_{uploaded_file.name}"
#         with open(unique_filename, "wb") as f:
#             f.write(uploaded_file.read())  # Save the file locally
#         loader = PyPDFLoader(unique_filename)
#         documents.extend(loader.load())

#     # Split documents into smaller chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     final_documents = text_splitter.split_documents(documents)

#     # Create the vector store with the embedded documents
#     vectors = FAISS.from_documents(final_documents, embeddings)
#     return vectors

# # Sidebar for PDF uploads
# st.sidebar.header("Upload Documents")
# uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# # Process PDFs if uploaded
# if uploaded_files:
#     with st.sidebar.status("Processing documents..."):
#         st.session_state.vectors = process_uploaded_pdfs(uploaded_files)
#     st.sidebar.success("Documents processed successfully!")

# # Chat input
# if st.session_state.vectors:
#     # Render existing chat history
#     for chat in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(chat['question'])
#         with st.chat_message("assistant"):
#             st.write(chat['answer'])

#     # Chat input with enter key support
#     prompt_input = st.chat_input("Ask a question about the uploaded documents")

#     # Process the question
#     if prompt_input:
#         try:
#             # Create document retrieval and QA chain
#             document_chain = create_stuff_documents_chain(llm, prompt)
#             retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})

#             # Create retrieval chain
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)

#             # Generate answer
#             with st.chat_message("user"):
#                 st.write(prompt_input)
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Generating answer..."):
#                     response = retrieval_chain.invoke({
#                         'input': prompt_input
#                     })

#                     # Extract and display answer
#                     if response and 'answer' in response:
#                         answer = response['answer']
#                         st.write(answer)

#                         # Save to chat history
#                         st.session_state.chat_history.append({
#                             'question': prompt_input,
#                             'answer': answer
#                         })
#                     else:
#                         st.error("Could not generate an answer.")
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
# else:
#     # Instruction for uploading documents
#     st.info("Please upload PDF documents in the sidebar to start chatting.")

# # Sidebar clear history button
# if st.sidebar.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     st.experimental_rerun()





################Multiple pdf upload########################


# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import time
# import uuid

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# # Set page configuration
# st.set_page_config(page_title="Multi-PDF Q&A Chatbot", page_icon="📄", layout="wide")

# # Title
# st.title("📄PDF Document Q&A Chatbot")

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'pdf_vectors' not in st.session_state:
#     st.session_state.pdf_vectors = {}

# # Initialize LLM
# @st.cache_resource
# def initialize_llm():
#     return ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

# llm = initialize_llm()

# # Prompt template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based ONLY on the provided context. 
#     If the answer is not in the context, clearly state "I cannot find the answer in the uploaded document."
    
#     Context:
#     {context}
    
#     Question: {input}
#     """
# )

# # Define vector embedding function for uploaded PDFs
# def process_uploaded_pdfs(uploaded_files):
#     # Initialize embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Dictionary to store vectors for each PDF
#     pdf_vectors = {}

#     # Load all uploaded PDFs
#     for uploaded_file in uploaded_files:
#         # Generate a unique ID for each PDF
#         pdf_id = str(uuid.uuid4())
        
#         # Ensure unique filename to avoid overwriting
#         unique_filename = f"temp_{pdf_id}_{uploaded_file.name}"
#         with open(unique_filename, "wb") as f:
#             f.write(uploaded_file.read())  # Save the file locally
        
#         # Load and process PDF
#         loader = PyPDFLoader(unique_filename)
#         documents = loader.load()

#         # Split documents into smaller chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         final_documents = text_splitter.split_documents(documents)

#         # Create vector store for this specific PDF
#         vectors = FAISS.from_documents(final_documents, embeddings)
        
#         # Store vectors with original filename for reference
#         pdf_vectors[uploaded_file.name] = {
#             'vectors': vectors,
#             'temp_filename': unique_filename
#         }

#     return pdf_vectors

# # Sidebar for PDF uploads
# st.sidebar.header("Upload Documents")
# uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# # Process PDFs if uploaded
# if uploaded_files:
#     with st.sidebar.status("Processing documents..."):
#         st.session_state.pdf_vectors = process_uploaded_pdfs(uploaded_files)
#     st.sidebar.success(f"{len(uploaded_files)} documents processed!")

# # PDF Selection for Querying
# selected_pdf = None
# if st.session_state.pdf_vectors:
#     # Create a dropdown to select PDF
#     pdf_options = list(st.session_state.pdf_vectors.keys())
#     selected_pdf = st.sidebar.selectbox(
#         "Select PDF to query", 
#         pdf_options, 
#         index=0
#     )

# # Chat input and processing
# if st.session_state.pdf_vectors:
#     # Render existing chat history
#     for chat in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(chat['question'])
#         with st.chat_message("assistant"):
#             st.write(chat['answer'])

#     # Chat input with enter key support
#     prompt_input = st.chat_input("Ask a question about the selected document")

#     # Process the question
#     if prompt_input and selected_pdf:
#         try:
#             # Get the vector store for the selected PDF
#             current_vectors = st.session_state.pdf_vectors[selected_pdf]['vectors']

#             # Create document retrieval and QA chain
#             document_chain = create_stuff_documents_chain(llm, prompt)
#             retriever = current_vectors.as_retriever(search_kwargs={"k": 3})

#             # Create retrieval chain
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)

#             # Generate answer
#             with st.chat_message("user"):
#                 st.write(f"[From {selected_pdf}] {prompt_input}")
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Generating answer..."):
#                     response = retrieval_chain.invoke({
#                         'input': prompt_input
#                     })

#                     # Extract and display answer
#                     if response and 'answer' in response:
#                         answer = response['answer']
#                         st.write(answer)

#                         # Save to chat history
#                         st.session_state.chat_history.append({
#                             'question': f"[From {selected_pdf}] {prompt_input}",
#                             'answer': answer
#                         })
#                     else:
#                         st.error("Could not generate an answer.")
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")
# else:
#     # Instruction for uploading documents
#     st.info("Please upload PDF documents in the sidebar to start chatting.")

# # Sidebar clear history button
# if st.sidebar.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     st.experimental_rerun()





################Internet enabled search bar#######################3





# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import uuid

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# # Set page configuration
# st.set_page_config(page_title="Multi-Mode Chatbot", page_icon="💬", layout="wide")

# # Title
# st.title("💬 Multi-Mode Chatbot")

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'pdf_vectors' not in st.session_state:
#     st.session_state.pdf_vectors = {}

# # Initialize LLMs
# @st.cache_resource
# def initialize_llms():
#     return {
#         'pdf': ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it"),
#         'internet': ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
#     }

# llms = initialize_llms()

# # PDF Prompt template
# pdf_prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the questions based ONLY on the provided context. 
#     If the answer is not in the context, clearly state "I cannot find the answer in the uploaded document."
    
#     Context:
#     {context}
    
#     Question: {input}
#     """
# )

# # Internet Chat Prompt
# internet_prompt = ChatPromptTemplate.from_template(
#     """
#     You are a helpful, respectful, and honest assistant. 
#     Always answer to the best of your ability while being clear and precise.
    
#     Question: {input}
#     """
# )

# # Define vector embedding function for uploaded PDFs
# def process_uploaded_pdfs(uploaded_files):
#     # Initialize embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Dictionary to store vectors for each PDF
#     pdf_vectors = {}

#     # Load all uploaded PDFs
#     for uploaded_file in uploaded_files:
#         # Generate a unique ID for each PDF
#         pdf_id = str(uuid.uuid4())
        
#         # Ensure unique filename to avoid overwriting
#         unique_filename = f"temp_{pdf_id}_{uploaded_file.name}"
#         with open(unique_filename, "wb") as f:
#             f.write(uploaded_file.read())  # Save the file locally
        
#         # Load and process PDF
#         loader = PyPDFLoader(unique_filename)
#         documents = loader.load()

#         # Split documents into smaller chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         final_documents = text_splitter.split_documents(documents)

#         # Create vector store for this specific PDF
#         vectors = FAISS.from_documents(final_documents, embeddings)
        
#         # Store vectors with original filename for reference
#         pdf_vectors[uploaded_file.name] = {
#             'vectors': vectors,
#             'temp_filename': unique_filename
#         }

#     return pdf_vectors

# # Sidebar for Mode Selection and PDF Upload
# st.sidebar.header("Chatbot Settings")
# chat_mode = st.sidebar.radio("Select Chat Mode", 
#     ["Internet Chat", "PDF Q&A"], 
#     index=0
# )

# # PDF Upload for PDF Q&A Mode
# if chat_mode == "PDF Q&A":
#     st.sidebar.subheader("Upload Documents")
#     uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

#     # Process PDFs if uploaded
#     if uploaded_files:
#         with st.sidebar.status("Processing documents..."):
#             st.session_state.pdf_vectors = process_uploaded_pdfs(uploaded_files)
#         st.sidebar.success(f"{len(uploaded_files)} documents processed!")

# # Chat Interface
# if chat_mode == "PDF Q&A" and st.session_state.pdf_vectors:
#     # PDF Selection for Querying
#     pdf_options = list(st.session_state.pdf_vectors.keys())
#     selected_pdf = st.sidebar.selectbox(
#         "Select PDF to query", 
#         pdf_options, 
#         index=0
#     )

#     # Render existing chat history
#     for chat in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(chat['question'])
#         with st.chat_message("assistant"):
#             st.write(chat['answer'])

#     # Chat input with enter key support
#     prompt_input = st.chat_input("Ask a question about the selected document")

#     # Process the question
#     if prompt_input and selected_pdf:
#         try:
#             # Get the vector store for the selected PDF
#             current_vectors = st.session_state.pdf_vectors[selected_pdf]['vectors']

#             # Create document retrieval and QA chain
#             document_chain = create_stuff_documents_chain(llms['pdf'], pdf_prompt)
#             retriever = current_vectors.as_retriever(search_kwargs={"k": 3})

#             # Create retrieval chain
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)

#             # Generate answer
#             with st.chat_message("user"):
#                 st.write(f"[From {selected_pdf}] {prompt_input}")
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Generating answer..."):
#                     response = retrieval_chain.invoke({
#                         'input': prompt_input
#                     })

#                     # Extract and display answer
#                     if response and 'answer' in response:
#                         answer = response['answer']
#                         st.write(answer)

#                         # Save to chat history
#                         st.session_state.chat_history.append({
#                             'question': f"[From {selected_pdf}] {prompt_input}",
#                             'answer': answer
#                         })
#                     else:
#                         st.error("Could not generate an answer.")
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# elif chat_mode == "Internet Chat":
#     # Render existing chat history
#     for chat in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(chat['question'])
#         with st.chat_message("assistant"):
#             st.write(chat['answer'])

#     # Chat input with enter key support
#     prompt_input = st.chat_input("Ask me anything...")

#     # Process the question
#     if prompt_input:
#         try:
#             # Generate answer using Internet Chat mode
#             with st.chat_message("user"):
#                 st.write(prompt_input)
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Generating response..."):
#                     # Create chat chain
#                     chat_chain = llms['internet'].invoke([
#                         {"role": "system", "content": internet_prompt.format(input=prompt_input)},
#                         {"role": "user", "content": prompt_input}
#                     ])

#                     # Extract and display answer
#                     answer = chat_chain.content
#                     st.write(answer)

#                     # Save to chat history
#                     st.session_state.chat_history.append({
#                         'question': prompt_input,
#                         'answer': answer
#                     })
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# else:
#     # Instruction for uploading documents in PDF mode
#     st.info("Please upload PDF documents in the sidebar to start chatting.")

# # Sidebar clear history button
# if st.sidebar.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     st.rerun()

# # Additional information
# st.sidebar.markdown("---")
# st.sidebar.info(
#     "**Modes:**\n"
#     "- *Internet Chat*: General conversation\n"
#     "- *PDF Q&A*: Ask questions about uploaded PDFs"
# )




# import os
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# import uuid

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# # Set page configuration
# st.set_page_config(page_title="Multi-Mode Chatbot", page_icon="💬", layout="wide")

# # Title
# st.title("💬 Multi-Mode Chatbot")

# # Initialize session state
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'pdf_vectors' not in st.session_state:
#     st.session_state.pdf_vectors = {}

# # Initialize LLMs
# @st.cache_resource
# def initialize_llms():
#     return {
#         'pdf': ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it"),
#         'internet': ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
#     }

# llms = initialize_llms()

# # PDF Prompt template with comprehensive answer instructions
# pdf_prompt = ChatPromptTemplate.from_template(
#     """
#     Provide a detailed and comprehensive answer based ONLY on the provided context. 
#     Ensure you extract and present all relevant information from the documents.
#     If the answer requires multiple paragraphs or involves details from different parts of the context, include them fully.
    
#     If no clear answer can be found, state "I cannot find a comprehensive answer in the uploaded document."
    
#     Context:
#     {context}
    
#     Question: {input}
    
#     Answer with full detail, using all available relevant information from the context.
#     """
# )

# # Internet Chat Prompt
# internet_prompt = ChatPromptTemplate.from_template(
#     """
#     You are a helpful, respectful, and honest assistant. 
#     Always answer to the best of your ability while being clear and precise.
    
#     Question: {input}
#     """
# )

# # Define vector embedding function for uploaded PDFs
# def process_uploaded_pdfs(uploaded_files):
#     # Initialize embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Dictionary to store vectors for each PDF
#     pdf_vectors = {}

#     # Load all uploaded PDFs
#     for uploaded_file in uploaded_files:
#         # Generate a unique ID for each PDF
#         pdf_id = str(uuid.uuid4())
        
#         # Ensure unique filename to avoid overwriting
#         unique_filename = f"temp_{pdf_id}_{uploaded_file.name}"
#         with open(unique_filename, "wb") as f:
#             f.write(uploaded_file.read())  # Save the file locally
        
#         # Load and process PDF
#         loader = PyPDFLoader(unique_filename)
#         documents = loader.load()

#         # Split documents into smaller chunks
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         final_documents = text_splitter.split_documents(documents)

#         # Create vector store for this specific PDF
#         vectors = FAISS.from_documents(final_documents, embeddings)
        
#         # Store vectors with original filename for reference
#         pdf_vectors[uploaded_file.name] = {
#             'vectors': vectors,
#             'temp_filename': unique_filename
#         }

#     return pdf_vectors

# # Sidebar for Mode Selection and PDF Upload
# st.sidebar.header("Chatbot Settings")
# chat_mode = st.sidebar.radio("Select Chat Mode", 
#     ["Internet Chat", "PDF Q&A"], 
#     index=0
# )

# # PDF Upload for PDF Q&A Mode
# if chat_mode == "PDF Q&A":
#     st.sidebar.subheader("Upload Documents")
#     uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

#     # Process PDFs if uploaded
#     if uploaded_files:
#         with st.sidebar.status("Processing documents..."):
#             st.session_state.pdf_vectors = process_uploaded_pdfs(uploaded_files)
#         st.sidebar.success(f"{len(uploaded_files)} documents processed!")

# # Chat Interface
# if chat_mode == "PDF Q&A" and st.session_state.pdf_vectors:
#     # PDF Selection for Querying
#     pdf_options = list(st.session_state.pdf_vectors.keys())
#     selected_pdf = st.sidebar.selectbox(
#         "Select PDF to query", 
#         pdf_options, 
#         index=0
#     )

#     # Render existing chat history
#     for chat in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(chat['question'])
#         with st.chat_message("assistant"):
#             st.write(chat['answer'])

#     # Chat input with enter key support
#     prompt_input = st.chat_input("Ask a question about the selected document")

#     # Process the question
#     if prompt_input and selected_pdf:
#         try:
#             # Get the vector store for the selected PDF
#             current_vectors = st.session_state.pdf_vectors[selected_pdf]['vectors']

#             # Create document retrieval and QA chain with enhanced retrieval
#             document_chain = create_stuff_documents_chain(
#                 llms['pdf'], 
#                 ChatPromptTemplate.from_template(
#                     """
#                     Based on the provided context, give a comprehensive and detailed answer to the question.
#                     If the answer spans multiple documents/chunks, ensure to include all relevant information.
                    
#                     Context:
#                     {context}
                    
#                     Question: {input}
                    
#                     Answer with as much detail as possible from the available context.
#                     """
#                 )
#             )

#             # Enhanced retriever with more document retrieval
#             retriever = current_vectors.as_retriever(
#                 search_kwargs={
#                     "k": 10,  # Increase number of retrieved documents
#                     "fetch_k": 20  # Increase initial fetch to have more documents to choose from
#                 }
#             )

#             # Create retrieval chain
#             retrieval_chain = create_retrieval_chain(retriever, document_chain)

#             # Generate answer
#             with st.chat_message("user"):
#                 st.write(f"[From {selected_pdf}] {prompt_input}")
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Generating comprehensive answer..."):
#                     response = retrieval_chain.invoke({
#                         'input': prompt_input
#                     })

#                     # Extract and display answer
#                     if response and 'answer' in response:
#                         answer = response['answer']
#                         st.write(answer)

#                         # Save to chat history
#                         st.session_state.chat_history.append({
#                             'question': f"[From {selected_pdf}] {prompt_input}",
#                             'answer': answer
#                         })
#                     else:
#                         st.error("Could not generate an answer.")
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# elif chat_mode == "Internet Chat":
#     # Render existing chat history
#     for chat in st.session_state.chat_history:
#         with st.chat_message("user"):
#             st.write(chat['question'])
#         with st.chat_message("assistant"):
#             st.write(chat['answer'])

#     # Chat input with enter key support
#     prompt_input = st.chat_input("Ask me anything...")

#     # Process the question
#     if prompt_input:
#         try:
#             # Generate answer using Internet Chat mode
#             with st.chat_message("user"):
#                 st.write(prompt_input)
            
#             with st.chat_message("assistant"):
#                 with st.spinner("Generating response..."):
#                     # Create chat chain
#                     chat_chain = llms['internet'].invoke([
#                         {"role": "system", "content": internet_prompt.format(input=prompt_input)},
#                         {"role": "user", "content": prompt_input}
#                     ])

#                     # Extract and display answer
#                     answer = chat_chain.content
#                     st.write(answer)

#                     # Save to chat history
#                     st.session_state.chat_history.append({
#                         'question': prompt_input,
#                         'answer': answer
#                     })
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")

# else:
#     # Instruction for uploading documents in PDF mode
#     st.info("Please upload PDF documents in the sidebar to start chatting.")

# # Sidebar clear history button
# if st.sidebar.button("Clear Chat History"):
#     st.session_state.chat_history = []
#     st.rerun()

# # Additional information
# st.sidebar.markdown("---")
# st.sidebar.info(
#     "**Modes:**\n"
#     "- *Internet Chat*: General conversation\n"
#     "- *PDF Q&A*: Ask questions about uploaded PDFs"
# )


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
        'pdf': ChatGroq(groq_api_key=groq_api_key, model_name="llama3.3"),
        'internet': ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
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
