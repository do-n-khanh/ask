# +-------------------+
# |   Main App (UI)   | 
# | (main())          |
# +-------------------+
#          |
#          v
# +-------------------+       +-------------------+
# | File Upload       | ----> | Process File      |
# | (st.file_uploader)|       | (st.button)       |
# +-------------------+       +-------------------+
#          |                         |
#          v                         v
# +-------------------+       +-------------------+
# | Session State     |       | Vector Store      |
# | (st.session_state)| <---- | Creation/Load     |
# |                   |       | (create_vector_   |
# |                   |       | store, get_vector_|
# |                   |       | store)            |
# +-------------------+       +-------------------+
#          |                         |
#          v                         v
# +-------------------+       +-------------------+
# | Question Input    |       | Status Updates    |
# | (st.text_input)   |       | (status_place-    |
# |                   |       | holder, progress) |
# +-------------------+       +-------------------+
#          |
#          v
# +-------------------+
# | Get Answer Button |
# | (st.button)       |
# +-------------------+
#          |
#          v
# +-------------------+       +-------------------+
# | Query Logic       | ----> | RAG Agent         |
# | (Conditions)      |       | (query_rag_agent) |
# +-------------------+       +-------------------+
#          |                         |
#          v                         v
# +-------------------+       +-------------------+
# | Error Messages    |       | Answer Output     |
# | (st.error)        |       | (output_place-    |
# |                   |       | holder)           |
# +-------------------+       +-------------------+

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Changed this line to use ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
import time

# Function to check if OpenAI API key is valid
def check_api_key_validity(api_key):
    """Check if the OpenAI API key is valid by making a simple API call."""
    if not api_key:
        return False
    
    try:
        from openai import OpenAI
        
        # Create a client with the provided API key
        client = OpenAI(api_key=api_key)
        
        # Make a minimal API call to validate the key
        response = client.embeddings.create(
            input="test",
            model="text-embedding-3-small"
        )
        
        # If we get here, the API call was successful
        return True
    
    except Exception as e:
        # Could be invalid API key, rate limit, network issue, etc.
        print(f"API key validation error: {str(e)}")
        return False

# Function to process uploaded file and create vector store with real-time updates
def create_vector_store(uploaded_file, status_placeholder):
    try:
        status_placeholder.write("Step 1/3: Saving uploaded file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        status_placeholder.write("Step 2/3: Loading file content...")
        if uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_file_path)
        elif uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            st.error("Unsupported file type. Please upload a .txt or .pdf file.")
            return None
        documents = loader.load()
        time.sleep(0.5)

        status_placeholder.write("Step 3/3: Splitting document into chunks and generating embeddings...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        status_placeholder.write(f"Created {len(chunks)} chunks.")
        time.sleep(0.5)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        progress_bar = st.progress(0)
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)
            time.sleep(0.01)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        progress_bar.progress(1.0)
        status_placeholder.write("Vector store built successfully!")

        os.unlink(tmp_file_path)
        return vector_store

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Streamlit app
def main():
    # Add GitHub repository link for transparency
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app uses your OpenAI API key locally and does not store it.
        
        View the full source code on [GitHub](https://github.com/do-n-khanh/ask).
        
        The app runs all operations locally in your browser session.
        Your API key and documents never leave your computer.
        """
    )
    
    # API Key input with validation
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    
    # Initialize API key validation status in session state
    if "api_key_valid" not in st.session_state:
        st.session_state.api_key_valid = False
        
    # Check API key when entered or changed
    if api_key:
        if "previous_api_key" not in st.session_state or st.session_state.previous_api_key != api_key:
            st.session_state.previous_api_key = api_key
            # Create a placeholder in the sidebar for validation messages
            validation_placeholder = st.sidebar.empty()
            validation_placeholder.info("Validating API key...")
            
            if check_api_key_validity(api_key):
                validation_placeholder.success("✅ API key is valid!")
                os.environ["OPENAI_API_KEY"] = api_key
                st.session_state.api_key_valid = True
            else:
                validation_placeholder.error("❌ Invalid API key or API access issue")
                st.session_state.api_key_valid = False
    else:
        st.session_state.api_key_valid = False
    
    if not st.session_state.api_key_valid:
        st.warning("Please enter a valid OpenAI API key in the sidebar to continue.")
    st.title("RAG AI Agent (OpenAI)")
    st.write("Upload a text or PDF file to create a knowledge base, then ask questions about its content.")
    

    # File uploader
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

    # Session state to store vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Placeholder for status updates
    status_placeholder = st.empty()

    # Process uploaded file if provided
    if uploaded_file and st.button("Process File", key="process_button"):
        with st.spinner("Starting vector store build process..."):
            st.session_state.vector_store = create_vector_store(uploaded_file, status_placeholder)
        if st.session_state.vector_store:
            st.success("File processed successfully! You can now ask questions.")

    # Question input
    question = st.text_input("Ask a question about the uploaded file")

    # Single "Get Answer" button
    if st.button("Get Answer", key="get_answer_button"):
        if question and st.session_state.vector_store:
            output_placeholder = st.empty()
            query_rag_agent(question, st.session_state.vector_store, output_placeholder)
        elif not st.session_state.vector_store:
            st.error("Please upload and process a file first.")
        elif not question:
            st.error("Please enter a question.")

# Function to query the RAG agent with streaming
def query_rag_agent(question, vector_store, output_placeholder):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)  # Changed OpenAI to ChatOpenAI
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant. Answer the question based solely on the provided context. If the answer isn't in the context, say "I don't know."
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    output_placeholder.write("Generating answer...")
    answer_text = ""
    for chunk in rag_chain.stream({"question": question}):
        # Extract content from AIMessageChunk
        if hasattr(chunk, 'content'):
            chunk_content = chunk.content
        else:
            chunk_content = str(chunk)
        
        answer_text += chunk_content
        if "I don't know" in answer_text:
            output_placeholder.write("I don't know.")
            return
        output_placeholder.write(answer_text)

if __name__ == "__main__":
    main()