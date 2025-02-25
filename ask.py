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
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
import time

# Define a path to save/load the vector store
VECTOR_STORE_PATH = "faiss_index"

# Function to process uploaded file and create vector store with real-time updates
def create_vector_store(uploaded_file, status_placeholder):
    try:
        status_placeholder.write("Step 1/4: Saving uploaded file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        status_placeholder.write("Step 2/4: Loading file content...")
        if uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_file_path)
        elif uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:
            st.error("Unsupported file type. Please upload a .txt or .pdf file.")
            return None
        documents = loader.load()
        time.sleep(0.5)

        status_placeholder.write("Step 3/4: Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        status_placeholder.write(f"Created {len(chunks)} chunks.")
        time.sleep(0.5)

        status_placeholder.write("Step 4/4: Generating embeddings and building vector store...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Using a dedicated embedding model
        progress_bar = st.progress(0)
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)
            time.sleep(0.01)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        progress_bar.progress(1.0)
        status_placeholder.write("Vector store built successfully!")

        vector_store.save_local(VECTOR_STORE_PATH)
        os.unlink(tmp_file_path)
        return vector_store

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Function to load or create vector store
def get_vector_store(uploaded_file, status_placeholder):
    if os.path.exists(VECTOR_STORE_PATH) and not uploaded_file:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Using a dedicated embedding model
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    elif uploaded_file:
        return create_vector_store(uploaded_file, status_placeholder)
    return None

# Function to query the RAG agent with streaming
def query_rag_agent(question, vector_store, output_placeholder):
    llm = OllamaLLM(model="deepscaler", temperature=0.1)
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant. Answer the question based solely on the provided context. If the answer isn’t in the context, say "I don’t know."
    
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
        answer_text += chunk
        if "I don’t know" in answer_text:
            output_placeholder.write("I don’t know.")
            return
        output_placeholder.write(answer_text)

# Streamlit app
def main():
    st.title("RAG AI Agent (Deepscaler)")
    st.write("Upload a text or PDF file to create a local knowledge base, then ask questions about its content.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file (optional if already processed)", type=["txt", "pdf"])

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
            st.success("File processed and saved successfully! You can now ask questions.")

    # Load existing vector store if no new file uploaded
    if not uploaded_file and not st.session_state.vector_store and os.path.exists(VECTOR_STORE_PATH):
        st.session_state.vector_store = get_vector_store(None, status_placeholder)
        st.info("Loaded existing knowledge base from disk.")

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

if __name__ == "__main__":
    main()