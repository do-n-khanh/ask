# RAG AI Agent (Deepscaler)

A Retrieval-Augmented Generation (RAG) AI agent built with Streamlit that allows users to upload documents and ask questions about their content.

## Features

- Upload text (.txt) or PDF (.pdf) files to create a knowledge base
- Ask questions about the uploaded document content
- Real-time progress updates during document processing
- Persistent vector store that saves between sessions
- Streaming responses from the LLM

## Requirements

- Python 3.8+
- Ollama with the "deepscaler" model and nomic-embed-text installed
- FAISS vector database
- Streamlit
- LangChain

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/do-n-khanh/ask.git
   cd ask
   ```

2. Install required packages:
   ```bash
   pip install streamlit langchain langchain_ollama faiss-cpu pypdf
   ```

3. Ensure you have Ollama installed with the deepscaler model:
   ```bash
   ollama pull deepscaler 
   ollama pull nomic-embed-text
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run ask.py
   ```

2. Upload a text or PDF file using the file uploader.

3. Click "Process File" to create the vector store from your document.

4. Ask questions in the text input box and click "Get Answer".

5. The application will retrieve relevant information from your document and generate an answer.

## How It Works

This application uses a Retrieval-Augmented Generation (RAG) approach:

1. **Document Processing**: Documents are chunked into smaller segments and embedded.
2. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient similarity search.
3. **Question Processing**: When you ask a question, the system finds the most relevant document chunks.
4. **Answer Generation**: The LLM generates an answer based on the retrieved context.

## Project Structure

```
.
├── ask.py          # Main application file
├── faiss_index/    # Directory for persistent vector store (created on first run)
└── readme.md       # This documentation
```

## Application Flow

```
+-------------------+
|   Main App (UI)   | 
| (main())          |
+-------------------+
         |
         v
+-------------------+       +-------------------+
| File Upload       | ----> | Process File      |
| (st.file_uploader)|       | (st.button)       |
+-------------------+       +-------------------+
         |                         |
         v                         v
+-------------------+       +-------------------+
| Session State     |       | Vector Store      |
| (st.session_state)| <---- | Creation/Load     |
+-------------------+       +-------------------+
         |                         |
         v                         v
+-------------------+       +-------------------+
| Question Input    |       | Status Updates    |
| (st.text_input)   |       |                   |
+-------------------+       +-------------------+
         |
         v
+-------------------+
| Get Answer Button |
+-------------------+
         |
         v
+-------------------+       +-------------------+
| Query Logic       | ----> | RAG Agent         |
|                   |       | (query_rag_agent) |
+-------------------+       +-------------------+
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
