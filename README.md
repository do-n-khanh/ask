# RAG AI Agent (OpenAI)

A Retrieval-Augmented Generation (RAG) AI agent built with Streamlit that allows users to upload documents and ask questions about their content.

## Features

- Upload text (.txt) or PDF (.pdf) files to create a knowledge base
- Ask questions about the uploaded document content
- Real-time progress updates during document processing
- In-session vector store (resets when the session ends)
- Streaming responses from the LLM

## Requirements

- Python 3.8+
- OpenAI API key
- FAISS vector database
- Streamlit
- LangChain

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/do-n-khanh/ask.git
   cd rag-ai-agent
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run ask.py
   ```

2. Enter your OpenAI API key in the sidebar.

3. Upload a text or PDF file using the file uploader.

4. Click "Process File" to create the vector store from your document.

5. Ask questions in the text input box and click "Get Answer".

6. The application will retrieve relevant information from your document and generate an answer.

## How It Works

This application uses a Retrieval-Augmented Generation (RAG) approach:

1. **Document Processing**: Documents are chunked into smaller segments and embedded using OpenAI's text-embedding-3-small model.
2. **Vector Storage**: Embeddings are stored in a FAISS vector database in the session state.
3. **Question Processing**: When you ask a question, the system finds the most relevant document chunks.
4. **Answer Generation**: GPT-3.5 Turbo generates an answer based on the retrieved context.

## Project Structure

```
.
├── ask.py          # Main application file
├── requirements.txt # Package requirements
└── README.md       # This documentation
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
| API Key Input     | ----> | File Upload       |
| (st.sidebar)      |       | (st.file_uploader)|
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
