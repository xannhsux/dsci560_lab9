# DSCI-560 Lab 9: PDF Q&A Chatbot

A sophisticated Question-and-Answer chatbot using Retrieval-Augmented Generation (RAG) with Large Language Models and vector embeddings.

---

## Features

- **Multi-Provider LLM Support**: OpenAI, Gemini, or Ollama (local)
- **Free Embeddings**: HuggingFace Sentence Transformers
- **Vector Database**: FAISS with cosine similarity search
- **Modern Web UI**: Clean, dark-themed chat interface
- **Conversation Memory**: Maintains chat history across questions
- **Source Retrieval**: Uses top-4 most relevant document chunks

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Gemini API Key (get from https://aistudio.google.com/app/apikey)

### 2. Installation

```bash
# Clone repository (if needed)
cd dsci560_lab9

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

**Option A: Using .env file (Recommended)**

The `.env` file is already configured. Just add your Gemini API key:

```bash
# Edit .env file
nano .env

# Update this line:
GOOGLE_API_KEY=your-actual-gemini-api-key-here
```

**Option B: Using environment variable**

```bash
export GEMINI_API_KEY=your-actual-gemini-api-key-here
```

The code automatically checks both `GOOGLE_API_KEY` and `GEMINI_API_KEY`.

### 4. Start Backend Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start FastAPI backend
uvicorn beckend.app:app --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5. Start Frontend Server

In a new terminal:

```bash
cd frontend
python3 -m http.server 3000
```

### 6. Open Web Interface

Open your browser to: **http://localhost:3000**

---

## Usage

### Uploading PDFs

1. In the sidebar, click **"Upload PDFs"**
2. Select one or more PDF files
3. Click **"Analyze PDFs"**
4. Wait for success message: "Vector store ready"

### Asking Questions

1. Type your question in the input field
2. Press **Enter** or click **Send**
3. The chatbot will:
   - Search the vector database
   - Retrieve relevant context
   - Generate an answer using Gemini 2.5 Pro

### Example Questions

```
What is this document about?
Explain the binomial random variable
How do I create a new workspace in ADS?
What are the main topics in Chapter 3?
```

---

## Architecture

```
┌─────────────────┐
│   Frontend      │ HTML/JS @ http://localhost:3000
│  (index.html)   │
└────────┬────────┘
         │ HTTP/REST API
         ↓
┌─────────────────┐
│   Backend       │ FastAPI @ http://localhost:8000
│   (app.py)      │
└────────┬────────┘
         │
    ┌────┴─────┬──────────┬──────────┐
    ↓          ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌───────┐ ┌────────┐
│ Vector │ │Chatbot │ │SQLite │ │ Gemini │
│  FAISS │ │ Engine │ │  DB   │ │  API   │
└────────┘ └────────┘ └───────┘ └────────┘
```

### RAG Pipeline

1. **Ingestion** (`/ingest` endpoint):
   - Extract text from PDFs (PyPDF2)
   - Split into 500-char chunks with 50-char overlap
   - Generate embeddings (HuggingFace)
   - Store in FAISS with cosine similarity

2. **Query** (`/ask` endpoint):
   - Embed user question
   - Retrieve top-4 similar chunks (k=4)
   - Format prompt with context + question
   - Send to Gemini 2.5 Pro
   - Return answer

---

## Configuration

Edit `.env` to customize:

```bash
# LLM Provider (openai, gemini, or oss)
PROVIDER=gemini

# Gemini Settings
GEMINI_MODEL=gemini-2.5-pro
GEMINI_EMBED_MODEL=models/embedding-001

# Embedding Provider (separate from LLM)
EMBED_PROVIDER=oss  # Use free HuggingFace embeddings

# HuggingFace Embeddings
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunking Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Temperature (0.0 = deterministic)
TEMP=0.0
```

---

## Project Structure

```
dsci560_lab9/
├── beckend/                 # Backend API (note: typo in original)
│   ├── app.py              # FastAPI application
│   ├── chatbot.py          # Conversation engine
│   └── vectorstore.py      # FAISS & embeddings
├── frontend/
│   └── index.html          # Web UI
├── data/
│   ├── vectorstore/faiss/  # FAISS index
│   ├── uploads/            # Uploaded PDFs
│   └── ads_texts.db        # SQLite database
├── .env                    # Configuration (API keys)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## Technical Details

### Embeddings
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Metric**: Cosine similarity
- **Provider**: HuggingFace (free, local)

### Vector Database
- **Engine**: FAISS
- **Index Type**: Flat (with L2 normalization for cosine)
- **Retrieval**: Top-k similarity search (k=4)

### LLM
- **Model**: Gemini 2.5 Pro
- **Max Input**: 2M tokens
- **Max Output**: 8,192 tokens
- **Typical Usage**: ~1,000 tokens/query

### Text Processing
- **Splitter**: CharacterTextSplitter (LangChain)
- **Separator**: Newline
- **Chunk Size**: 500 characters
- **Overlap**: 50 characters
- **Chunks per Query**: 4

---

## Implementation Notes

### Lab Requirements Met

All required components implemented:
- **PDF Text Extraction**: PyPDF2 for direct text extraction
- **Text Chunking**: CharacterTextSplitter with 500-char chunks
- **Vector Database**: FAISS with embeddings
- **Conversation Chain**: Custom ConversationEngine class
- **Open-Source Embeddings**: HuggingFace Sentence Transformers
- **Web Interface**: HTML/CSS/JavaScript with chat UI
- **PDF Upload**: Multi-file upload with analysis

### Design Decisions

1. **Gemini over OpenAI**: Better performance, generous free tier
2. **HuggingFace Embeddings**: No cost, works offline, good quality
3. **Cosine Similarity**: Industry standard for semantic search
4. **FastAPI**: Modern, async, auto-documented API
5. **FAISS**: Fast approximate nearest neighbor search

---

## Troubleshooting

### "Vector store not found"
**Solution**: Upload PDFs via the web interface first

### "Failed to fetch" in browser
**Solution**: Make sure backend is running on port 8000

### "GEMINI_API_KEY not found"
**Solution**: Check your `.env` file or set environment variable

### Slow embedding generation
**Solution**: First time downloads the model (~80MB), subsequent runs are fast

### CORS errors
**Solution**: Access frontend via http://localhost:3000, not file://

---

## Performance

- **Embedding Generation**: ~1-2 seconds (first time: model download)
- **Vector Search**: ~0.3 seconds (4 chunks from 120 documents)
- **LLM Response**: ~5-20 seconds (depends on answer complexity)
- **Total Query Time**: ~6-25 seconds

---

## Security

- API keys stored in `.env` (gitignored)
- Keys never exposed in frontend
- No API keys in source code
- CORS configured for localhost only

---

## Dependencies

Core libraries:
- `fastapi` - Web API framework
- `uvicorn` - ASGI server
- `langchain` - LLM orchestration
- `langchain-google-genai` - Gemini integration
- `faiss-cpu` - Vector database
- `sentence-transformers` - Embeddings
- `PyPDF2` - PDF text extraction

See `requirements.txt` for full list.

---

## Team

[Add your team name and member names here]

---

## License

Educational project for DSCI-560, Fall 2025

---

## Acknowledgments

- LangChain for RAG framework
- Google Gemini for LLM API
- HuggingFace for embedding models
- FAISS for vector search
