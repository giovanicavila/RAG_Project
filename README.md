# RAG Project

A high-performance Retrieval-Augmented Generation (RAG) system built with Python and FastAPI. The application provides modular endpoints for document ingestion, semantic text chunking, vector embedding generation, and fast semantic retrieval.

## Technical Stack
- **Framework:** FastAPI (Asynchronous API layer powered by Uvicorn and Uvloop)
- **Chunking Engine:** Chonkie (Advanced semantic token-based text splitting)
- **Embeddings Pipeline:** Google GenAI / Gemini Embeddings (`gemini-embedding-001`) & Catsu
- **Vector Database:** ChromaDB (Local vector storage and semantic search)

## Prerequisites
- Python 3.12+
- Google Gemini API Key

## Installation

1. Clone the repository:
```bash
git clone https://github.com
cd RAG_Project
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables. Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-001
```

## Running the Application

Start the local development server with automatic reloading:
```bash
uvicorn main:app --reload
```

Once running, you can access the interactive Swagger API documentation directly at:
http://127.0.0.1:8000/docs

