# RAG Chatbot Backend

This backend service extracts content from the deployed Docusaurus book, generates embeddings using Cohere, and stores them in Qdrant Cloud for RAG chatbot use.

## Prerequisites

- Python 3.10+
- Cohere API key
- Qdrant Cloud instance or local Qdrant server

## Setup

1. Clone the repository
2. Navigate to the backend directory: `cd backend`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Create a `.env` file with your API keys (see `.env.example`)

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```bash
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

## Running the Application

### Direct execution:
```bash
uvicorn main:app --reload --port 8000
```

### Using Docker:
```bash
docker-compose up --build
```

## API Endpoints

- `GET /health` - Health check
- `POST /extract` - Extract content from Docusaurus book
- `POST /process` - Process extracted content into chunks and embeddings
- `POST /store` - Store embeddings in Qdrant vector database
- `POST /search` - Semantic search in the vector database
- `POST /validate` - Validate stored embeddings
- `GET /jobs/{job_id}` - Get job status

## Development

Run tests:
```bash
pytest tests/
```

## Architecture

The application follows a service-oriented architecture with the following components:

- **Models**: Data models for text chunks, embeddings, etc.
- **Services**: Business logic for content processing, embedding generation, and vector database operations
- **API**: FastAPI endpoints for external interaction

## Pipeline

The processing pipeline follows these steps:
1. Content extraction from Docusaurus site
2. Text cleaning and preprocessing
3. Text chunking (512 token target size)
4. Embedding generation using Cohere
5. Storage in Qdrant with metadata
6. Validation of results