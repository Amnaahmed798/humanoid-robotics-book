from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import uuid
import asyncio
from datetime import datetime

import sys
import os

# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import the required modules for the functions mentioned in the plan
# These will be implemented in their respective files
from extract_content import get_all_urls, extract_text_from_url
from chunk_text import chunk_text
from generate_embeddings import embed
from qdrant_operations import create_collection, save_chunk
from services.content_service import ContentService
from services.embedding_service import EmbeddingService
from services.vector_db_service import VectorDBService

# Import retrieval and validation modules
from retrieval.api import router as retrieval_router

app = FastAPI(
    title="RAG Chatbot Backend API",
    description="API for extracting, processing, and storing book content for RAG chatbot use",
    version="1.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job tracking in memory (in production, use a persistent store)
jobs: Dict[str, Dict[str, Any]] = {}

class ExtractRequest(BaseModel):
    base_url: str
    pages: Optional[List[str]] = None

class ExtractResponse(BaseModel):
    job_id: str
    status: str
    total_pages: int

class ProcessRequest(BaseModel):
    job_id: str
    chunk_size: Optional[int] = 512
    overlap: Optional[int] = 50

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    total_chunks: int

class StoreRequest(BaseModel):
    job_id: str
    collection_name: str = "rag_embedding"

class StoreResponse(BaseModel):
    job_id: str
    status: str
    total_embeddings: int

class SearchRequest(BaseModel):
    query: str
    collection_name: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.3

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

class ValidateRequest(BaseModel):
    collection_name: str

class ValidateResponse(BaseModel):
    collection_name: str
    vector_count: int
    expected_count: int
    vector_dimensions: int
    model_name: str
    is_valid: bool
    issues: List[str]

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now()
    }

@app.post("/extract", response_model=ExtractResponse)
async def extract_content(request: ExtractRequest, background_tasks: BackgroundTasks):
    """Extract content from Docusaurus book"""
    job_id = f"extract_{uuid.uuid4().hex[:8]}"

    # Initialize job tracking
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "extract"
    }

    # In a real implementation, this would be done in the background
    # For now, we'll simulate the process
    try:
        # Get all URLs from the Docusaurus site
        urls = await get_all_urls(request.base_url)

        if request.pages:
            # Filter to only requested pages
            urls = [url for url in urls if any(page in url for page in request.pages)]

        # Update job with total pages
        jobs[job_id]["total_pages"] = len(urls)
        jobs[job_id]["urls"] = urls

        # This would be run in the background in a real implementation
        # background_tasks.add_task(_process_extraction, job_id, urls)

        return ExtractResponse(
            job_id=job_id,
            status="processing",
            total_pages=len(urls)
        )
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessResponse)
async def process_content(request: ProcessRequest):
    """Process extracted content into chunks and embeddings"""
    job_id = f"process_{uuid.uuid4().hex[:8]}"

    # Initialize job tracking
    jobs[job_id] = {
        "status": "processing",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "process"
    }

    try:
        # This would process the content from the extraction job
        # For now, we'll just return a placeholder response
        # In a real implementation: chunks = await process_extraction_job(request.job_id, request.chunk_size, request.overlap)

        # Placeholder: assume we'll create some chunks
        total_chunks = 10  # This would be calculated from the actual content

        jobs[job_id]["total_chunks"] = total_chunks

        return ProcessResponse(
            job_id=job_id,
            status="processing",
            total_chunks=total_chunks
        )
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store", response_model=StoreResponse)
async def store_embeddings(request: StoreRequest):
    """Store embeddings in Qdrant vector database"""
    job_id = f"store_{uuid.uuid4().hex[:8]}"

    # Initialize job tracking
    jobs[job_id] = {
        "status": "uploading",
        "progress": 0.0,
        "created_at": datetime.now(),
        "type": "store"
    }

    try:
        # Create collection if it doesn't exist
        await create_collection(request.collection_name)

        # In a real implementation, this would get embeddings from the process job
        # and save them to Qdrant
        # total_embeddings = await save_embeddings_to_qdrant(request.job_id, request.collection_name)

        # Placeholder: assume we'll store some embeddings
        total_embeddings = 10  # This would be from the actual processed data

        jobs[job_id]["total_embeddings"] = total_embeddings

        return StoreResponse(
            job_id=job_id,
            status="uploading",
            total_embeddings=total_embeddings
        )
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_content(request: SearchRequest):
    """Semantic search in the vector database"""
    try:
        # Use the vector DB service to perform the search
        vector_db_service = VectorDBService()
        results = await vector_db_service.search(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            min_score=request.min_score
        )

        return SearchResponse(
            query=request.query,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate", response_model=ValidateResponse)
async def validate_embeddings(request: ValidateRequest):
    """Validate stored embeddings"""
    try:
        vector_db_service = VectorDBService()
        validation_result = await vector_db_service.validate_collection(request.collection_name)

        return validation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        details=job.get("details")
    )


# Include retrieval and validation routes
app.include_router(retrieval_router, prefix="/api", tags=["retrieval-validation"])

# Include agent routes (for the /query endpoint)
try:
    from agent.api import router as agent_router
    app.include_router(agent_router, prefix="/api/v1", tags=["agent"])
    print("Agent API loaded successfully")
except Exception as e:
    print(f"Agent API not available - skipping agent routes: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)