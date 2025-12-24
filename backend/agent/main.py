from fastapi import FastAPI
from .api import router as api_router
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(
    title="Retrieval-Enabled Agent API",
    description="An API that uses Qdrant embeddings to provide grounded answers from book content using OpenAI agent",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["agent"])

@app.get("/")
async def root():
    return {"message": "Retrieval-Enabled Agent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)