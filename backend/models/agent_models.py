from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from .retrieval_models import RetrievedChunk


class QueryRequest(BaseModel):
    """
    Request model for querying the book content.
    """
    question: str
    top_k: Optional[int] = 3  # Number of chunks to retrieve
    temperature: Optional[float] = 0.1  # Temperature for response generation


class Source(BaseModel):
    """
    Model representing a source document/chunk.
    """
    id: str
    content: str
    score: float  # Similarity score
    metadata: dict  # Additional metadata about the source


class QueryResponse(BaseModel):
    """
    Response model for query results.
    """
    answer: str
    sources: List[RetrievedChunk]  # Using the existing RetrievedChunk model
    confidence: float  # Confidence score between 0 and 1
    timestamp: datetime = datetime.now()


class AgentResponse(BaseModel):
    """
    Model for the agent's response with additional metadata.
    """
    answer: str
    sources: List[RetrievedChunk]  # Using the existing RetrievedChunk model
    confidence: float
    raw_response: Optional[str] = None  # Raw response from the LLM