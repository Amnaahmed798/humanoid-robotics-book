"""
Data models for retrieval and validation of Qdrant embeddings.

These models represent the entities defined in the data model specification
for the retrieval and validation system.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RetrievedChunk(BaseModel):
    """
    Represents a chunk of text retrieved from Qdrant based on semantic similarity.
    """
    id: str = Field(..., description="Unique identifier for the retrieved chunk")
    text: str = Field(..., description="The actual text content retrieved from the book")
    source_location: str = Field(
        ...,
        description="Reference to the original location in the book (e.g., chapter, section)"
    )
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score between query and chunk",
        ge=0.0,
        le=1.0
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the retrieval context"
    )


class Query(BaseModel):
    """
    Represents a query executed against the Qdrant collection.
    """
    id: str = Field(..., description="Unique identifier for the query")
    text: str = Field(..., description="The input query text")
    top_k: int = Field(
        default=3,
        description="Number of results requested (default: 3)",
        gt=0
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="Minimum similarity threshold for results",
        ge=0.0,
        le=1.0
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the query was executed"
    )


class ValidationResult(BaseModel):
    """
    Represents the validation result for a retrieved chunk.
    """
    query_id: str = Field(..., description="Reference to the associated query")
    chunk_id: str = Field(..., description="Reference to the retrieved chunk")
    accuracy_score: float = Field(
        ...,
        description="Measure of how well the chunk matches expected content",
        ge=0.0,
        le=1.0
    )
    original_text: str = Field(..., description="The original book text for comparison")
    retrieved_text: str = Field(..., description="The text retrieved from Qdrant")
    validation_passed: bool = Field(
        ...,
        description="Whether the validation criteria were met (accuracy >= 0.95)"
    )
    validation_details: str = Field(
        ...,
        description="Explanation of validation results"
    )


class ValidationLog(BaseModel):
    """
    Represents a comprehensive log of a validation run.
    """
    id: str = Field(..., description="Unique identifier for the validation log entry")
    query: str = Field(..., description="The query text that was executed")
    results: List[RetrievedChunk] = Field(
        ...,
        description="The chunks retrieved by the query"
    )
    validations: List[ValidationResult] = Field(
        ...,
        description="Validation results for each chunk"
    )
    overall_accuracy: float = Field(
        ...,
        description="Overall accuracy percentage for the query",
        ge=0.0,
        le=1.0
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the validation was performed"
    )
    book_section: str = Field(
        ...,
        description="Which book section was tested"
    )


class QueryResult(BaseModel):
    """
    Represents the result of executing a single query with validation.
    """
    query_id: str = Field(..., description="Unique identifier for this query")
    query: str = Field(..., description="The original query text")
    results: List[RetrievedChunk] = Field(
        ...,
        description="The retrieved chunks from Qdrant"
    )
    validations: List[ValidationResult] = Field(
        ...,
        description="Validation results for each retrieved chunk"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the query was executed"
    )


class ValidationRunRequest(BaseModel):
    """
    Request model for batch validation runs.
    """
    queries: List[str] = Field(
        ...,
        description="Sample queries to execute against the Qdrant collection"
    )
    top_k: int = Field(
        default=3,
        description="Number of results to retrieve for each query",
        gt=0
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="Minimum similarity threshold for results",
        ge=0.0,
        le=1.0
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Name of the Qdrant collection to validate"
    )
    book_sections: List[str] = Field(
        default=[],
        description="Specific book sections to test against"
    )


class ValidationRunResponse(BaseModel):
    """
    Response model for batch validation runs.
    """
    validation_id: str = Field(..., description="Unique identifier for this validation run")
    total_queries: int = Field(..., description="Number of queries executed")
    total_sections_tested: int = Field(..., description="Number of book sections tested")
    overall_accuracy: float = Field(
        ...,
        description="Overall accuracy percentage across all queries",
        ge=0.0,
        le=1.0
    )
    results: List[ValidationResult] = Field(
        ...,
        description="Individual validation results"
    )
    logs: List[ValidationLog] = Field(
        ...,
        description="Detailed validation logs"
    )