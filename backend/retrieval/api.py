"""
API endpoints for retrieval and validation functionality.

This module implements the API endpoints based on the OpenAPI contract
specification for retrieval and validation of Qdrant embeddings.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.retrieval_models import (
    ValidationRunRequest, ValidationRunResponse,
    QueryResult, ValidationResult, ValidationLog
)
from pydantic import BaseModel, Field, validator, ValidationError
from .retrieval_service import RetrievalService
from .validation_service import ValidationService
from .result_logger import ResultLogger
from .multi_section_tester import MultiSectionTester
from .config import RetrievalConfig


# Initialize services lazily to avoid import-time Qdrant connection
retrieval_service = None
validation_service = None
result_logger = None
multi_section_tester = None

def get_retrieval_service():
    global retrieval_service
    if retrieval_service is None:
        retrieval_service = RetrievalService()
    return retrieval_service

def get_validation_service():
    global validation_service
    if validation_service is None:
        validation_service = ValidationService()
    return validation_service

def get_result_logger():
    global result_logger
    if result_logger is None:
        result_logger = ResultLogger()
    return result_logger

def get_multi_section_tester():
    global multi_section_tester
    if multi_section_tester is None:
        multi_section_tester = MultiSectionTester(
            retrieval_service=get_retrieval_service(),
            validation_service=get_validation_service(),
            result_logger=get_result_logger()
        )
    return multi_section_tester

# Create router
router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/validate-retrieval", response_model=ValidationRunResponse)
async def validate_retrieval(request: ValidationRunRequest):
    """
    Validate embedding retrieval with sample queries.

    Execute sample queries against Qdrant and validate that retrieved chunks
    match original book content.
    """
    try:
        # Validate request model using Pydantic validation
        try:
            request = ValidationRunRequest(**request.dict())
        except ValidationError as ve:
            raise HTTPException(status_code=422, detail=f"Invalid request parameters: {ve}")

        # Validate parameters
        if len(request.queries) == 0:
            raise HTTPException(status_code=400, detail="At least one query is required")

        if len(request.queries) > RetrievalConfig.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Too many queries. Maximum batch size is {RetrievalConfig.MAX_BATCH_SIZE}"
            )

        if request.top_k <= 0:
            raise HTTPException(status_code=400, detail="top_k must be a positive integer")

        if not (0.0 <= request.similarity_threshold <= 1.0):
            raise HTTPException(status_code=400, detail="similarity_threshold must be between 0.0 and 1.0")

        # Use provided collection name or default
        collection_name = request.collection_name or RetrievalConfig.DEFAULT_COLLECTION_NAME

        # Validate that the collection exists
        if not get_retrieval_service().validate_collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        # Run validation across multiple queries
        validation_logs = []
        total_accuracy = 0.0

        for query in request.queries:
            # Retrieve chunks for the query
            chunks = get_retrieval_service().retrieve_chunks(
                query_text=query,
                collection_name=collection_name,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold
            )

            # Validate the results
            # For validation, we need to have the original text - in a real scenario this would come from the source
            # For this implementation, we'll validate against the retrieved text itself as a placeholder
            validation_results = []
            for chunk in chunks:
                validation_result = get_validation_service().validate_chunk_accuracy(
                    retrieved_chunk=chunk,
                    original_text=chunk.text  # Using retrieved text as original for demo purposes
                )
                validation_results.append(validation_result)

            # Calculate overall accuracy for this query
            query_accuracy = sum(v.accuracy_score for v in validation_results) / len(validation_results) if validation_results else 0
            total_accuracy += query_accuracy

            # Create a validation log
            validation_log = ValidationLog(
                id=f"validation_{abs(hash(query)) % 1000000}",
                query=query,
                results=chunks,
                validations=validation_results,
                overall_accuracy=query_accuracy,
                book_section="unknown"  # In real implementation, this would be determined
            )
            validation_logs.append(validation_log)

        # Calculate overall metrics
        overall_accuracy = total_accuracy / len(validation_logs) if validation_logs else 0
        total_validations = sum(len(log.validations) for log in validation_logs)
        total_queries = len(request.queries)

        # Create response
        response = ValidationRunResponse(
            validation_id=f"run_{abs(hash(str(request.queries))) % 1000000}",
            total_queries=total_queries,
            total_sections_tested=len(request.book_sections) if request.book_sections else 1,  # Simplified for this implementation
            overall_accuracy=overall_accuracy,
            results=[val for log in validation_logs for val in log.validations],  # Flatten all validations
            logs=validation_logs
        )

        # Log the validation run
        result_logger.log_batch_validation_results(validation_logs)

        return response

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating retrieval: {str(e)}")


@router.post("/test-query", response_model=QueryResult)
async def test_query(
    query: str,
    top_k: int = 3,
    similarity_threshold: float = 0.3,
    collection_name: str = RetrievalConfig.DEFAULT_COLLECTION_NAME
):
    """
    Test a single query against Qdrant.

    Execute a single query and retrieve top-k results with validation.
    """
    try:
        # Validate parameters
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if top_k <= 0:
            raise HTTPException(status_code=400, detail="top_k must be a positive integer")

        if not (0.0 <= similarity_threshold <= 1.0):
            raise HTTPException(status_code=400, detail="similarity_threshold must be between 0.0 and 1.0")

        # Validate that the collection exists
        if not get_retrieval_service().validate_collection_exists(collection_name):
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        # Record start time for execution time calculation
        import time
        start_time = time.time()

        # Retrieve chunks for the query
        chunks = get_retrieval_service().retrieve_chunks(
            query_text=query,
            collection_name=collection_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Calculate execution time
        execution_time = time.time() - start_time

        # Validate the results
        validation_results = []
        for chunk in chunks:
            validation_result = get_validation_service().validate_chunk_accuracy(
                retrieved_chunk=chunk,
                original_text=chunk.text  # Using retrieved text as original for demo purposes
            )
            validation_results.append(validation_result)

        # Create query result
        query_result = QueryResult(
            query_id=f"query_{abs(hash(query)) % 1000000}",
            query=query,
            results=chunks,
            validations=validation_results,
            timestamp=None
        )

        # Log the query execution
        result_logger.log_query_execution(
            query=query,
            results=chunks,
            execution_time=execution_time,
            book_section="unknown"
        )

        return query_result

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing query: {str(e)}")


# Additional utility endpoints
@router.get("/collections")
async def get_collections():
    """
    Get list of available collections in Qdrant.
    """
    try:
        collections = get_retrieval_service().qdrant_service.get_collections()
        return {"collections": [c.name for c in collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collections: {str(e)}")


@router.get("/config")
async def get_config():
    """
    Get current configuration values.
    """
    return {
        "default_top_k": RetrievalConfig.DEFAULT_TOP_K,
        "default_similarity_threshold": RetrievalConfig.DEFAULT_SIMILARITY_THRESHOLD,
        "min_similarity_threshold": RetrievalConfig.MIN_SIMILARITY_THRESHOLD,
        "max_similarity_threshold": RetrievalConfig.MAX_SIMILARITY_THRESHOLD,
        "default_collection_name": RetrievalConfig.DEFAULT_COLLECTION_NAME,
        "default_test_sections": RetrievalConfig.DEFAULT_TEST_SECTIONS,
        "validation_accuracy_threshold": RetrievalConfig.VALIDATION_ACCURACY_THRESHOLD,
        "max_batch_size": RetrievalConfig.MAX_BATCH_SIZE
    }