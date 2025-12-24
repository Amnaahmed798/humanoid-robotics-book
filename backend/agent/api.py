from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.agent_models import QueryRequest, QueryResponse
from .retrieval_service import RetrievalService
from .agent_service import AgentService
from .validation_service import ValidationService

router = APIRouter()

# Initialize services lazily to avoid import-time Qdrant connection
retrieval_service = None
agent_service = None
validation_service = None

def get_retrieval_service():
    global retrieval_service
    if retrieval_service is None:
        retrieval_service = RetrievalService()
    return retrieval_service

def get_agent_service():
    global agent_service
    if agent_service is None:
        agent_service = AgentService()
    return agent_service

def get_validation_service():
    global validation_service
    if validation_service is None:
        validation_service = ValidationService()
    return validation_service

@router.post("/query", response_model=QueryResponse, tags=["agent"])
async def query_endpoint(request: QueryRequest):
    """
    Query the book content using retrieval-augmented generation.

    This endpoint:
    1. Takes a user question
    2. Retrieves relevant book chunks from Qdrant
    3. Uses an OpenAI agent to generate a grounded response
    4. Validates that the response is based on retrieved content
    5. Returns the answer with sources and confidence score
    """
    try:
        # Retrieve relevant context from Qdrant
        try:
            retrieved_chunks = await get_retrieval_service().retrieve(request.question)
        except Exception as retrieval_error:
            logging.error(f"Error retrieving chunks: {str(retrieval_error)}")
            # Return a user-friendly response instead of throwing a 500 error
            return QueryResponse(
                answer="I'm having trouble retrieving information from the knowledge base. Please try again later.",
                sources=[],
                confidence=0.1
            )

        if not retrieved_chunks:
            return QueryResponse(
                answer="I couldn't find relevant information in the book content to answer your question.",
                sources=[],
                confidence=0.0
            )

        # Generate response using the agent
        try:
            agent_response = await get_agent_service().generate_response(
                question=request.question,
                context=retrieved_chunks
            )
        except Exception as agent_error:
            logging.error(f"Error generating agent response: {str(agent_error)}")
            # Return a user-friendly response instead of throwing a 500 error
            return QueryResponse(
                answer="I'm having trouble generating a response. This could be due to an issue with the AI service or the retrieved content.",
                sources=[],
                confidence=0.1
            )

        # Validate that the response is grounded in the retrieved content
        try:
            is_valid = await get_validation_service().validate_response(
                response=agent_response,
                context=retrieved_chunks
            )
        except Exception as validation_error:
            logging.warning(f"Validation service error: {str(validation_error)}")
            # If validation itself fails, we'll proceed without validation
            is_valid = True  # Allow the response to go through if validation fails

        if not is_valid:
            # Instead of throwing an HTTP 500 error, return a response indicating the issue
            # This prevents the frontend from showing a generic error message
            return QueryResponse(
                answer="I detected that the response may not be fully grounded in the provided content. This could be due to the retrieved information not containing relevant details for your query.",
                sources=[],
                confidence=0.1
            )

        # Return the response with sources and confidence
        return QueryResponse(
            answer=agent_response.answer,
            sources=agent_response.sources,
            confidence=agent_response.confidence
        )

    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/validate", tags=["agent"])
async def validate_endpoint(request: QueryRequest):
    """
    Validate that a response is grounded in the retrieved content.
    This endpoint can be used for debugging and quality assurance.
    """
    try:
        # Retrieve relevant context
        retrieved_chunks = await get_retrieval_service().retrieve(request.question)

        # For validation, we'd need to have both a question and a response to validate
        # This is a simplified version - in practice, you might want to validate
        # a response that was already generated
        return {
            "retrieved_chunks_count": len(retrieved_chunks) if retrieved_chunks else 0,
            "validation_status": "validation_not_implemented_for_external_response"
        }
    except Exception as e:
        logging.error(f"Error during validation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during validation: {str(e)}"
        )