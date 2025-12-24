"""
Embedding generation module for the RAG chatbot backend.
This module provides functions to generate embeddings using Cohere.
"""
import asyncio
import os
from typing import List
from models.embedding import EmbeddingVector
from models.chunk import TextChunk
import cohere
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Cohere client
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY environment variable is required")

co = cohere.Client(api_key)
model_name = "embed-multilingual-v3.0"  # As decided in research


async def embed(text: str) -> EmbeddingVector:
    """
    Generate embedding for a single text.

    Args:
        text: Text to generate embedding for

    Returns:
        EmbeddingVector object
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_sync, text)


def _embed_sync(text: str) -> EmbeddingVector:
    """
    Synchronous version of embed to run in thread pool.
    """
    try:
        response = co.embed(
            texts=[text],
            model=model_name,
            input_type="search_document"  # Using search_document as this is for RAG
        )

        embedding_data = response.embeddings[0]

        embedding_vector = EmbeddingVector(
            vector=embedding_data,
            chunk_id=None,  # Will be set when associated with a chunk
            model_name=model_name
        )

        return embedding_vector

    except Exception as e:
        logger.error(f"Error generating embedding for text: {str(e)}")
        raise


async def embed_texts(texts: List[str]) -> List[EmbeddingVector]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to generate embeddings for

    Returns:
        List of EmbeddingVector objects
    """
    if not texts:
        return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_texts_sync, texts)


def _embed_texts_sync(texts: List[str]) -> List[EmbeddingVector]:
    """
    Synchronous version of embed_texts to run in thread pool.
    """
    try:
        # Cohere has a limit on the number of texts per request (typically 96)
        # For now, we'll process all at once, but in production we might need batching
        response = co.embed(
            texts=texts,
            model=model_name,
            input_type="search_document"
        )

        embeddings = []
        for i, embedding_data in enumerate(response.embeddings):
            embedding_vector = EmbeddingVector(
                vector=embedding_data,
                chunk_id=f"text_{i}",  # Temporary ID, should be set properly when used
                model_name=model_name
            )
            embeddings.append(embedding_vector)

        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings for texts: {str(e)}")
        raise


async def embed_chunks(chunks: List[TextChunk]) -> List[EmbeddingVector]:
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks: List of TextChunk objects

    Returns:
        List of EmbeddingVector objects with proper chunk_id associations
    """
    if not chunks:
        return []

    # Extract the text from each chunk
    texts = [chunk.text for chunk in chunks]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_chunks_sync, chunks, texts)


def _embed_chunks_sync(chunks: List[TextChunk], texts: List[str]) -> List[EmbeddingVector]:
    """
    Synchronous version of embed_chunks to run in thread pool.
    """
    try:
        # Generate embeddings for all texts at once
        response = co.embed(
            texts=texts,
            model=model_name,
            input_type="search_document"
        )

        embeddings = []
        for i, embedding_data in enumerate(response.embeddings):
            chunk = chunks[i]
            embedding_vector = EmbeddingVector(
                id=f"emb_{chunk.id}" if chunk.id else f"emb_{i}",
                vector=embedding_data,
                chunk_id=chunk.id or f"chunk_{i}",
                model_name=model_name
            )
            embeddings.append(embedding_vector)

        return embeddings

    except Exception as e:
        logger.error(f"Error generating embeddings for chunks: {str(e)}")
        raise


async def embed_query(query: str) -> EmbeddingVector:
    """
    Generate embedding for a search query.

    Args:
        query: Query text to generate embedding for

    Returns:
        EmbeddingVector object
    """
    # For search queries, we use a different input type
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed_query_sync, query)


def _embed_query_sync(query: str) -> EmbeddingVector:
    """
    Synchronous version of embed_query to run in thread pool.
    """
    try:
        response = co.embed(
            texts=[query],
            model=model_name,
            input_type="search_query"  # Using search_query for queries
        )

        embedding_data = response.embeddings[0]

        embedding_vector = EmbeddingVector(
            vector=embedding_data,
            chunk_id=None,  # Queries don't have a chunk ID
            model_name=model_name
        )

        return embedding_vector

    except Exception as e:
        logger.error(f"Error generating embedding for query: {str(e)}")
        raise


def validate_embedding_vector(embedding: EmbeddingVector) -> bool:
    """
    Validate that an embedding meets requirements.

    Args:
        embedding: EmbeddingVector to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check dimensions
        if len(embedding.vector) != 1024:
            logger.warning(f"Embedding has {len(embedding.vector)} dimensions, expected 1024")
            return False

        # Check that all values are valid numbers
        if not all(isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val or abs(val) == float('inf'))) for val in embedding.vector):
            logger.warning("Embedding contains invalid values (NaN or infinity)")
            return False

        # Check model name
        if not embedding.model_name or embedding.model_name != model_name:
            logger.warning(f"Embedding model name {embedding.model_name} doesn't match expected {model_name}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating embedding: {str(e)}")
        return False


def get_embedding_model_info() -> dict:
    """
    Get information about the embedding model being used.

    Returns:
        Dictionary with model information
    """
    return {
        "model_name": model_name,
        "dimensions": 1024,
        "description": "Cohere multilingual embedding model v3.0, optimized for multi-lingual content and RAG applications"
    }


# Example usage if run as a script
if __name__ == "__main__":
    async def example():
        sample_text = "This is a sample text to generate an embedding for."

        # Generate embedding for single text
        embedding = await embed(sample_text)
        print(f"Generated embedding with {len(embedding.vector)} dimensions")

        # Generate embeddings for multiple texts
        texts = [
            "First text for embedding",
            "Second text for embedding",
            "Third text for embedding"
        ]
        embeddings = await embed_texts(texts)
        print(f"Generated {len(embeddings)} embeddings")

        # Validate an embedding
        is_valid = validate_embedding_vector(embedding)
        print(f"Is embedding valid? {is_valid}")

    # Run the example
    # asyncio.run(example())