"""
Qdrant vector database operations module for the RAG chatbot backend.
This module provides functions to interact with Qdrant Cloud for storing and retrieving embeddings.
"""
import asyncio
import os
from typing import List, Dict, Any, Optional
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from models.embedding import EmbeddingVector
from dotenv import load_dotenv
import logging
from services.qdrant_client_manager import get_qdrant_client

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Get centralized Qdrant client to avoid multiple connections
client = get_qdrant_client()

vector_size = 1024  # For Cohere multilingual model
distance = Distance.COSINE


async def create_collection(collection_name: str) -> bool:
    """
    Create a Qdrant collection with the specified name.

    Args:
        collection_name: Name of the collection to create (named rag_embedding as specified)

    Returns:
        True if successful, False otherwise
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _create_collection_sync, collection_name)
    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {str(e)}")
        return False


def _create_collection_sync(collection_name: str) -> bool:
    """
    Synchronous version of create_collection to run in thread pool.
    """
    try:
        # Check if collection already exists
        try:
            client.get_collection(collection_name)
            logger.info(f"Collection {collection_name} already exists")
            return True
        except:
            # Collection doesn't exist, so create it
            pass

        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )

        logger.info(f"Created collection {collection_name}")
        return True

    except Exception as e:
        logger.error(f"Error creating collection {collection_name}: {str(e)}")
        return False


async def save_chunk(embedding: EmbeddingVector, collection_name: str, metadata: Optional[Dict] = None) -> bool:
    """
    Save a single embedding to the specified collection.

    Args:
        embedding: EmbeddingVector to save
        collection_name: Name of the collection to save to
        metadata: Optional metadata to associate with the embedding

    Returns:
        True if successful, False otherwise
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _save_chunk_sync, embedding, collection_name, metadata)
    except Exception as e:
        logger.error(f"Error saving embedding to collection {collection_name}: {str(e)}")
        return False


def _save_chunk_sync(embedding: EmbeddingVector, collection_name: str, metadata: Optional[Dict] = None) -> bool:
    """
    Synchronous version of save_chunk to run in thread pool.
    """
    try:
        # Ensure the collection exists
        _create_collection_sync(collection_name)

        # Prepare the point to insert
        point = PointStruct(
            id=embedding.id or embedding.chunk_id,  # Use chunk_id as fallback for ID
            vector=embedding.vector,
            payload=metadata or {}
        )

        # Insert the point
        client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        logger.info(f"Saved embedding {embedding.id or embedding.chunk_id} to collection {collection_name}")
        return True

    except Exception as e:
        logger.error(f"Error saving embedding to collection {collection_name}: {str(e)}")
        return False


async def save_chunks(embeddings: List[EmbeddingVector], collection_name: str, metadatas: Optional[List[Dict]] = None) -> bool:
    """
    Save multiple embeddings to the specified collection.

    Args:
        embeddings: List of EmbeddingVector objects to save
        collection_name: Name of the collection to save to
        metadatas: Optional list of metadata dictionaries (one per embedding)

    Returns:
        True if successful, False otherwise
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _save_chunks_sync, embeddings, collection_name, metadatas)
    except Exception as e:
        logger.error(f"Error saving embeddings to collection {collection_name}: {str(e)}")
        return False


def _save_chunks_sync(embeddings: List[EmbeddingVector], collection_name: str, metadatas: Optional[List[Dict]] = None) -> bool:
    """
    Synchronous version of save_chunks to run in thread pool.
    """
    try:
        # Ensure the collection exists
        _create_collection_sync(collection_name)

        # Prepare the points to insert
        points = []
        for i, embedding in enumerate(embeddings):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

            point = PointStruct(
                id=embedding.id or embedding.chunk_id or f"emb_{i}",
                vector=embedding.vector,
                payload=metadata
            )
            points.append(point)

        # Insert the points
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Saved {len(embeddings)} embeddings to collection {collection_name}")
        return True

    except Exception as e:
        logger.error(f"Error saving {len(embeddings)} embeddings to collection {collection_name}: {str(e)}")
        return False


async def search_embeddings(query_vector: List[float], collection_name: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
    """
    Perform a semantic search in the specified collection.

    Args:
        query_vector: Query embedding vector to search for
        collection_name: Name of the collection to search in
        top_k: Number of results to return (default 5)
        min_score: Minimum similarity score (0.0 to 1.0, default 0.3)

    Returns:
        List of result dictionaries with id, score, and payload
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search_embeddings_sync, query_vector, collection_name, top_k, min_score)
    except Exception as e:
        logger.error(f"Error searching in collection {collection_name}: {str(e)}")
        raise


def _search_embeddings_sync(query_vector: List[float], collection_name: str, top_k: int = 5, min_score: float = 0.3) -> List[Dict[str, Any]]:
    """
    Synchronous version of search_embeddings to run in thread pool.
    """
    try:
        # Perform the search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=min_score
        )

        # Convert results to dictionary format
        results = []
        for hit in search_results:
            result = {
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload
            }
            results.append(result)

        logger.info(f"Search in {collection_name} returned {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Error searching in collection {collection_name}: {str(e)}")
        raise


async def get_collection_info(collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Dictionary with collection info, or None if collection doesn't exist
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_collection_info_sync, collection_name)
    except Exception as e:
        logger.error(f"Error getting info for collection {collection_name}: {str(e)}")
        return None


def _get_collection_info_sync(collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Synchronous version of get_collection_info to run in thread pool.
    """
    try:
        collection_info = client.get_collection(collection_name)

        return {
            "name": collection_name,
            "vector_size": vector_size,
            "distance_metric": str(distance),
            "point_count": collection_info.points_count
        }

    except Exception as e:
        logger.error(f"Error getting info for collection {collection_name}: {str(e)}")
        return None


async def validate_collection(collection_name: str) -> Dict[str, Any]:
    """
    Validate that a collection has the expected properties.

    Args:
        collection_name: Name of the collection to validate

    Returns:
        Dictionary with validation results
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _validate_collection_sync, collection_name)
    except Exception as e:
        logger.error(f"Error validating collection {collection_name}: {str(e)}")
        return {
            "collection_name": collection_name,
            "vector_count": 0,
            "expected_count": 0,
            "vector_dimensions": vector_size,
            "model_name": "embed-multilingual-v3.0",
            "is_valid": False,
            "issues": [str(e)]
        }


def _validate_collection_sync(collection_name: str) -> Dict[str, Any]:
    """
    Synchronous version of validate_collection to run in thread pool.
    """
    try:
        # Get collection info
        collection_info = _get_collection_info_sync(collection_name)

        if not collection_info:
            return {
                "collection_name": collection_name,
                "vector_count": 0,
                "expected_count": 0,
                "vector_dimensions": vector_size,
                "model_name": "embed-multilingual-v3.0",
                "is_valid": False,
                "issues": [f"Collection {collection_name} does not exist"]
            }

        # Additional validation: check if vectors have correct dimensions
        # Get a sample of vectors to check dimensions
        sample_result = client.scroll(
            collection_name=collection_name,
            limit=1
        )

        issues = []
        if sample_result[0] is not None:  # If there are any points in the collection
            sample_point = sample_result[0]
            if len(sample_point.vector) != vector_size:
                issues.append(f"Vector dimensions mismatch: expected {vector_size}, got {len(sample_point.vector)}")

        is_valid = len(issues) == 0

        return {
            "collection_name": collection_name,
            "vector_count": collection_info["point_count"],
            "expected_count": collection_info["point_count"],  # In this case, expected = actual
            "vector_dimensions": vector_size,
            "model_name": "embed-multilingual-v3.0",
            "is_valid": is_valid,
            "issues": issues
        }

    except Exception as e:
        logger.error(f"Error validating collection {collection_name}: {str(e)}")
        return {
            "collection_name": collection_name,
            "vector_count": 0,
            "expected_count": 0,
            "vector_dimensions": vector_size,
            "model_name": "embed-multilingual-v3.0",
            "is_valid": False,
            "issues": [str(e)]
        }


async def delete_collection(collection_name: str) -> bool:
    """
    Delete a collection.

    Args:
        collection_name: Name of the collection to delete

    Returns:
        True if successful, False otherwise
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _delete_collection_sync, collection_name)
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {str(e)}")
        return False


def _delete_collection_sync(collection_name: str) -> bool:
    """
    Synchronous version of delete_collection to run in thread pool.
    """
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection {collection_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting collection {collection_name}: {str(e)}")
        return False


async def count_vectors(collection_name: str) -> int:
    """
    Count the number of vectors in a collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Number of vectors in the collection
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _count_vectors_sync, collection_name)
    except Exception as e:
        logger.error(f"Error counting vectors in collection {collection_name}: {str(e)}")
        return 0


def _count_vectors_sync(collection_name: str) -> int:
    """
    Synchronous version of count_vectors to run in thread pool.
    """
    try:
        collection_info = _get_collection_info_sync(collection_name)
        return collection_info["point_count"] if collection_info else 0
    except Exception as e:
        logger.error(f"Error counting vectors in collection {collection_name}: {str(e)}")
        return 0


# Example usage if run as a script
if __name__ == "__main__":
    async def example():
        # Example of how to use these functions
        collection_name = "rag_embedding"  # As specified in the requirements

        # Create collection
        success = await create_collection(collection_name)
        print(f"Collection created: {success}")

        # Get collection info
        info = await get_collection_info(collection_name)
        print(f"Collection info: {info}")

    # Run the example
    # asyncio.run(example())