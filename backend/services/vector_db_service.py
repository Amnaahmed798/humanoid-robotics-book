from typing import List, Dict, Any, Optional
from models.embedding import EmbeddingVector, QdrantCollection
from models.chunk import TextChunk, SearchChunk
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
from .qdrant_client_manager import get_qdrant_client

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class VectorDBService:
    """
    Service for vector database operations using Qdrant.
    """

    def __init__(self):
        # Initialize Qdrant client from centralized manager to avoid multiple connections
        self.client = get_qdrant_client()

        self.vector_size = 1024  # For Cohere multilingual model
        self.distance = Distance.COSINE

    async def create_collection(self, collection_name: str) -> bool:
        """
        Create a Qdrant collection with the specified name.

        Args:
            collection_name: Name of the collection to create

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if collection already exists
            try:
                self.client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists")
                return True
            except:
                # Collection doesn't exist, so create it
                pass

            # Create the collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                )
            )

            logger.info(f"Created collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {str(e)}")
            return False

    async def save_embedding(self, embedding: EmbeddingVector, collection_name: str, metadata: Optional[Dict] = None) -> bool:
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
            # Ensure the collection exists
            await self.create_collection(collection_name)

            # Prepare the point to insert
            point = PointStruct(
                id=embedding.id or embedding.chunk_id,  # Use chunk_id as fallback for ID
                vector=embedding.vector,
                payload=metadata or {}
            )

            # Insert the point
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )

            logger.info(f"Saved embedding {embedding.id or embedding.chunk_id} to collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving embedding to collection {collection_name}: {str(e)}")
            return False

    async def save_embeddings(self, embeddings: List[EmbeddingVector], collection_name: str, metadatas: Optional[List[Dict]] = None) -> bool:
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
            # Ensure the collection exists
            await self.create_collection(collection_name)

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
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"Saved {len(embeddings)} embeddings to collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving {len(embeddings)} embeddings to collection {collection_name}: {str(e)}")
            return False

    async def search(self, query: str, collection_name: str, top_k: int = 5, min_score: float = 0.3) -> List[SearchChunk]:
        """
        Perform a semantic search in the specified collection.

        Args:
            query: Query text to search for
            collection_name: Name of the collection to search in
            top_k: Number of results to return (default 5)
            min_score: Minimum similarity score (0.0 to 1.0, default 0.3)

        Returns:
            List of SearchChunk objects with the search results
        """
        try:
            # Generate embedding for the query
            from services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService()
            query_embedding = await embedding_service.embed_text(query)

            # Perform the search
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding.vector,
                limit=top_k,
                score_threshold=min_score
            )

            # Convert results to SearchChunk objects
            results = []
            for hit in search_results:
                result = SearchChunk(
                    id=hit.id,
                    text=hit.payload.get('text', ''),
                    score=hit.score,
                    metadata=hit.payload
                )
                results.append(result)

            logger.info(f"Search in {collection_name} returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching in collection {collection_name}: {str(e)}")
            raise

    async def get_collection_info(self, collection_name: str) -> Optional[QdrantCollection]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            QdrantCollection object with collection info, or None if collection doesn't exist
        """
        try:
            collection_info = self.client.get_collection(collection_name)

            qdrant_collection = QdrantCollection(
                name=collection_name,
                vector_size=self.vector_size,
                distance_metric=str(self.distance),
                point_count=collection_info.points_count
            )

            return qdrant_collection

        except Exception as e:
            logger.error(f"Error getting info for collection {collection_name}: {str(e)}")
            return None

    async def validate_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Validate that a collection has the expected properties.

        Args:
            collection_name: Name of the collection to validate

        Returns:
            Dictionary with validation results
        """
        try:
            # Get collection info
            collection_info = await self.get_collection_info(collection_name)

            if not collection_info:
                return {
                    "collection_name": collection_name,
                    "vector_count": 0,
                    "expected_count": 0,
                    "vector_dimensions": self.vector_size,
                    "model_name": "embed-multilingual-v3.0",
                    "is_valid": False,
                    "issues": [f"Collection {collection_name} does not exist"]
                }

            # Additional validation: check if vectors have correct dimensions
            # Get a sample of vectors to check dimensions
            sample_result = self.client.scroll(
                collection_name=collection_name,
                limit=1
            )

            issues = []
            if sample_result[0] is not None:  # If there are any points in the collection
                sample_point = sample_result[0]
                if len(sample_point.vector) != self.vector_size:
                    issues.append(f"Vector dimensions mismatch: expected {self.vector_size}, got {len(sample_point.vector)}")

            is_valid = len(issues) == 0

            return {
                "collection_name": collection_name,
                "vector_count": collection_info.point_count,
                "expected_count": collection_info.point_count,  # In this case, expected = actual
                "vector_dimensions": self.vector_size,
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
                "vector_dimensions": self.vector_size,
                "model_name": "embed-multilingual-v3.0",
                "is_valid": False,
                "issues": [str(e)]
            }

    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False

    async def count_vectors(self, collection_name: str) -> int:
        """
        Count the number of vectors in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of vectors in the collection
        """
        try:
            collection_info = await self.get_collection_info(collection_name)
            return collection_info.point_count if collection_info else 0
        except Exception as e:
            logger.error(f"Error counting vectors in collection {collection_name}: {str(e)}")
            return 0