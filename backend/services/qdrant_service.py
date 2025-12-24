"""
Qdrant connection service for retrieving embeddings.

This service handles the connection to Qdrant and provides methods for
retrieving embeddings based on semantic similarity queries.
"""
from qdrant_client.http import models
from typing import List, Optional
import os
from dotenv import load_dotenv
from .qdrant_client_manager import get_qdrant_client

# Load environment variables
load_dotenv()

class QdrantService:
    """
    Service class for interacting with Qdrant vector database.
    """

    def __init__(self):
        """
        Initialize the Qdrant service with configuration from environment variables.
        The actual client connection is managed centrally to avoid concurrent access issues.
        """
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")

    @property
    def client(self):
        """
        Get the shared Qdrant client instance from the centralized manager.
        """
        return get_qdrant_client()

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 3,
        similarity_threshold: float = 0.3
    ) -> List[models.ScoredPoint]:
        """
        Search for similar vectors in the specified collection.

        Args:
            collection_name: Name of the Qdrant collection to search
            query_vector: The vector to search for similarity
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score threshold

        Returns:
            List of ScoredPoint objects containing the search results
        """
        try:
            # Perform the search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=similarity_threshold
            )

            return results
        except Exception as e:
            print(f"Error searching in Qdrant: {e}")
            raise

    def get_collections(self) -> List[models.CollectionInfo]:
        """
        Get list of all collections in the Qdrant instance.

        Returns:
            List of CollectionInfo objects
        """
        try:
            collections = self.client.get_collections()
            return collections.collections
        except Exception as e:
            print(f"Error getting collections from Qdrant: {e}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Qdrant.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.get_collections()
            return any(collection.name == collection_name for collection in collections)
        except Exception:
            return False

    def get_collection_info(self, collection_name: str) -> Optional[models.CollectionInfo]:
        """
        Get information about a specific collection.

        Args:
            collection_name: Name of the collection to get info for

        Returns:
            CollectionInfo object if collection exists, None otherwise
        """
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return info
        except Exception:
            return None