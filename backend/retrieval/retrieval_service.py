"""
Retrieval service for Qdrant embeddings.

This service handles the retrieval of text chunks from Qdrant based on
semantic similarity to input queries.
"""
from typing import List, Optional
import cohere
from qdrant_client.http import models
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from services.qdrant_service import QdrantService
from models.retrieval_models import RetrievedChunk, Query
from .config import RetrievalConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RetrievalService:
    """
    Service class for retrieving embeddings from Qdrant.
    """

    def __init__(self):
        """
        Initialize the retrieval service with Qdrant and Cohere clients.
        """
        self.qdrant_service = QdrantService()

        # Initialize Cohere client
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        self.cohere_client = cohere.Client(cohere_api_key)

    def retrieve_chunks(
        self,
        query_text: str,
        collection_name: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve text chunks from Qdrant based on semantic similarity to the query.

        Args:
            query_text: The query text to search for
            collection_name: Name of the Qdrant collection to search in
            top_k: Number of results to retrieve (uses default if None)
            similarity_threshold: Minimum similarity threshold (uses default if None)

        Returns:
            List of RetrievedChunk objects containing the search results
        """
        # Validate and set default parameters
        top_k = RetrievalConfig.get_top_k(top_k)
        similarity_threshold = RetrievalConfig.get_similarity_threshold(similarity_threshold)

        try:
            # Generate embedding for the query using Cohere
            response = self.cohere_client.embed(
                texts=[query_text],
                model="embed-multilingual-v3.0",  # Using multilingual model for book content
                input_type="search_query"
            )

            query_embedding = response.embeddings[0]

            # Search in Qdrant
            results = self.qdrant_service.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )

            # Convert Qdrant results to RetrievedChunk objects
            # Results from Qdrant are already ranked by similarity score (descending)
            retrieved_chunks = []
            for result in results:
                # Extract text and source location from the payload
                payload = result.payload
                text = payload.get("text", "")
                source_location = payload.get("source_location", "")

                # If source_location is not in payload, try alternative keys
                if not source_location:
                    source_location = payload.get("section", payload.get("location", "unknown"))

                chunk = RetrievedChunk(
                    id=str(result.id),
                    text=text,
                    source_location=source_location,
                    similarity_score=result.score,
                    metadata=payload
                )
                retrieved_chunks.append(chunk)

            return retrieved_chunks

        except ConnectionError as e:
            print(f"Connection error retrieving chunks from Qdrant: {e}")
            raise ConnectionError(f"Could not connect to Qdrant at {self.qdrant_service.qdrant_url}")
        except Exception as e:
            print(f"Error retrieving chunks from Qdrant: {e}")
            raise

    def retrieve_chunks_with_configurable_params(
        self,
        query_text: str,
        collection_name: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[RetrievedChunk]:
        """
        Retrieve text chunks with configurable top_k and similarity threshold parameters.

        Args:
            query_text: The query text to search for
            collection_name: Name of the Qdrant collection to search in
            top_k: Number of results to retrieve (configurable)
            similarity_threshold: Minimum similarity threshold (configurable)

        Returns:
            List of RetrievedChunk objects containing the search results
        """
        return self.retrieve_chunks(
            query_text=query_text,
            collection_name=collection_name,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

    def rank_by_similarity(
        self,
        chunks: List[RetrievedChunk],
        query_text: str
    ) -> List[RetrievedChunk]:
        """
        Re-rank retrieved chunks based on semantic similarity to the query.

        Args:
            chunks: List of RetrievedChunk objects to rank
            query_text: The query text to rank against

        Returns:
            List of RetrievedChunk objects ranked by similarity to the query
        """
        try:
            # Extract all text contents for batch embedding
            texts = [chunk.text for chunk in chunks]

            # Generate embeddings for the chunks using Cohere
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-multilingual-v3.0",
                input_type="search_document"
            )

            chunk_embeddings = response.embeddings

            # Generate embedding for the query using Cohere
            query_response = self.cohere_client.embed(
                texts=[query_text],
                model="embed-multilingual-v3.0",
                input_type="search_query"
            )

            query_embedding = query_response.embeddings[0]

            # Calculate cosine similarity between query and each chunk
            import numpy as np
            similarities = []
            for chunk_emb in chunk_embeddings:
                # Calculate cosine similarity
                dot_product = np.dot(query_embedding, chunk_emb)
                norm_query = np.linalg.norm(query_embedding)
                norm_chunk = np.linalg.norm(chunk_emb)
                if norm_query == 0 or norm_chunk == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_query * norm_chunk)
                similarities.append(similarity)

            # Pair chunks with their similarity scores and sort by similarity (descending)
            chunk_similarity_pairs = list(zip(chunks, similarities))
            chunk_similarity_pairs.sort(key=lambda x: x[1], reverse=True)

            # Return just the ranked chunks
            ranked_chunks = [pair[0] for pair in chunk_similarity_pairs]

            return ranked_chunks

        except Exception as e:
            print(f"Error ranking chunks by similarity: {e}")
            # Return original order if ranking fails
            return chunks

    def batch_retrieve(
        self,
        queries: List[str],
        collection_name: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[List[RetrievedChunk]]:
        """
        Retrieve chunks for multiple queries in batch.

        Args:
            queries: List of query texts
            collection_name: Name of the Qdrant collection to search in
            top_k: Number of results to retrieve per query (uses default if None)
            similarity_threshold: Minimum similarity threshold (uses default if None)

        Returns:
            List of lists, where each inner list contains RetrievedChunk objects for the corresponding query
        """
        results = []
        for query in queries:
            chunks = self.retrieve_chunks(
                query_text=query,
                collection_name=collection_name,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            results.append(chunks)

        return results

    def validate_collection_exists(self, collection_name: str) -> bool:
        """
        Validate that the specified collection exists in Qdrant.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        return self.qdrant_service.collection_exists(collection_name)

    def get_collection_info(self, collection_name: str) -> Optional[models.CollectionInfo]:
        """
        Get information about the specified collection.

        Args:
            collection_name: Name of the collection to get info for

        Returns:
            CollectionInfo object if collection exists, None otherwise
        """
        return self.qdrant_service.get_collection_info(collection_name)