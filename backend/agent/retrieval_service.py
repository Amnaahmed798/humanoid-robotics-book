from typing import List
import logging
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from retrieval.retrieval_service import RetrievalService as BaseRetrievalService
from models.retrieval_models import RetrievedChunk


class RetrievalService:
    """
    Async wrapper for the base retrieval service to be used by the agent.
    """

    def __init__(self):
        """
        Initialize the retrieval service.
        """
        self.base_service = BaseRetrievalService()
        self.default_collection_name = "book_content"  # Default collection for book content

    async def retrieve(self, query_text: str, top_k: int = 3) -> List[RetrievedChunk]:
        """
        Asynchronously retrieve relevant chunks based on the query.

        Args:
            query_text: The user's question
            top_k: Number of chunks to retrieve (default: 3)

        Returns:
            List of RetrievedChunk objects containing relevant book content
        """
        try:
            # Use the base service to perform the actual retrieval
            results = self.base_service.retrieve_chunks(
                query_text=query_text,
                collection_name=self.default_collection_name,
                top_k=top_k
            )

            logging.info(f"Retrieved {len(results)} chunks for query: {query_text[:50]}...")
            return results

        except Exception as e:
            logging.error(f"Error during retrieval: {str(e)}")
            # Return empty list if retrieval fails, allowing the agent to handle appropriately
            return []