from typing import List, Dict, Any
from models.embedding import EmbeddingVector
from models.chunk import TextChunk
import cohere
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating embeddings using Cohere.
    """

    def __init__(self):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        self.co = cohere.Client(api_key)
        self.model_name = "embed-multilingual-v3.0"  # As decided in research

    async def embed_text(self, text: str) -> EmbeddingVector:
        """
        Generate embedding for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            EmbeddingVector object
        """
        try:
            response = self.co.embed(
                texts=[text],
                model=self.model_name,
                input_type="search_document"  # Using search_document as this is for RAG
            )

            embedding_data = response.embeddings[0]

            embedding_vector = EmbeddingVector(
                vector=embedding_data,
                chunk_id="",  # Will be set when associated with a chunk
                model_name=self.model_name
            )

            return embedding_vector

        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of EmbeddingVector objects
        """
        if not texts:
            return []

        try:
            # Cohere has a limit on the number of texts per request
            # For now, we'll process all at once, but in production we might need batching
            response = self.co.embed(
                texts=texts,
                model=self.model_name,
                input_type="search_document"
            )

            embeddings = []
            for i, embedding_data in enumerate(response.embeddings):
                embedding_vector = EmbeddingVector(
                    vector=embedding_data,
                    chunk_id=f"text_{i}",  # Temporary ID, should be set properly when used
                    model_name=self.model_name
                )
                embeddings.append(embedding_vector)

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings for texts: {str(e)}")
            raise

    async def embed_chunks(self, chunks: List[TextChunk]) -> List[EmbeddingVector]:
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

        try:
            # Generate embeddings for all texts at once
            response = self.co.embed(
                texts=texts,
                model=self.model_name,
                input_type="search_document"
            )

            embeddings = []
            for i, embedding_data in enumerate(response.embeddings):
                chunk = chunks[i]
                embedding_vector = EmbeddingVector(
                    id=f"emb_{chunk.id}" if chunk.id else f"emb_{i}",
                    vector=embedding_data,
                    chunk_id=chunk.id or f"chunk_{i}",
                    model_name=self.model_name
                )
                embeddings.append(embedding_vector)

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings for chunks: {str(e)}")
            raise

    async def validate_embedding(self, embedding: EmbeddingVector) -> bool:
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
            if not embedding.model_name or embedding.model_name != self.model_name:
                logger.warning(f"Embedding model name {embedding.model_name} doesn't match expected {self.model_name}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating embedding: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model being used.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimensions": 1024,
            "description": "Cohere multilingual embedding model v3.0, optimized for multi-lingual content and RAG applications"
        }