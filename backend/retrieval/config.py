"""
Configuration module for retrieval and validation system.

This module provides default values and configuration options
for top-k retrieval and similarity thresholds.
"""
import os
from typing import Optional


class RetrievalConfig:
    """
    Configuration class for retrieval parameters.
    """

    # Default top-k value for retrieval
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))

    # Default similarity threshold for retrieval
    DEFAULT_SIMILARITY_THRESHOLD: float = float(
        os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.3")
    )

    # Minimum similarity threshold allowed
    MIN_SIMILARITY_THRESHOLD: float = float(
        os.getenv("MIN_SIMILARITY_THRESHOLD", "0.0")
    )

    # Maximum similarity threshold allowed
    MAX_SIMILARITY_THRESHOLD: float = float(
        os.getenv("MAX_SIMILARITY_THRESHOLD", "1.0")
    )

    # Default Qdrant collection name
    DEFAULT_COLLECTION_NAME: str = os.getenv(
        "DEFAULT_COLLECTION_NAME", "rag_embedding"
    )

    # Number of book sections to test by default
    DEFAULT_TEST_SECTIONS: int = int(os.getenv("DEFAULT_TEST_SECTIONS", "10"))

    # Accuracy threshold for validation (95% as per spec)
    VALIDATION_ACCURACY_THRESHOLD: float = float(
        os.getenv("VALIDATION_ACCURACY_THRESHOLD", "0.95")
    )

    # Maximum number of queries to process in a single validation run
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "50"))

    @classmethod
    def get_top_k(cls, top_k: Optional[int] = None) -> int:
        """
        Get the top-k value, using the provided value or the default.

        Args:
            top_k: Optional top-k value to use

        Returns:
            The top-k value to use for retrieval
        """
        if top_k is not None and top_k > 0:
            return top_k
        return cls.DEFAULT_TOP_K

    @classmethod
    def get_similarity_threshold(cls, threshold: Optional[float] = None) -> float:
        """
        Get the similarity threshold, using the provided value or the default.

        Args:
            threshold: Optional threshold value to use

        Returns:
            The similarity threshold to use for retrieval
        """
        if threshold is not None and cls.MIN_SIMILARITY_THRESHOLD <= threshold <= cls.MAX_SIMILARITY_THRESHOLD:
            return threshold
        return cls.DEFAULT_SIMILARITY_THRESHOLD