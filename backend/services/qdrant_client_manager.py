"""
Global Qdrant client manager to ensure single connection for local storage.
This module provides a single shared instance of the Qdrant client to avoid
concurrent access issues with local file-based storage.
"""
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global Qdrant client instance
_qdrant_client = None

def get_qdrant_client():
    """
    Get the shared Qdrant client instance.
    This ensures only one connection exists for local file-based storage.
    """
    global _qdrant_client

    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")

        # Check if using local file-based storage
        if qdrant_url.startswith("./") or qdrant_url.startswith("/") or qdrant_url.startswith("../"):
            # Local file-based instance - use single client to avoid locking
            _qdrant_client = QdrantClient(path=qdrant_url)
        elif "localhost" in qdrant_url or "http://" in qdrant_url:
            # HTTP connection to local or remote instance
            _qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
        else:
            # Cloud instance with URL
            _qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=True
            )

    return _qdrant_client