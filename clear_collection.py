#!/usr/bin/env python3
"""Script to clear the Qdrant collection before re-storing content."""

import sys
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv('./backend/.env')

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
    # Local instance
    client = QdrantClient(url=qdrant_url)
else:
    # Cloud instance
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        prefer_grpc=True
    )

collection_name = "book_content"

try:
    # Delete the collection
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' deleted successfully")
except Exception as e:
    print(f"Error deleting collection '{collection_name}': {e}")

# Create the collection again
try:
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,  # For Cohere multilingual model
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' recreated successfully")
except Exception as e:
    print(f"Error creating collection '{collection_name}': {e}")