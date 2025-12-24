#!/usr/bin/env python3
"""Script to check if content is actually stored in Qdrant."""

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
    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' exists")
    print(f"Points count: {collection_info.points_count}")
    print(f"Config: {collection_info.config}")

    # Try to get a sample of points
    if collection_info.points_count > 0:
        # Get the first few points to verify content
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=3
        )
        print(f"\nSample points from collection:")
        for i, point in enumerate(scroll_result[0]):
            print(f"Point {i+1}:")
            print(f"  ID: {point.id} (type: {type(point.id)})")
            print(f"  Vector length: {len(point.vector)}")
            print(f"  Payload keys: {list(point.payload.keys())}")
            print(f"  Text preview: {point.payload.get('text', '')[:100]}...")
            print()
    else:
        print(f"\nNo points found in collection '{collection_name}'")

except Exception as e:
    print(f"Error checking collection '{collection_name}': {e}")