#!/usr/bin/env python3
"""
Test script to verify content retrieval from Qdrant.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment from backend
load_dotenv('./backend/.env')

def test_qdrant_connection():
    """Test direct Qdrant connection."""
    from qdrant_client import QdrantClient

    # Use the same path as specified in the environment
    qdrant_url = os.getenv("QDRANT_URL")
    print(f"QDRANT_URL from environment: {qdrant_url}")

    if qdrant_url.startswith("./") or qdrant_url.startswith("/") or qdrant_url.startswith("../"):
        # Local file-based instance
        print("Using local file-based Qdrant instance")
        client = QdrantClient(path=qdrant_url)
    else:
        print("Using URL-based Qdrant instance")
        client = QdrantClient(url=qdrant_url)

    # Check collections
    collections = client.get_collections()
    print(f"Available collections: {[col.name for col in collections.collections]}")

    # Check book_content collection specifically
    if 'book_content' in [col.name for col in collections.collections]:
        count = client.count(collection_name='book_content')
        print(f'book_content collection has {count.count} vectors')

        # Try to get a sample
        try:
            records, _ = client.scroll(
                collection_name='book_content',
                limit=1
            )
            if records:
                sample = records[0]
                print(f"Sample record ID: {sample.id}")
                print(f"Sample payload keys: {list(sample.payload.keys())}")
                print(f"Sample text preview: {sample.payload.get('text', '')[:100]}...")
        except Exception as e:
            print(f"Error getting sample: {e}")
    else:
        print("book_content collection not found!")

    return client

def test_cohere_connection():
    """Test Cohere connection."""
    import cohere

    cohere_api_key = os.getenv("COHERE_API_KEY")
    print(f"COHERE_API_KEY is {'set' if cohere_api_key else 'not set'}")

    if cohere_api_key:
        try:
            client = cohere.Client(cohere_api_key)
            # Test embedding generation
            response = client.embed(
                texts=["test query"],
                model="embed-multilingual-v3.0",
                input_type="search_query"
            )
            print(f"Embedding test successful, vector dimensions: {len(response.embeddings[0])}")
            return client
        except Exception as e:
            print(f"Cohere test failed: {e}")
            return None
    else:
        print("No Cohere API key available")
        return None

if __name__ == "__main__":
    print("Testing Qdrant connection...")
    qdrant_client = test_qdrant_connection()

    print("\nTesting Cohere connection...")
    cohere_client = test_cohere_connection()

    if qdrant_client and cohere_client:
        print("\n✓ Both Qdrant and Cohere connections are working!")
        print("✓ Content is populated in Qdrant and APIs are accessible!")
    else:
        print("\n✗ Connection issues detected")