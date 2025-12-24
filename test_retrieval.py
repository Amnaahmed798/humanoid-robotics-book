#!/usr/bin/env python3
"""Script to test retrieval directly."""

import sys
import os
import asyncio
from dotenv import load_dotenv

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

# Load environment variables
load_dotenv('./backend/.env')

from retrieval.retrieval_service import RetrievalService

async def test_retrieval():
    """Test the retrieval service directly."""
    print("Testing retrieval service...")

    try:
        # Initialize retrieval service
        retrieval_service = RetrievalService()

        # Test search
        query = "ROS 2 humanoid robots"
        collection_name = "book_content"
        results = retrieval_service.retrieve_chunks(query, collection_name, top_k=3)

        print(f"Found {len(results)} results for query: '{query}'")

        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"  ID: {result.id}")
            print(f"  Score: {result.similarity_score}")
            print(f"  Source: {result.source_location}")
            print(f"  Text preview: {result.text[:200]}...")

        if not results:
            print("No results found - this indicates the retrieval isn't working properly")

        return len(results) > 0

    except Exception as e:
        print(f"Error testing retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_retrieval())
    print(f"\nRetrieval test {'succeeded' if success else 'failed'}")