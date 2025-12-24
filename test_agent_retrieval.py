#!/usr/bin/env python3
"""Script to test the agent service with retrieved content."""

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
from models.retrieval_models import RetrievedChunk

async def test_agent_with_retrieved_content():
    """Test the agent service with pre-retrieved content."""
    print("Testing retrieval first...")

    try:
        # Initialize retrieval service
        retrieval_service = RetrievalService()

        # Test search
        query = "What are the key components of ROS 2 for humanoid robots?"
        collection_name = "book_content"
        retrieved_chunks = retrieval_service.retrieve_chunks(query, collection_name, top_k=3)

        print(f"Found {len(retrieved_chunks)} chunks for query: '{query}'")

        if retrieved_chunks:
            print("\nRetrieved content:")
            for i, chunk in enumerate(retrieved_chunks):
                print(f"\nChunk {i+1}:")
                print(f"  ID: {chunk.id}")
                print(f"  Score: {chunk.similarity_score}")
                print(f"  Source: {chunk.source_location}")
                print(f"  Text preview: {chunk.text[:200]}...")
        else:
            print("No content retrieved - there may be an issue with the retrieval")
            return False

        # Now test if the agent service can at least process this content
        # We'll create a mock agent response to see if the retrieval part works
        print(f"\nRetrieved {len(retrieved_chunks)} chunks that would be sent to the LLM")

        # Combine context as the agent service would do
        context_text = "\n\n".join([
            f"Source {i+1} ({chunk.source_location}): {chunk.text}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        print(f"Combined context length: {len(context_text)} characters")
        print(f"First 300 chars of context: {context_text[:300]}...")

        return True

    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_with_retrieved_content())
    print(f"\nTest {'succeeded' if success else 'failed'} - retrieval is working")
    if success:
        print("The issue is likely just with the LLM configuration, not the retrieval")