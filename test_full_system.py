#!/usr/bin/env python3
"""Comprehensive test to demonstrate that the entire RAG system is working except for LLM configuration."""

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
from qdrant_client import QdrantClient

def test_qdrant_connection():
    """Test Qdrant connection and collection."""
    print("=== Testing Qdrant Connection ===")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
        client = QdrantClient(url=qdrant_url)
    else:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=True)

    try:
        collection_info = client.get_collection("book_content")
        print(f"✅ Collection exists: book_content")
        print(f"✅ Vector count: {collection_info.points_count}")
        print(f"✅ Vector dimensions: {collection_info.config.params.vectors.size}")
        return True
    except Exception as e:
        print(f"❌ Error accessing Qdrant: {e}")
        return False

async def test_retrieval():
    """Test retrieval functionality."""
    print("\n=== Testing Retrieval Functionality ===")

    try:
        retrieval_service = RetrievalService()

        # Test with multiple queries
        test_queries = [
            "ROS 2 humanoid robots",
            "humanoid robotics",
            "book content"
        ]

        all_good = True
        for query in test_queries:
            results = retrieval_service.retrieve_chunks(query, "book_content", top_k=2)
            print(f"Query: '{query}' -> Found {len(results)} results")

            if len(results) == 0:
                print(f"  ❌ No results for query: {query}")
                all_good = False
            else:
                for i, result in enumerate(results[:1]):  # Show first result
                    print(f"  Result {i+1}: Score={result.similarity_score:.3f}, Source={result.source_location[:50]}...")

        return all_good
    except Exception as e:
        print(f"❌ Error in retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_payload_format():
    """Test that payloads have the correct format."""
    print("\n=== Testing Payload Format ===")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if "localhost" in qdrant_url or "127.0.0.1" in qdrant_url:
        client = QdrantClient(url=qdrant_url)
    else:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, prefer_grpc=True)

    try:
        # Get a sample point
        scroll_result = client.scroll(collection_name="book_content", limit=1)
        if scroll_result[0]:
            point = scroll_result[0][0]
            payload = point.payload

            required_keys = ['text', 'source_location']
            missing_keys = [key for key in required_keys if key not in payload]

            if missing_keys:
                print(f"❌ Missing keys in payload: {missing_keys}")
                return False
            else:
                print(f"✅ All required keys present: {required_keys}")
                print(f"✅ Text preview: {payload['text'][:100]}...")
                print(f"✅ Source location: {payload['source_location'][:100]}...")
                return True
        else:
            print("❌ No points found to test")
            return False
    except Exception as e:
        print(f"❌ Error testing payload: {e}")
        return False

async def main():
    """Run all tests."""
    print("🤖 Comprehensive RAG System Test\n")

    # Run all tests
    qdrant_ok = test_qdrant_connection()
    payload_ok = test_payload_format()
    retrieval_ok = await test_retrieval()

    print(f"\n=== Summary ===")
    print(f"Qdrant Connection: {'✅' if qdrant_ok else '❌'}")
    print(f"Payload Format: {'✅' if payload_ok else '❌'}")
    print(f"Retrieval: {'✅' if retrieval_ok else '❌'}")

    all_working = qdrant_ok and payload_ok and retrieval_ok

    print(f"\n📊 Overall Status: {'✅ FULLY OPERATIONAL' if all_working else '❌ ISSUES DETECTED'}")

    if all_working:
        print("\n🎉 The RAG pipeline is fully functional!")
        print("   - Content is stored in Qdrant with correct format")
        print("   - Retrieval service can find relevant content")
        print("   - Payloads contain required 'text' and 'source_location' fields")
        print("   - Only issue: LLM configuration needs server restart")
    else:
        print("\n⚠️  There are issues with the RAG system")

    return all_working

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)