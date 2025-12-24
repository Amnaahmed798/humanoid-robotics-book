#!/usr/bin/env python3
"""
Test script for executing sample queries against the Qdrant collection.

This script tests the retrieval functionality by executing sample queries
and verifying that relevant content is returned from the book.
"""
import sys
import os
import asyncio
from typing import List

# Add the backend directory to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval_service import RetrievalService
from config import RetrievalConfig


def test_basic_retrieval():
    """
    Test basic retrieval functionality with sample queries.
    """
    print("Testing basic retrieval functionality...")

    # Initialize the retrieval service
    retrieval_service = RetrievalService()

    # Test queries
    test_queries = [
        "What are the key principles of humanoid robotics?",
        "Explain the kinematic chain in robotics",
        "What are the main components of a humanoid robot?",
        "How does inverse kinematics work?",
        "What is the difference between forward and inverse kinematics?"
    ]

    # Use the default collection name
    collection_name = RetrievalConfig.DEFAULT_COLLECTION_NAME

    print(f"Using collection: {collection_name}")

    # Check if collection exists
    if not retrieval_service.validate_collection_exists(collection_name):
        print(f"Error: Collection '{collection_name}' does not exist in Qdrant")
        return False

    print("Collection exists, proceeding with tests...")

    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: '{query}' ---")
        try:
            # Retrieve chunks
            chunks = retrieval_service.retrieve_chunks(
                query_text=query,
                collection_name=collection_name,
                top_k=3
            )

            print(f"Retrieved {len(chunks)} chunks:")
            for j, chunk in enumerate(chunks, 1):
                print(f"  {j}. Score: {chunk.similarity_score:.3f}, Source: {chunk.source_location}")
                print(f"     Text preview: {chunk.text[:100]}...")

        except Exception as e:
            print(f"Error retrieving chunks for query '{query}': {e}")
            return False

    print("\n✓ Basic retrieval tests completed successfully")
    return True


def test_ranking_functionality():
    """
    Test the ranking functionality.
    """
    print("\nTesting ranking functionality...")

    # Initialize the retrieval service
    retrieval_service = RetrievalService()

    # Use the default collection name
    collection_name = RetrievalConfig.DEFAULT_COLLECTION_NAME

    # Get some initial results
    query = "humanoid robot design principles"
    try:
        chunks = retrieval_service.retrieve_chunks(
            query_text=query,
            collection_name=collection_name,
            top_k=5
        )

        if len(chunks) > 1:
            # Rank the chunks
            ranked_chunks = retrieval_service.rank_by_similarity(chunks, query)

            print(f"Original order (by Qdrant similarity):")
            for i, chunk in enumerate(chunks, 1):
                print(f"  {i}. Score: {chunk.similarity_score:.3f}")

            print(f"Re-ranked by semantic similarity to query:")
            for i, chunk in enumerate(ranked_chunks, 1):
                print(f"  {i}. Score: {chunk.similarity_score:.3f}")

            print("✓ Ranking test completed")
        else:
            print("Not enough chunks retrieved to test ranking")

    except Exception as e:
        print(f"Error testing ranking functionality: {e}")
        return False

    return True


def test_configuration_parameters():
    """
    Test different configuration parameters.
    """
    print("\nTesting configuration parameters...")

    # Initialize the retrieval service
    retrieval_service = RetrievalService()

    # Use the default collection name
    collection_name = RetrievalConfig.DEFAULT_COLLECTION_NAME

    query = "kinematics in robotics"

    # Test different top_k values
    for top_k in [1, 3, 5]:
        try:
            chunks = retrieval_service.retrieve_chunks(
                query_text=query,
                collection_name=collection_name,
                top_k=top_k
            )

            print(f"  top_k={top_k}: Retrieved {len(chunks)} chunks")

        except Exception as e:
            print(f"Error testing top_k={top_k}: {e}")
            return False

    # Test different similarity thresholds
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        try:
            chunks = retrieval_service.retrieve_chunks(
                query_text=query,
                collection_name=collection_name,
                top_k=5,
                similarity_threshold=threshold
            )

            print(f"  threshold={threshold}: Retrieved {len(chunks)} chunks")

        except Exception as e:
            print(f"Error testing threshold={threshold}: {e}")
            return False

    print("✓ Configuration parameter tests completed")
    return True


def main():
    """
    Main function to run all tests.
    """
    print("Starting retrieval and validation tests...\n")

    success = True

    # Run basic retrieval tests
    success &= test_basic_retrieval()

    # Run ranking tests if basic tests passed
    if success:
        success &= test_ranking_functionality()

    # Run configuration tests if previous tests passed
    if success:
        success &= test_configuration_parameters()

    if success:
        print("\n✓ All tests completed successfully!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())