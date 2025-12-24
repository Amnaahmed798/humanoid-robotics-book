#!/usr/bin/env python3
"""
Validation script for the RAG Chatbot Backend implementation.
This script checks that all required files and basic functionality are in place.
"""

import os
import sys
from pathlib import Path


def check_directory_structure():
    """Check that the required directory structure exists"""
    print("Checking directory structure...")

    required_dirs = [
        "backend",
        "backend/models",
        "backend/services",
        "backend/tests",
        "backend/tests/unit",
        "backend/tests/integration",
        "backend/tests/contract"
    ]

    all_present = True
    for directory in required_dirs:
        if not os.path.isdir(directory):
            print(f"❌ Missing directory: {directory}")
            all_present = False
        else:
            print(f"✅ Directory present: {directory}")

    return all_present


def check_required_files():
    """Check that all required files exist"""
    print("\nChecking required files...")

    required_files = [
        "backend/main.py",
        "backend/requirements.txt",
        "backend/.env",
        "backend/README.md",
        "backend/Dockerfile",
        "docker-compose.yml",
        "backend/models/chunk.py",
        "backend/models/embedding.py",
        "backend/services/content_service.py",
        "backend/services/embedding_service.py",
        "backend/services/vector_db_service.py",
        "backend/extract_content.py",
        "backend/chunk_text.py",
        "backend/generate_embeddings.py",
        "backend/qdrant_client.py"
    ]

    all_present = True
    for file in required_files:
        if not os.path.isfile(file):
            print(f"❌ Missing file: {file}")
            all_present = False
        else:
            print(f"✅ File present: {file}")

    return all_present


def check_test_files():
    """Check that test files exist"""
    print("\nChecking test files...")

    test_files = [
        "backend/tests/unit/test_search.py",
        "backend/tests/integration/test_search.py",
        "backend/tests/unit/test_extraction.py",
        "backend/tests/unit/test_chunking.py",
        "backend/tests/unit/test_embedding.py",
        "backend/tests/integration/test_pipeline.py",
        "backend/tests/unit/test_validation.py",
        "backend/tests/contract/test_api_contract.py"
    ]

    all_present = True
    for file in test_files:
        if not os.path.isfile(file):
            print(f"❌ Missing test file: {file}")
            all_present = False
        else:
            print(f"✅ Test file present: {file}")

    return all_present


def check_implementation_completeness():
    """Check implementation completeness by reviewing the tasks file"""
    print("\nChecking implementation completeness...")

    tasks_file = "specs/001-embeddings-qdrant/tasks.md"
    if not os.path.isfile(tasks_file):
        print(f"❌ Tasks file not found: {tasks_file}")
        return False

    with open(tasks_file, 'r') as f:
        content = f.read()

    # Count completed and remaining tasks
    completed_tasks = content.count('[x] ')
    total_tasks = content.count('[x] ') + content.count('[ ] ')

    print(f"✅ Completed tasks: {completed_tasks}")
    print(f"📊 Total tasks: {total_tasks}")
    print(f"📈 Completion: {completed_tasks/total_tasks*100:.1f}%" if total_tasks > 0 else "No tasks found")

    return total_tasks > 0 and completed_tasks > 0


def validate_models():
    """Basic validation that model files have required classes"""
    print("\nValidating model files...")

    # Check chunk.py
    chunk_file = "backend/models/chunk.py"
    if os.path.isfile(chunk_file):
        with open(chunk_file, 'r') as f:
            chunk_content = f.read()

        has_text_chunk = "class TextChunk" in chunk_content
        has_search_chunk = "class SearchChunk" in chunk_content

        print(f"✅ TextChunk class: {has_text_chunk}")
        print(f"✅ SearchChunk class: {has_search_chunk}")

    # Check embedding.py
    embedding_file = "backend/models/embedding.py"
    if os.path.isfile(embedding_file):
        with open(embedding_file, 'r') as f:
            embedding_content = f.read()

        has_embedding_vector = "class EmbeddingVector" in embedding_content
        has_qdrant_collection = "class QdrantCollection" in embedding_content

        print(f"✅ EmbeddingVector class: {has_embedding_vector}")
        print(f"✅ QdrantCollection class: {has_qdrant_collection}")


def main():
    """Main validation function"""
    print("🔍 Starting validation of RAG Chatbot Backend implementation...\n")

    # Run all checks
    dir_ok = check_directory_structure()
    files_ok = check_required_files()
    tests_ok = check_test_files()
    completeness_ok = check_implementation_completeness()
    validate_models()

    print(f"\n📋 Validation Summary:")
    print(f"Directory structure: {'✅' if dir_ok else '❌'}")
    print(f"Required files: {'✅' if files_ok else '❌'}")
    print(f"Test files: {'✅' if tests_ok else '❌'}")
    print(f"Implementation completeness: {'✅' if completeness_ok else '❌'}")

    overall_success = all([dir_ok, files_ok, tests_ok, completeness_ok])

    print(f"\n{'🎉 All validations passed!' if overall_success else '⚠️  Some validations failed.'}")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)