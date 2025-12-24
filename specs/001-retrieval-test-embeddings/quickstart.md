# Quickstart: Retrieval & Test Embeddings

## Overview
This guide explains how to set up and run the retrieval validation system for Qdrant embeddings. The system tests that embeddings stored in Qdrant can be properly retrieved and that the retrieved content matches the original book text.

## Prerequisites
- Python 3.10+
- Access to the existing Qdrant collection with book embeddings
- Cohere API key for embedding generation
- Python packages: `qdrant-client`, `cohere`, `pytest`

## Setup
1. Ensure the RAG backend is properly configured with Qdrant connection
2. Set environment variables:
   ```bash
   export COHERE_API_KEY="your-api-key"
   export QDRANT_URL="your-qdrant-url"
   export QDRANT_API_KEY="your-qdrant-api-key"
   ```

## Running Validation Tests
1. Execute the main validation script:
   ```bash
   python -m backend.retrieval.test_queries --sections 10 --top-k 3
   ```

2. Or run individual tests:
   ```bash
   python -m backend.retrieval.retrieval_service --query "your sample query"
   ```

## Key Components
- `retrieval_service.py`: Core logic for querying Qdrant
- `validation_service.py`: Logic for comparing retrieved chunks with original text
- `result_logger.py`: Handles logging of validation results
- `test_queries.py`: Contains sample queries and test execution logic

## Configuration Options
- `--top-k`: Number of results to retrieve (default: 3)
- `--similarity-threshold`: Minimum similarity score for results (default: 0.3)
- `--sections`: Number of book sections to test (minimum: 10)

## Output
The validation process generates:
- Validation logs with accuracy metrics
- Detailed comparison reports
- Performance statistics
- Sample queries and their results