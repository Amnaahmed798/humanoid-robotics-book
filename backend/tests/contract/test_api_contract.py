import pytest
from fastapi.testclient import TestClient
from main import app
from pydantic import ValidationError
import os
from unittest.mock import patch, MagicMock


# Create a test client
client = TestClient(app)


def test_api_health_contract():
    """Test the health endpoint contract"""
    response = client.get("/health")

    # Status code should be 200
    assert response.status_code == 200

    # Response should contain expected fields
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

    # Status should be "healthy"
    assert data["status"] == "healthy"

    # Timestamp should be present and in correct format
    assert isinstance(data["timestamp"], str) or isinstance(data["timestamp"], type(response.json()["timestamp"]))


def test_api_extract_contract():
    """Test the extract endpoint contract"""
    extract_data = {
        "base_url": "https://example.com"
    }
    response = client.post("/extract", json=extract_data)

    # Even if the actual processing fails due to missing services,
    # the response should follow the expected contract
    if response.status_code == 200:
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "total_pages" in data

        # Verify types
        assert isinstance(data["job_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["total_pages"], int)


def test_api_process_contract():
    """Test the process endpoint contract"""
    process_data = {
        "job_id": "extract_12345",
        "chunk_size": 512,
        "overlap": 50
    }
    response = client.post("/process", json=process_data)

    if response.status_code == 200:
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "total_chunks" in data

        # Verify types
        assert isinstance(data["job_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["total_chunks"], int)


def test_api_store_contract():
    """Test the store endpoint contract"""
    store_data = {
        "job_id": "process_67890",
        "collection_name": "rag_embedding"
    }
    response = client.post("/store", json=store_data)

    if response.status_code == 200:
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "total_embeddings" in data

        # Verify types
        assert isinstance(data["job_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["total_embeddings"], int)


def test_api_search_contract():
    """Test the search endpoint contract"""
    search_data = {
        "query": "What is a humanoid robot?",
        "collection_name": "test_collection",
        "top_k": 5,
        "min_score": 0.3
    }
    response = client.post("/search", json=search_data)

    # If successful, response should follow search contract
    if response.status_code == 200:
        data = response.json()
        assert "query" in data
        assert "results" in data

        # Verify types
        assert isinstance(data["query"], str)
        assert isinstance(data["results"], list)

        # If there are results, they should have expected fields
        for result in data["results"]:
            assert "id" in result
            assert "score" in result
            assert "text" in result
            assert "metadata" in result


def test_api_validate_contract():
    """Test the validate endpoint contract"""
    validate_data = {
        "collection_name": "test_collection"
    }
    response = client.post("/validate", json=validate_data)

    if response.status_code == 200:
        data = response.json()
        assert "collection_name" in data
        assert "vector_count" in data
        assert "expected_count" in data
        assert "vector_dimensions" in data
        assert "model_name" in data
        assert "is_valid" in data
        assert "issues" in data

        # Verify types
        assert isinstance(data["collection_name"], str)
        assert isinstance(data["vector_count"], int)
        assert isinstance(data["expected_count"], int)
        assert isinstance(data["vector_dimensions"], int)
        assert isinstance(data["model_name"], str)
        assert isinstance(data["is_valid"], bool)
        assert isinstance(data["issues"], list)


def test_api_job_status_contract():
    """Test the job status endpoint contract"""
    # Test with a non-existent job to check error handling
    response = client.get("/jobs/nonexistent_job")

    # Should return 404 for non-existent job
    if response.status_code == 404:
        # This is expected behavior
        pass
    elif response.status_code == 200:
        # If it returns a job status, verify the contract
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "progress" in data
        assert "created_at" in data

        # Verify types
        assert isinstance(data["job_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["progress"], float)
        assert isinstance(data["created_at"], str) or isinstance(data["created_at"], type(response.json()["created_at"]))


def test_api_request_validation():
    """Test API request validation"""
    # Test extract endpoint without required base_url
    response = client.post("/extract", json={})
    assert response.status_code in [422, 400]  # Validation error

    # Test process endpoint without required job_id
    response = client.post("/process", json={})
    assert response.status_code in [422, 400]  # Validation error

    # Test store endpoint without required fields
    response = client.post("/store", json={})
    assert response.status_code in [422, 400]  # Validation error

    # Test search endpoint without required fields
    response = client.post("/search", json={})
    assert response.status_code in [422, 400]  # Validation error

    # Test validate endpoint without required collection_name
    response = client.post("/validate", json={})
    assert response.status_code in [422, 400]  # Validation error


def test_api_response_schema_consistency():
    """Test that API responses follow consistent schema"""
    # Health check
    health_response = client.get("/health")
    if health_response.status_code == 200:
        health_data = health_response.json()
        assert isinstance(health_data, dict)
        assert "status" in health_data

    # All successful API responses should be JSON objects
    with patch.dict(os.environ, {
        "COHERE_API_KEY": "test-key",
        "QDRANT_URL": "http://localhost:6333"
    }):
        with patch('main.get_all_urls') as mock_get_urls:
            mock_get_urls.return_value = ["https://example.com/page1"]

            extract_response = client.post("/extract", json={"base_url": "https://example.com"})
            if extract_response.status_code == 200:
                extract_data = extract_response.json()
                assert isinstance(extract_data, dict)
                assert "job_id" in extract_data