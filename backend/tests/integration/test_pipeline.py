import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, MagicMock
import os


# Create a test client
client = TestClient(app)


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_extract_endpoint():
    """Test the extract endpoint with mocked external services"""
    with patch('main.get_all_urls') as mock_get_urls:
        mock_get_urls.return_value = ["https://example.com/page1", "https://example.com/page2"]

        extract_data = {
            "base_url": "https://example.com"
        }
        response = client.post("/extract", json=extract_data)

        # Should succeed with mocked dependencies
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "processing"
        assert data["total_pages"] == 2


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_process_endpoint():
    """Test the process endpoint"""
    process_data = {
        "job_id": "extract_12345",
        "chunk_size": 512,
        "overlap": 50
    }
    response = client.post("/process", json=process_data)

    # Should succeed (the actual processing would be mocked in real usage)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "processing"


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_store_endpoint():
    """Test the store endpoint"""
    store_data = {
        "job_id": "process_67890",
        "collection_name": "rag_embedding"
    }
    response = client.post("/store", json=store_data)

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "uploading"


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_validate_endpoint():
    """Test the validate endpoint"""
    validate_data = {
        "collection_name": "rag_embedding"
    }
    response = client.post("/validate", json=validate_data)

    # Should succeed with mocked dependencies
    assert response.status_code == 200
    data = response.json()
    assert data["collection_name"] == "rag_embedding"
    assert "is_valid" in data


def test_job_status_endpoint():
    """Test the job status endpoint"""
    # Test with a job that doesn't exist
    response = client.get("/jobs/nonexistent_job")
    assert response.status_code == 404

    # In a real test, we'd create a job first and then check its status
    # For now, we'll just verify the endpoint exists and handles missing jobs properly


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_end_to_end_pipeline():
    """Test the end-to-end pipeline with mocked services"""
    # Step 1: Extract content
    with patch('main.get_all_urls') as mock_get_urls:
        mock_get_urls.return_value = ["https://example.com/page1"]

        extract_data = {"base_url": "https://example.com"}
        extract_response = client.post("/extract", json=extract_data)
        assert extract_response.status_code == 200
        extract_result = extract_response.json()
        job_id = extract_result["job_id"]
        assert job_id.startswith("extract_")

    # Step 2: Process content
    process_data = {"job_id": job_id, "chunk_size": 512}
    process_response = client.post("/process", json=process_data)
    assert process_response.status_code == 200

    # Step 3: Store embeddings
    store_data = {"job_id": "process_mock", "collection_name": "rag_embedding"}
    store_response = client.post("/store", json=store_data)
    assert store_response.status_code == 200

    # Step 4: Validate collection
    validate_data = {"collection_name": "rag_embedding"}
    validate_response = client.post("/validate", json=validate_data)
    assert validate_response.status_code == 200


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_error_handling_in_endpoints():
    """Test error handling in various endpoints"""
    # Test extract endpoint with invalid data
    response = client.post("/extract", json={})  # Missing required base_url
    assert response.status_code in [422, 400]  # Validation error

    # Test process endpoint with invalid data
    response = client.post("/process", json={})  # Missing required job_id
    assert response.status_code in [422, 400]  # Validation error

    # Test store endpoint with invalid data
    response = client.post("/store", json={})  # Missing required fields
    assert response.status_code in [422, 400]  # Validation error