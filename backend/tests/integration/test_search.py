import pytest
from fastapi.testclient import TestClient
from main import app
import os
from unittest.mock import patch


# Create a test client
client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_search_endpoint_with_mock():
    """Test the search endpoint with mocked external services"""
    # This test requires mocking external services since we don't have real Cohere/Qdrant in test environment
    with patch('main.VectorDBService') as mock_service_class:
        mock_service_instance = mock_service_class.return_value
        mock_service_instance.search = mock_search_function

        # Make a search request
        search_data = {
            "query": "What is a humanoid robot?",
            "collection_name": "test_collection",
            "top_k": 3,
            "min_score": 0.5
        }
        response = client.post("/search", json=search_data)

        # Should succeed if the service is properly mocked
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert data["query"] == "What is a humanoid robot?"


def mock_search_function(query, collection_name, top_k=5, min_score=0.3):
    """Mock search function to simulate VectorDBService.search"""
    from models.chunk import SearchChunk

    # Return mock results
    mock_results = [
        SearchChunk(
            id="mock_id_1",
            text="A humanoid robot is a robot with physical features resembling those of a human...",
            score=0.95,
            metadata={"page_url": "/docs/intro", "heading": "Introduction"}
        ),
        SearchChunk(
            id="mock_id_2",
            text="Humanoid robots are designed to mimic human movements and behaviors...",
            score=0.87,
            metadata={"page_url": "/docs/design", "heading": "Design Principles"}
        )
    ]
    return mock_results


@patch.dict(os.environ, {
    "COHERE_API_KEY": "test-key",
    "QDRANT_URL": "http://localhost:6333"
})
def test_search_endpoint_invalid_request():
    """Test the search endpoint with invalid request data"""
    # Make a search request with missing required fields
    search_data = {
        "query": "What is a humanoid robot?"
        # Missing collection_name which should be required
    }
    response = client.post("/search", json=search_data)

    # Should return 422 for validation error or 400 for bad request
    assert response.status_code in [422, 400]


def test_job_status_endpoint():
    """Test the job status endpoint with a non-existent job"""
    response = client.get("/jobs/nonexistent_job_id")
    assert response.status_code == 404