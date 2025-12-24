import pytest
from fastapi.testclient import TestClient
from backend.agent.main import app
from unittest.mock import patch, Mock
from backend.models.retrieval_models import RetrievedChunk


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.asyncio
@patch('backend.agent.api.retrieval_service')
@patch('backend.agent.api.agent_service')
@patch('backend.agent.api.validation_service')
def test_query_endpoint_success(mock_validation_service, mock_agent_service, mock_retrieval_service, client):
    """Test successful query endpoint request."""
    # Mock the retrieval service
    mock_retrieval_result = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]
    mock_retrieval_service.retrieve = Mock(return_value=mock_retrieval_result)

    # Mock the agent service
    from backend.models.agent_models import AgentResponse
    mock_agent_response = AgentResponse(
        answer="Humanoid robots are robots with physical features resembling humans.",
        sources=mock_retrieval_result,
        confidence=0.8
    )
    mock_agent_service.generate_response = Mock(return_value=mock_agent_response)

    # Mock the validation service
    mock_validation_service.validate_response = Mock(return_value=True)

    # Make the request
    response = client.post(
        "/api/v1/query",
        json={"question": "What are humanoid robots?"}
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "Humanoid robots" in data["answer"]


@pytest.mark.asyncio
@patch('backend.agent.api.retrieval_service')
@patch('backend.agent.api.agent_service')
@patch('backend.agent.api.validation_service')
def test_query_endpoint_no_retrieved_chunks(mock_validation_service, mock_agent_service, mock_retrieval_service, client):
    """Test query endpoint when no chunks are retrieved."""
    # Mock the retrieval service to return empty list
    mock_retrieval_service.retrieve = Mock(return_value=[])

    # Make the request
    response = client.post(
        "/api/v1/query",
        json={"question": "What are humanoid robots?"}
    )

    # Should return a response indicating no information found
    assert response.status_code == 200
    data = response.json()
    assert "not found" in data["answer"] or "not available" in data["answer"]


@pytest.mark.asyncio
@patch('backend.agent.api.retrieval_service')
@patch('backend.agent.api.agent_service')
@patch('backend.agent.api.validation_service')
def test_query_endpoint_validation_fails(mock_validation_service, mock_agent_service, mock_retrieval_service, client):
    """Test query endpoint when response validation fails."""
    # Mock the retrieval service
    mock_retrieval_result = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]
    mock_retrieval_service.retrieve = Mock(return_value=mock_retrieval_result)

    # Mock the agent service
    from backend.models.agent_models import AgentResponse
    mock_agent_response = AgentResponse(
        answer="Humanoid robots are robots with physical features resembling humans.",
        sources=mock_retrieval_result,
        confidence=0.8
    )
    mock_agent_service.generate_response = Mock(return_value=mock_agent_response)

    # Mock the validation service to return False
    mock_validation_service.validate_response = Mock(return_value=False)

    # Make the request
    response = client.post(
        "/api/v1/query",
        json={"question": "What are humanoid robots?"}
    )

    # Should return a 500 error when validation fails
    assert response.status_code == 500


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Retrieval-Enabled Agent API is running" in data["message"]