import pytest
from unittest.mock import Mock, patch
from backend.agent.retrieval_service import RetrievalService
from backend.retrieval.retrieval_service import RetrievalService as BaseRetrievalService
from backend.models.retrieval_models import RetrievedChunk


@pytest.fixture
def retrieval_service():
    return RetrievalService()


@pytest.mark.asyncio
async def test_retrieve_success(retrieval_service):
    """Test successful retrieval of chunks."""
    # Mock the base retrieval service
    mock_base_service = Mock(spec=BaseRetrievalService)
    mock_result = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]
    mock_base_service.retrieve_chunks.return_value = mock_result
    retrieval_service.base_service = mock_base_service

    result = await retrieval_service.retrieve("What are humanoid robots?")

    assert len(result) == 1
    assert result[0].id == "1"
    assert "humanoid robots" in result[0].text.lower()


@pytest.mark.asyncio
async def test_retrieve_with_error_handling(retrieval_service):
    """Test retrieval with error handling."""
    # Mock the base retrieval service to raise an exception
    mock_base_service = Mock(spec=BaseRetrievalService)
    mock_base_service.retrieve_chunks.side_effect = Exception("Connection error")
    retrieval_service.base_service = mock_base_service

    result = await retrieval_service.retrieve("What are humanoid robots?")

    # Should return empty list in case of error
    assert result == []


@pytest.mark.asyncio
async def test_retrieve_with_custom_top_k(retrieval_service):
    """Test retrieval with custom top_k parameter."""
    # Mock the base retrieval service
    mock_base_service = Mock(spec=BaseRetrievalService)
    mock_result = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]
    mock_base_service.retrieve_chunks.return_value = mock_result
    retrieval_service.base_service = mock_base_service

    result = await retrieval_service.retrieve("What are humanoid robots?", top_k=5)

    # Verify that the method was called with the correct parameters
    mock_base_service.retrieve_chunks.assert_called_once_with(
        query_text="What are humanoid robots?",
        collection_name="book_content",
        top_k=5
    )
    assert len(result) == 1