import pytest
from unittest.mock import AsyncMock, patch
from main import search_content, SearchRequest
from models.chunk import SearchChunk


@pytest.mark.asyncio
async def test_search_content():
    """Test the search content functionality"""
    # Mock request
    request = SearchRequest(
        query="What is a humanoid robot?",
        collection_name="test_collection",
        top_k=5,
        min_score=0.3
    )

    # Mock the VectorDBService
    with patch('main.VectorDBService') as mock_service_class:
        mock_service_instance = AsyncMock()
        mock_service_class.return_value = mock_service_instance

        mock_result = [
            SearchChunk(
                id="test_id",
                text="A humanoid robot is a robot with physical features resembling those of a human...",
                score=0.95,
                metadata={"page_url": "/docs/intro", "heading": "Introduction"}
            )
        ]
        mock_service_instance.search.return_value = mock_result

        # Call the search function
        result = await search_content(request)

        # Verify the result
        assert result.query == "What is a humanoid robot?"
        assert len(result.results) == 1
        assert result.results[0].text == "A humanoid robot is a robot with physical features resembling those of a human..."
        assert result.results[0].score == 0.95

        # Verify that the service method was called with correct parameters
        mock_service_instance.search.assert_called_once_with(
            query="What is a humanoid robot?",
            collection_name="test_collection",
            top_k=5,
            min_score=0.3
        )


@pytest.mark.asyncio
async def test_search_content_exception():
    """Test search content functionality when an exception occurs"""
    request = SearchRequest(
        query="Test query",
        collection_name="test_collection"
    )

    # Mock the VectorDBService to raise an exception
    with patch('main.VectorDBService') as mock_service_class:
        mock_service_instance = AsyncMock()
        mock_service_class.return_value = mock_service_instance
        mock_service_instance.search.side_effect = Exception("Test error")

        # Expect HTTPException to be raised
        with pytest.raises(Exception):
            await search_content(request)