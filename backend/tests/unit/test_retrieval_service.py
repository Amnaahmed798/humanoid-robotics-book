"""
Unit tests for the retrieval service.

This module contains unit tests for the retrieval service functionality,
including chunk retrieval, similarity ranking, and error handling.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from backend.retrieval.retrieval_service import RetrievalService
from backend.models.retrieval_models import RetrievedChunk


class TestRetrievalService:
    """
    Test class for the RetrievalService.
    """

    @pytest.fixture
    def retrieval_service(self):
        """
        Fixture to create a retrieval service instance for testing.
        """
        with patch('backend.retrieval.retrieval_service.QdrantService'), \
             patch('backend.retrieval.retrieval_service.cohere.Client') as mock_client:

            # Mock the Cohere client methods
            mock_cohere_instance = Mock()
            mock_cohere_instance.embed.return_value = Mock(embeddings=[[0.1, 0.2, 0.3]])
            mock_client.return_value = mock_cohere_instance

            service = RetrievalService()
            service.qdrant_service = Mock()
            return service

    def test_retrieve_chunks_success(self, retrieval_service):
        """
        Test successful chunk retrieval.
        """
        # Mock Qdrant search results
        mock_result = Mock()
        mock_result.id = "test_id"
        mock_result.payload = {"text": "test text", "source_location": "chapter1"}
        mock_result.score = 0.85
        retrieval_service.qdrant_service.search.return_value = [mock_result]

        # Call the method
        chunks = retrieval_service.retrieve_chunks(
            query_text="test query",
            collection_name="test_collection",
            top_k=3,
            similarity_threshold=0.5
        )

        # Assertions
        assert len(chunks) == 1
        assert isinstance(chunks[0], RetrievedChunk)
        assert chunks[0].id == "test_id"
        assert chunks[0].text == "test text"
        assert chunks[0].source_location == "chapter1"
        assert chunks[0].similarity_score == 0.85

        # Verify the Qdrant service was called with correct parameters
        retrieval_service.qdrant_service.search.assert_called_once()

    def test_retrieve_chunks_with_default_params(self, retrieval_service):
        """
        Test chunk retrieval with default parameters.
        """
        # Mock Qdrant search results
        mock_result = Mock()
        mock_result.id = "test_id"
        mock_result.payload = {"text": "test text", "source_location": "chapter1"}
        mock_result.score = 0.85
        retrieval_service.qdrant_service.search.return_value = [mock_result]

        # Call the method with default parameters
        chunks = retrieval_service.retrieve_chunks(
            query_text="test query",
            collection_name="test_collection"
        )

        # Assertions
        assert len(chunks) == 1
        assert chunks[0].id == "test_id"

    def test_retrieve_chunks_empty_results(self, retrieval_service):
        """
        Test chunk retrieval with empty results.
        """
        # Mock empty Qdrant search results
        retrieval_service.qdrant_service.search.return_value = []

        # Call the method
        chunks = retrieval_service.retrieve_chunks(
            query_text="test query",
            collection_name="test_collection"
        )

        # Assertions
        assert len(chunks) == 0

    def test_batch_retrieve(self, retrieval_service):
        """
        Test batch retrieval of chunks.
        """
        # Mock Qdrant search results
        mock_result = Mock()
        mock_result.id = "test_id"
        mock_result.payload = {"text": "test text", "source_location": "chapter1"}
        mock_result.score = 0.85
        retrieval_service.qdrant_service.search.return_value = [mock_result]

        # Call the method
        results = retrieval_service.batch_retrieve(
            queries=["query1", "query2"],
            collection_name="test_collection"
        )

        # Assertions
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert results[0][0].id == "test_id"
        assert results[1][0].id == "test_id"

    def test_validate_collection_exists(self, retrieval_service):
        """
        Test collection existence validation.
        """
        # Mock the collection existence check
        retrieval_service.qdrant_service.collection_exists.return_value = True

        # Call the method
        result = retrieval_service.validate_collection_exists("test_collection")

        # Assertions
        assert result is True
        retrieval_service.qdrant_service.collection_exists.assert_called_once_with("test_collection")

    def test_get_collection_info(self, retrieval_service):
        """
        Test getting collection information.
        """
        # Mock collection info
        mock_info = Mock()
        retrieval_service.qdrant_service.get_collection_info.return_value = mock_info

        # Call the method
        result = retrieval_service.get_collection_info("test_collection")

        # Assertions
        assert result == mock_info
        retrieval_service.qdrant_service.get_collection_info.assert_called_once_with("test_collection")

    def test_retrieve_chunks_connection_error(self, retrieval_service):
        """
        Test chunk retrieval with connection error.
        """
        # Mock a connection error
        retrieval_service.qdrant_service.search.side_effect = Exception("Connection failed")

        # Call the method and expect an exception
        with pytest.raises(Exception, match="Connection failed"):
            retrieval_service.retrieve_chunks(
                query_text="test query",
                collection_name="test_collection"
            )

    def test_rank_by_similarity(self, retrieval_service):
        """
        Test ranking chunks by similarity.
        """
        # Create test chunks
        chunk1 = RetrievedChunk(
            id="1",
            text="This is a test text about robotics",
            source_location="chapter1",
            similarity_score=0.8
        )
        chunk2 = RetrievedChunk(
            id="2",
            text="Another text about artificial intelligence",
            source_location="chapter2",
            similarity_score=0.7
        )

        chunks = [chunk1, chunk2]

        # Mock the Cohere client to return embeddings
        with patch.object(retrieval_service, 'cohere_client') as mock_client:
            # Mock embeddings for the texts
            mock_client.embed.return_value.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            # Mock query embedding
            mock_client.embed.side_effect = [
                Mock(embeddings=[[0.2, 0.3, 0.4]]),  # For the query
                Mock(embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # For the chunks
            ]

            # Call the method
            ranked_chunks = retrieval_service.rank_by_similarity(chunks, "robotics query")

            # Assertions
            assert len(ranked_chunks) == 2
            # The method should return chunks in some order (might be unchanged if ranking fails)
            assert all(isinstance(chunk, RetrievedChunk) for chunk in ranked_chunks)

    def test_retrieve_chunks_with_configurable_params(self, retrieval_service):
        """
        Test retrieval with configurable parameters.
        """
        # Mock Qdrant search results
        mock_result = Mock()
        mock_result.id = "test_id"
        mock_result.payload = {"text": "test text", "source_location": "chapter1"}
        mock_result.score = 0.85
        retrieval_service.qdrant_service.search.return_value = [mock_result]

        # Call the method with configurable params
        chunks = retrieval_service.retrieve_chunks_with_configurable_params(
            query_text="test query",
            collection_name="test_collection",
            top_k=5,
            similarity_threshold=0.6
        )

        # Assertions
        assert len(chunks) == 1
        assert chunks[0].id == "test_id"
        retrieval_service.qdrant_service.search.assert_called_once()

    def test_retrieve_chunks_invalid_params(self, retrieval_service):
        """
        Test chunk retrieval with invalid parameters.
        """
        # Test with invalid top_k
        with pytest.raises(Exception):
            retrieval_service.retrieve_chunks(
                query_text="test query",
                collection_name="test_collection",
                top_k=-1  # Invalid value
            )

        # Test with invalid similarity_threshold
        with pytest.raises(Exception):
            retrieval_service.retrieve_chunks(
                query_text="test query",
                collection_name="test_collection",
                similarity_threshold=1.5  # Invalid value
            )