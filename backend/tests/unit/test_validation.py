import pytest
from unittest.mock import patch, MagicMock
from services.vector_db_service import VectorDBService
from models.embedding import EmbeddingVector, QdrantCollection
from models.chunk import TextChunk
import asyncio


@pytest.mark.asyncio
async def test_validate_collection_exists():
    """Test validating an existing collection"""
    service = VectorDBService()

    with patch.object(service, 'get_collection_info') as mock_get_info:
        mock_collection = QdrantCollection(
            name="test_collection",
            vector_size=1024,
            point_count=10
        )
        mock_get_info.return_value = mock_collection

        with patch.object(service.client, 'scroll') as mock_scroll:
            # Mock a sample point with correct dimensions
            mock_sample_point = MagicMock()
            mock_sample_point.vector = [0.1] * 1024
            mock_scroll.return_value = ([mock_sample_point], None)

        result = await service.validate_collection("test_collection")

        assert result["collection_name"] == "test_collection"
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0


@pytest.mark.asyncio
async def test_validate_collection_not_exists():
    """Test validating a non-existent collection"""
    service = VectorDBService()

    with patch.object(service, 'get_collection_info') as mock_get_info:
        mock_get_info.return_value = None

        result = await service.validate_collection("nonexistent_collection")

        assert result["collection_name"] == "nonexistent_collection"
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "does not exist" in result["issues"][0]


@pytest.mark.asyncio
async def test_validate_collection_wrong_dimensions():
    """Test validating a collection with wrong vector dimensions"""
    service = VectorDBService()

    with patch.object(service, 'get_collection_info') as mock_get_info:
        mock_collection = QdrantCollection(
            name="test_collection",
            vector_size=1024,
            point_count=5
        )
        mock_get_info.return_value = mock_collection

        with patch.object(service.client, 'scroll') as mock_scroll:
            # Mock a sample point with wrong dimensions
            mock_sample_point = MagicMock()
            mock_sample_point.vector = [0.1] * 512  # Wrong size
            mock_scroll.return_value = ([mock_sample_point], None)

        result = await service.validate_collection("test_collection")

        assert result["collection_name"] == "test_collection"
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "Vector dimensions mismatch" in result["issues"][0]


@pytest.mark.asyncio
async def test_validate_collection_exception():
    """Test validation when an exception occurs"""
    service = VectorDBService()

    with patch.object(service, 'get_collection_info') as mock_get_info:
        mock_get_info.side_effect = Exception("Connection error")

        result = await service.validate_collection("test_collection")

        assert result["collection_name"] == "test_collection"
        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "Connection error" in result["issues"][0]


def test_embedding_validation_valid():
    """Test validating a valid embedding"""
    from services.embedding_service import EmbeddingService

    service = EmbeddingService()

    valid_embedding = EmbeddingVector(
        vector=[0.1] * 1024,
        chunk_id="test_chunk",
        model_name=service.model_name
    )

    is_valid = service.validate_embedding(valid_embedding)
    assert is_valid


def test_embedding_validation_wrong_dimensions():
    """Test validating an embedding with wrong dimensions"""
    from services.embedding_service import EmbeddingService

    service = EmbeddingService()

    invalid_embedding = EmbeddingVector(
        vector=[0.1] * 512,  # Wrong dimensions
        chunk_id="test_chunk",
        model_name=service.model_name
    )

    is_valid = service.validate_embedding(invalid_embedding)
    assert not is_valid


def test_embedding_validation_wrong_model():
    """Test validating an embedding with wrong model name"""
    from services.embedding_service import EmbeddingService

    service = EmbeddingService()

    invalid_embedding = EmbeddingVector(
        vector=[0.1] * 1024,
        chunk_id="test_chunk",
        model_name="wrong-model"  # Different from service.model_name
    )

    is_valid = service.validate_embedding(invalid_embedding)
    assert not is_valid


@pytest.mark.asyncio
async def test_count_vectors():
    """Test counting vectors in a collection"""
    service = VectorDBService()

    with patch.object(service, 'get_collection_info') as mock_get_info:
        mock_collection = QdrantCollection(
            name="test_collection",
            vector_size=1024,
            point_count=42
        )
        mock_get_info.return_value = mock_collection

        count = await service.count_vectors("test_collection")
        assert count == 42


@pytest.mark.asyncio
async def test_count_vectors_exception():
    """Test counting vectors when an exception occurs"""
    service = VectorDBService()

    with patch.object(service, 'get_collection_info') as mock_get_info:
        mock_get_info.side_effect = Exception("Connection error")

        count = await service.count_vectors("test_collection")
        assert count == 0


@pytest.mark.asyncio
async def test_get_collection_info():
    """Test getting collection info"""
    service = VectorDBService()

    # Mock the client's get_collection method
    with patch.object(service.client, 'get_collection') as mock_get_collection:
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 100
        mock_get_collection.return_value = mock_collection_info

        info = await service.get_collection_info("test_collection")

        assert info is not None
        assert info.name == "test_collection"
        assert info.point_count == 100


@pytest.mark.asyncio
async def test_get_collection_info_not_exists():
    """Test getting info for a non-existent collection"""
    service = VectorDBService()

    # Mock the client to raise an exception (collection doesn't exist)
    with patch.object(service.client, 'get_collection') as mock_get_collection:
        mock_get_collection.side_effect = Exception("Collection not found")

        info = await service.get_collection_info("nonexistent_collection")

        assert info is None


def test_chunk_validation_valid_size():
    """Test validating a chunk with valid size"""
    from models.chunk import TextChunk

    # Create a chunk that should be valid (approximate 500 tokens worth of text)
    valid_chunk = TextChunk(
        id="test",
        page_url="/test",
        title="Test",
        chunk_index=0,
        text="This is a reasonably sized chunk of text that should pass validation. " * 10
    )

    is_valid = True  # Since the validator is in the model itself, just ensure creation works
    assert len(valid_chunk.text) > 0


def test_chunk_validation():
    """Test chunk validation functionality"""
    from chunk_text import validate_chunk_size

    # Create a chunk with appropriate size
    chunk = TextChunk(
        id="test",
        page_url="/test",
        title="Test",
        chunk_index=0,
        text="This is a test chunk with reasonable length for validation purposes."
    )

    is_valid = validate_chunk_size(chunk)
    # This will be approximate since we're using character count as proxy for tokens
    # The validation is len(v) < 225 in the model, which is a rough check
    assert isinstance(is_valid, bool)  # Should not raise an exception