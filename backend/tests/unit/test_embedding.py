import pytest
from unittest.mock import patch, MagicMock
from generate_embeddings import embed, embed_texts, embed_chunks, embed_query, validate_embedding_vector, get_embedding_model_info
from models.embedding import EmbeddingVector
from models.chunk import TextChunk


@pytest.mark.asyncio
async def test_embed_single_text():
    """Test embedding generation for a single text"""
    text = "This is a test sentence."

    with patch('generate_embeddings._embed_sync') as mock_sync_func:
        mock_embedding = EmbeddingVector(
            vector=[0.1] * 1024,  # 1024-dimensional vector
            chunk_id="test_chunk",
            model_name="embed-multilingual-v3.0"
        )
        mock_sync_func.return_value = mock_embedding

        result = await embed(text)

        assert result == mock_embedding
        mock_sync_func.assert_called_once_with(text)


@pytest.mark.asyncio
async def test_embed_texts_multiple():
    """Test embedding generation for multiple texts"""
    texts = ["Text one", "Text two", "Text three"]

    with patch('generate_embeddings._embed_texts_sync') as mock_sync_func:
        mock_embeddings = [
            EmbeddingVector(vector=[0.1] * 1024, chunk_id="text_0", model_name="embed-multilingual-v3.0"),
            EmbeddingVector(vector=[0.2] * 1024, chunk_id="text_1", model_name="embed-multilingual-v3.0"),
            EmbeddingVector(vector=[0.3] * 1024, chunk_id="text_2", model_name="embed-multilingual-v3.0")
        ]
        mock_sync_func.return_value = mock_embeddings

        result = await embed_texts(texts)

        assert result == mock_embeddings
        assert len(result) == 3
        mock_sync_func.assert_called_once_with(texts)


@pytest.mark.asyncio
async def test_embed_texts_empty_list():
    """Test embedding generation for empty list"""
    result = await embed_texts([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_chunks():
    """Test embedding generation for text chunks"""
    chunks = [
        TextChunk(id="chunk1", page_url="/test", title="Test", chunk_index=0, text="First chunk"),
        TextChunk(id="chunk2", page_url="/test", title="Test", chunk_index=1, text="Second chunk")
    ]

    with patch('generate_embeddings._embed_chunks_sync') as mock_sync_func:
        mock_embeddings = [
            EmbeddingVector(vector=[0.1] * 1024, chunk_id="chunk1", model_name="embed-multilingual-v3.0"),
            EmbeddingVector(vector=[0.2] * 1024, chunk_id="chunk2", model_name="embed-multilingual-v3.0")
        ]
        mock_sync_func.return_value = mock_embeddings

        result = await embed_chunks(chunks)

        assert result == mock_embeddings
        assert len(result) == 2


@pytest.mark.asyncio
async def test_embed_query():
    """Test embedding generation for a search query"""
    query = "What is a humanoid robot?"

    with patch('generate_embeddings._embed_query_sync') as mock_sync_func:
        mock_embedding = EmbeddingVector(
            vector=[0.5] * 1024,
            chunk_id="",  # Queries don't have chunk IDs
            model_name="embed-multilingual-v3.0"
        )
        mock_sync_func.return_value = mock_embedding

        result = await embed_query(query)

        assert result == mock_embedding
        mock_sync_func.assert_called_once_with(query)


def test_validate_embedding_vector_valid():
    """Test validating a valid embedding vector"""
    valid_embedding = EmbeddingVector(
        vector=[0.1] * 1024,  # Correct dimensionality
        chunk_id="test_chunk",
        model_name="embed-multilingual-v3.0"
    )

    is_valid = validate_embedding_vector(valid_embedding)
    assert is_valid


def test_validate_embedding_vector_wrong_dimension():
    """Test validating an embedding with wrong dimensionality"""
    invalid_embedding = EmbeddingVector(
        vector=[0.1] * 512,  # Wrong dimensionality
        chunk_id="test_chunk",
        model_name="embed-multilingual-v3.0"
    )

    is_valid = validate_embedding_vector(invalid_embedding)
    assert not is_valid


def test_validate_embedding_vector_nan_values():
    """Test validating an embedding with NaN values"""
    import math
    invalid_embedding = EmbeddingVector(
        vector=[math.nan] + [0.1] * 1023,  # Contains NaN
        chunk_id="test_chunk",
        model_name="embed-multilingual-v3.0"
    )

    is_valid = validate_embedding_vector(invalid_embedding)
    assert not is_valid


def test_validate_embedding_vector_infinity_values():
    """Test validating an embedding with infinity values"""
    import math
    invalid_embedding = EmbeddingVector(
        vector=[math.inf] + [0.1] * 1023,  # Contains infinity
        chunk_id="test_chunk",
        model_name="embed-multilingual-v3.0"
    )

    is_valid = validate_embedding_vector(invalid_embedding)
    assert not is_valid


def test_get_embedding_model_info():
    """Test getting embedding model information"""
    info = get_embedding_model_info()

    assert "model_name" in info
    assert "dimensions" in info
    assert "description" in info

    assert info["model_name"] == "embed-multilingual-v3.0"
    assert info["dimensions"] == 1024


@pytest.mark.asyncio
async def test_embed_exception_handling():
    """Test exception handling in embed function"""
    with patch('generate_embeddings._embed_sync') as mock_sync_func:
        mock_sync_func.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await embed("test text")


@pytest.mark.asyncio
async def test_embed_texts_exception_handling():
    """Test exception handling in embed_texts function"""
    with patch('generate_embeddings._embed_texts_sync') as mock_sync_func:
        mock_sync_func.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            await embed_texts(["text1", "text2"])