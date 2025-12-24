import pytest
from chunk_text import chunk_text, chunk_with_heading_preservation, create_chunk_with_metadata, validate_chunk_size, merge_small_chunks
from models.chunk import TextChunk


def test_chunk_text_basic():
    """Test basic text chunking functionality"""
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = chunk_text(text, chunk_size=10, overlap=2)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert len(chunk.text) > 0


def test_chunk_text_with_overlap():
    """Test text chunking with overlap"""
    text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O."
    chunks = chunk_text(text, chunk_size=5, overlap=1)

    assert len(chunks) > 1  # Should create multiple chunks

    # Check that chunks have content
    for chunk in chunks:
        assert len(chunk.text) > 0


def test_create_chunk_with_metadata():
    """Test creating a chunk with metadata"""
    content = "This is a test chunk."
    page_url = "/docs/test"
    title = "Test Page"

    chunk = create_chunk_with_metadata(content, page_url, title)

    assert chunk.text == content
    assert chunk.page_url == page_url
    assert chunk.title == title
    assert chunk.chunk_index == 0  # Default value
    assert chunk.source_hash is not None  # Should be computed


def test_create_chunk_with_custom_index():
    """Test creating a chunk with custom index and heading"""
    content = "This is a test chunk."
    page_url = "/docs/test"
    title = "Test Page"
    heading = "Test Heading"
    chunk_index = 5

    chunk = create_chunk_with_metadata(content, page_url, title, heading, chunk_index)

    assert chunk.text == content
    assert chunk.page_url == page_url
    assert chunk.title == title
    assert chunk.heading == heading
    assert chunk.chunk_index == chunk_index


def test_validate_chunk_size_valid():
    """Test validating a chunk with valid size"""
    # Create a chunk with appropriate size
    chunk = TextChunk(
        id="test",
        page_url="/test",
        title="Test",
        chunk_index=0,
        text="This is a reasonably sized chunk of text that should pass validation. " * 5  # ~30 words, ~120 chars
    )

    is_valid = validate_chunk_size(chunk)
    assert is_valid  # Should be valid


def test_validate_chunk_size_too_small():
    """Test validating a chunk that's too small"""
    # Create a chunk that's too small
    chunk = TextChunk(
        id="test",
        page_url="/test",
        title="Test",
        chunk_index=0,
        text="Small."
    )

    is_valid = validate_chunk_size(chunk, min_tokens=10, max_tokens=100)
    assert not is_valid  # Should be invalid (too small)


def test_validate_chunk_size_too_large():
    """Test validating a chunk that's too large"""
    # Create a chunk that's too large
    large_text = "This is a very large chunk of text that exceeds the maximum allowed size. " * 50
    chunk = TextChunk(
        id="test",
        page_url="/test",
        title="Test",
        chunk_index=0,
        text=large_text
    )

    is_valid = validate_chunk_size(chunk, min_tokens=10, max_tokens=20)
    assert not is_valid  # Should be invalid (too large)


def test_merge_small_chunks():
    """Test merging small chunks together"""
    # Create several small chunks
    small_chunks = [
        TextChunk(id="1", page_url="/test", title="Test", chunk_index=0, text="First small chunk."),
        TextChunk(id="2", page_url="/test", title="Test", chunk_index=1, text="Second small chunk."),
        TextChunk(id="3", page_url="/test", title="Test", chunk_index=2, text="Third small chunk."),
    ]

    merged = merge_small_chunks(small_chunks, max_size=100)  # Small max size to force merging

    # Should have fewer chunks after merging
    assert len(merged) <= len(small_chunks)

    # Check that merged chunks have updated indices
    for i, chunk in enumerate(merged):
        assert chunk.chunk_index == i


def test_chunk_with_heading_preservation():
    """Test chunking while preserving heading context"""
    text = "Introduction content here. More introduction. Main topic starts here. More content about main topic."
    headings = [
        {"position": 0, "text": "Introduction"},
        {"position": 25, "text": "Main Topic"}
    ]

    chunks = chunk_with_heading_preservation(text, headings, chunk_size=10)

    # Should have chunks with heading context
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        # At least some chunks should have heading context
        if chunk.heading:
            assert chunk.heading in ["Introduction", "Main Topic"]


def test_chunk_empty_text():
    """Test chunking with empty text"""
    chunks = chunk_text("", chunk_size=10)
    assert len(chunks) == 0


def test_chunk_single_sentence():
    """Test chunking with a single sentence"""
    text = "This is a single sentence."
    chunks = chunk_text(text, chunk_size=10)

    assert len(chunks) == 1
    assert chunks[0].text == text