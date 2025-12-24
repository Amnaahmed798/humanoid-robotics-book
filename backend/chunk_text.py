"""
Text chunking module for the RAG chatbot backend.
This module provides functions to chunk text into appropriate sizes with metadata.
"""
import re
from typing import List, Dict, Any
from models.chunk import TextChunk
import hashlib
from datetime import datetime


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[TextChunk]:
    """
    Chunk text into segments with specified size and overlap.

    Args:
        text: The text to chunk
        chunk_size: Target size for text chunks in tokens (default 512)
        overlap: Overlap between chunks in tokens (default 50)

    Returns:
        List of TextChunk objects
    """
    # This is a simplified approach using character count as a proxy for token count
    # In a real implementation, we would use a proper tokenizer
    char_per_token_approx = 4  # Rough approximation: 1 token ~ 4 characters

    max_chunk_chars = chunk_size * char_per_token_approx
    overlap_chars = overlap * char_per_token_approx

    # Split text into sentences to maintain semantic boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        # Check if adding this sentence would exceed the chunk size
        if len(current_chunk) + len(sentence) <= max_chunk_chars:
            # Add sentence to current chunk
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            # Current chunk is full, save it
            if current_chunk.strip():
                chunk = TextChunk(
                    id=f"chunk_{chunk_index}",
                    page_url="",  # Will be set when associated with source
                    title="",  # Will be set when associated with source
                    chunk_index=chunk_index,
                    text=current_chunk.strip()
                )
                chunks.append(chunk)
                chunk_index += 1

            # Start new chunk with overlap from previous chunk
            if overlap_chars > 0 and len(current_chunk) > overlap_chars:
                # Add overlap from the end of the previous chunk
                overlap_text = current_chunk[-overlap_chars:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunk = TextChunk(
            id=f"chunk_{chunk_index}",
            page_url="",  # Will be set when associated with source
            title="",  # Will be set when associated with source
            chunk_index=chunk_index,
            text=current_chunk.strip()
        )
        chunks.append(chunk)

    return chunks


def chunk_with_heading_preservation(text: str, headings: List[Dict[str, Any]] = None, chunk_size: int = 512, overlap: int = 50) -> List[TextChunk]:
    """
    Chunk text while preserving heading context for each chunk.

    Args:
        text: The text to chunk
        headings: List of headings with their positions in the text
        chunk_size: Target size for text chunks in tokens (default 512)
        overlap: Overlap between chunks in tokens (default 50)

    Returns:
        List of TextChunk objects with heading context
    """
    char_per_token_approx = 4  # Rough approximation: 1 token ~ 4 characters
    max_chunk_chars = chunk_size * char_per_token_approx
    overlap_chars = overlap * char_per_token_approx

    # If no headings provided, just use the basic chunking
    if not headings:
        return chunk_text(text, chunk_size, overlap)

    # Sort headings by position
    sorted_headings = sorted(headings, key=lambda x: x['position'])

    chunks = []
    chunk_index = 0
    start_pos = 0

    for i, heading in enumerate(sorted_headings):
        heading_pos = heading['position']

        # Extract content from start_pos to heading_pos
        content = text[start_pos:heading_pos].strip()

        if content:
            # Chunk this content section
            content_chunks = chunk_text(content, chunk_size, overlap)

            # Update each chunk with the relevant heading
            for chunk in content_chunks:
                chunk.id = f"chunk_{chunk_index}"
                chunk.heading = heading['text']
                chunk.chunk_index = chunk_index
                chunks.append(chunk)
                chunk_index += 1

        # Update start position to the beginning of the next section
        # Find the end of this heading's content (next heading or end of text)
        next_start = heading_pos + len(heading['text'])
        next_pos = sorted_headings[i + 1]['position'] if i + 1 < len(sorted_headings) else len(text)

        # Include the content under this heading
        heading_content = text[next_start:next_pos].strip()

        if heading_content:
            # Check if the heading content alone is too large for one chunk
            if len(heading_content) > max_chunk_chars:
                # Chunk the heading content separately
                heading_content_chunks = chunk_text(heading_content, chunk_size, overlap)

                for chunk in heading_content_chunks:
                    chunk.id = f"chunk_{chunk_index}"
                    chunk.heading = heading['text']
                    chunk.chunk_index = chunk_index
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                # Create a single chunk with heading content
                chunk = TextChunk(
                    id=f"chunk_{chunk_index}",
                    page_url="",  # Will be set when associated with source
                    title="",  # Will be set when associated with source
                    heading=heading['text'],
                    chunk_index=chunk_index,
                    text=heading_content
                )
                chunks.append(chunk)
                chunk_index += 1

        start_pos = next_pos

    # Handle any remaining content after the last heading
    if start_pos < len(text):
        remaining_content = text[start_pos:].strip()
        if remaining_content:
            remaining_chunks = chunk_text(remaining_content, chunk_size, overlap)

            # Use the last heading for these chunks
            last_heading = sorted_headings[-1]['text'] if sorted_headings else ""

            for chunk in remaining_chunks:
                chunk.id = f"chunk_{chunk_index}"
                chunk.heading = last_heading
                chunk.chunk_index = chunk_index
                chunks.append(chunk)
                chunk_index += 1

    return chunks


def create_chunk_with_metadata(content: str, page_url: str, title: str, heading: str = None, chunk_index: int = 0) -> TextChunk:
    """
    Create a TextChunk with proper metadata.

    Args:
        content: The text content for the chunk
        page_url: The URL of the source page
        title: The title of the source page
        heading: The heading associated with this chunk (optional)
        chunk_index: The index of this chunk within the page

    Returns:
        TextChunk object with all metadata
    """
    # Create a hash of the content for change detection
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    chunk = TextChunk(
        id=f"chunk_{hash(content) % 1000000}",  # Simple ID based on content hash
        page_url=page_url,
        title=title,
        heading=heading,
        chunk_index=chunk_index,
        text=content,
        source_hash=content_hash,
        created_at=datetime.now()
    )

    return chunk


def validate_chunk_size(chunk: TextChunk, min_tokens: int = 300, max_tokens: int = 1000) -> bool:
    """
    Validate that a chunk's size is within the acceptable range.

    Args:
        chunk: The TextChunk to validate
        min_tokens: Minimum acceptable token count (default 300)
        max_tokens: Maximum acceptable token count (default 1000)

    Returns:
        True if chunk size is valid, False otherwise
    """
    # This is a rough approximation using character count
    char_per_token_approx = 4
    min_chars = min_tokens * char_per_token_approx
    max_chars = max_tokens * char_per_token_approx

    text_length = len(chunk.text)

    return min_chars <= text_length <= max_chars


def merge_small_chunks(chunks: List[TextChunk], max_size: int = 1000) -> List[TextChunk]:
    """
    Merge small chunks together to avoid having chunks that are too small.

    Args:
        chunks: List of TextChunk objects to process
        max_size: Maximum size for merged chunks (in characters, approximate)

    Returns:
        List of TextChunk objects with small chunks merged
    """
    if not chunks:
        return chunks

    merged = []
    current_chunk = chunks[0]

    for chunk in chunks[1:]:
        # Check if merging would exceed the size limit
        combined_text = current_chunk.text + " " + chunk.text
        if len(combined_text) <= max_size:
            # Merge the chunks
            current_chunk.text = combined_text
            # Update the ID to reflect that this is a merged chunk
            current_chunk.id = f"merged_{current_chunk.id}_{chunk.id}"
        else:
            # Keep the current chunk and start a new one
            merged.append(current_chunk)
            current_chunk = chunk

    # Add the last chunk
    merged.append(current_chunk)

    # Update chunk indices
    for i, chunk in enumerate(merged):
        chunk.chunk_index = i

    return merged


# Example usage if run as a script
if __name__ == "__main__":
    sample_text = """
    This is a sample text that we want to chunk. It contains multiple sentences.
    Each sentence should be properly handled. We want to make sure that the chunks
    are of appropriate size. This is important for the RAG system. The text continues
    with more content that should be chunked appropriately. We need to make sure that
    the semantic meaning is preserved when we chunk the text. This is the end of the sample text.
    """

    chunks = chunk_text(sample_text, chunk_size=30, overlap=5)

    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk.text)} chars - '{chunk.text[:50]}...'")