from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime
import re


class TextChunk(BaseModel):
    """
    A segment of book content with associated metadata for RAG operations.
    """
    id: Optional[str] = None
    page_url: str
    title: str
    heading: Optional[str] = None
    chunk_index: int
    text: str
    created_at: datetime = datetime.now()
    source_hash: Optional[str] = None

    class Config:
        # Allow extra fields during development
        extra = "allow"

    @validator('text')
    def validate_text_length(cls, v):
        """Validate that text is between 300-1000 tokens (approximately 225-750 words)"""
        # This is a rough approximation - in a real implementation,
        # we would count actual tokens using a tokenizer
        word_count = len(v.split())
        if word_count < 225:  # Approximate lower bound for 300 tokens
            raise ValueError('Text is too short, must be at least ~300 tokens')
        if word_count > 750:  # Approximate upper bound for 1000 tokens
            raise ValueError('Text is too long, must be at most ~1000 tokens')
        return v

    @validator('page_url')
    def validate_page_url(cls, v):
        """Validate that page_url is a valid URL path"""
        if not v.startswith('/'):
            # If it's not a relative path, check if it's a valid URL
            if not re.match(r'^https?://', v):
                raise ValueError('page_url must be a valid URL or relative path starting with /')
        return v

    @validator('chunk_index')
    def validate_chunk_index(cls, v):
        """Validate that chunk_index is non-negative"""
        if v < 0:
            raise ValueError('chunk_index must be non-negative')
        return v

    def __str__(self):
        return f"TextChunk(id={self.id}, page_url={self.page_url}, chunk_index={self.chunk_index})"

    def __repr__(self):
        return self.__str__()


class SearchChunk(BaseModel):
    """
    A text chunk specifically for search results, with additional metadata.
    """
    id: str
    text: str
    score: float  # Similarity score from vector search
    metadata: dict

    @validator('score')
    def validate_score(cls, v):
        """Validate that score is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v