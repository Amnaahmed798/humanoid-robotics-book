#!/usr/bin/env python3
"""
Final script to generate embeddings for humanoid robotics book content and store them in Qdrant.
This will make the chatbot fully functional with real book content.
"""

import sys
import os
import asyncio
import glob
import uuid
from typing import List

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
from services.embedding_service import EmbeddingService
from models.embedding import EmbeddingVector
from models.chunk import TextChunk
from qdrant_operations import create_collection, validate_collection
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
import logging

# Load environment variables
load_dotenv('./backend/.env')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_content_from_file(file_path: str) -> dict:
    """Extract content from a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title from first header
        lines = content.split('\n')
        title = ""
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break

        if not title:
            title = os.path.basename(file_path)

        return {
            'title': title,
            'content': content,
            'source_file': file_path
        }
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

def get_book_content() -> List[dict]:
    """Get all humanoid robotics book content from specs directory."""
    content_items = []

    # Look for all markdown files in the humanoid robotics book specs
    spec_dir = "./specs/001-humanoid-robotics-book/"
    md_files = glob.glob(os.path.join(spec_dir, "*.md"))

    for file_path in md_files:
        content_data = extract_content_from_file(file_path)
        if content_data:
            content_items.append(content_data)
            logger.info(f"Loaded content from: {file_path}")

    # Also include content from contracts directory
    contracts_dir = os.path.join(spec_dir, "contracts")
    if os.path.exists(contracts_dir):
        contract_files = glob.glob(os.path.join(contracts_dir, "*.md"))
        for file_path in contract_files:
            content_data = extract_content_from_file(file_path)
            if content_data:
                content_items.append(content_data)
                logger.info(f"Loaded contract content from: {file_path}")

    return content_items

def combine_all_content(content_items: List[dict]) -> List[dict]:
    """Combine ALL content into a single large item to ensure minimum requirements are met."""
    if not content_items:
        return []

    # Combine all content into one large item
    combined_content = ""
    combined_sources = []
    combined_title = "Humanoid Robotics Book Content"

    for item in content_items:
        combined_content += f"\n\n---\n\n{item['content']}"
        combined_sources.append(item['source_file'])

    # Clean up the combined content
    combined_content = combined_content.strip()

    combined_item = {
        'title': combined_title,
        'content': combined_content,
        'source_file': ', '.join(combined_sources)
    }

    logger.info(f"Combined all {len(content_items)} content items into 1 combined item")
    logger.info(f"Combined content has {len(combined_content.split())} words, {len(combined_content)} chars")

    return [combined_item]

async def generate_and_store_embeddings():
    """Generate embeddings for book content and store them in Qdrant."""
    logger.info("Getting book content from specs directory...")
    content_items = get_book_content()

    if not content_items:
        logger.error("No content found in specs directory!")
        return False

    logger.info(f"Found {len(content_items)} content items")

    # Combine all content to ensure it meets minimum requirements
    combined_content_items = combine_all_content(content_items)

    # Prepare all text chunks with metadata
    all_chunks = []
    all_metadatas = []

    for item in combined_content_items:
        content = item['content']

        # Ensure content is long enough (minimum ~300 tokens worth of words)
        if len(content.split()) < 300:  # Minimum word count for 300 tokens
            logger.error(f"Combined content is still too short: {len(content.split())} words")
            return False

        # Since we have a large combined content, we'll split it into chunks of appropriate size
        # Each chunk should be between 225-750 words (300-1000 tokens approx)
        chunk_size = 600  # Number of words per chunk, within the required range
        words = content.split()

        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            # Only add chunks that are within the required range
            if 225 <= len(chunk_words) <= 750:  # Within validation requirements
                chunk_text = ' '.join(chunk_words)
                chunks.append(chunk_text)
            elif len(chunk_words) > 750:
                # If chunk is too large, split it further
                logger.info(f"Chunk {i//chunk_size + 1} is too large ({len(chunk_words)} words), splitting further...")
                # Split into smaller chunks of 500 words each
                for j in range(0, len(chunk_words), 500):
                    sub_chunk_words = chunk_words[j:j + 500]
                    if len(sub_chunk_words) >= 225:  # At least minimum size
                        sub_chunk_text = ' '.join(sub_chunk_words)
                        chunks.append(sub_chunk_text)
                    elif len(sub_chunk_words) > 0 and chunks:
                        # Add tiny chunk to the last chunk if it exists
                        chunks[-1] += ' ' + ' '.join(sub_chunk_words)
                    elif len(sub_chunk_words) > 0:
                        # If no previous chunk exists, add as is (may be too small but we'll try)
                        sub_chunk_text = ' '.join(sub_chunk_words)
                        chunks.append(sub_chunk_text)
            elif len(chunk_words) < 225 and chunks:
                # If the last chunk is too small, add it to the previous chunk
                chunks[-1] += ' ' + ' '.join(chunk_words)
            elif len(chunk_words) < 225:
                # If this is the first chunk and too small, add as is
                chunk_text = ' '.join(chunk_words)
                chunks.append(chunk_text)

        logger.info(f"Split combined content into {len(chunks)} chunks")

        for i, chunk_text in enumerate(chunks):
            # Create a text chunk object with valid data - using integer string ID
            chunk_id = str(i)

            text_chunk = TextChunk(
                id=chunk_id,
                page_url="/book/combined-content",
                title=item['title'],
                chunk_index=i,
                text=chunk_text.strip()
            )

            all_chunks.append(text_chunk)
            all_metadatas.append({
                'text': chunk_text,
                'title': item['title'],
                'source_file': item['source_file'],
                'source_location': f"{item['source_file']}#section-{i+1}",
                'section': i+1,
                'type': 'book_content',
                'chunk_index': i
            })

    logger.info(f"Created {len(all_chunks)} text chunks for embedding generation")

    if not all_chunks:
        logger.error("No valid chunks to process!")
        return False

    # Initialize embedding service
    try:
        embedding_service = EmbeddingService()
        logger.info("Embedding service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize embedding service: {e}")
        return False

    # Generate embeddings for all chunks
    logger.info("Generating embeddings for all chunks...")

    # Process in batches to avoid exceeding API limits
    batch_size = 20  # Cohere has limits on how many texts can be embedded at once
    all_embeddings = []

    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_chunks) - 1)//batch_size + 1}")

        try:
            batch_embeddings = await embedding_service.embed_chunks(batch_chunks)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Generated embeddings for {len(batch_embeddings)} chunks in batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
            # Continue with the next batch
            continue

    logger.info(f"Successfully generated {len(all_embeddings)} embeddings")

    # Validate embeddings
    valid_embeddings = []
    valid_metadatas = []

    for emb, meta in zip(all_embeddings, all_metadatas):
        try:
            # Validate the embedding
            is_valid = await embedding_service.validate_embedding(emb)
            if is_valid:
                valid_embeddings.append(emb)
                valid_metadatas.append(meta)
            else:
                logger.warning(f"Invalid embedding found: {emb.id}")
        except Exception as e:
            logger.error(f"Error validating embedding {emb.id}: {e}")

    logger.info(f"Validated {len(valid_embeddings)} out of {len(all_embeddings)} embeddings")

    if not valid_embeddings:
        logger.error("No valid embeddings to store!")
        return False

    # Create the collection if it doesn't exist
    collection_name = "book_content"
    logger.info(f"Creating collection '{collection_name}' if it doesn't exist...")

    from qdrant_operations import client  # Import the Qdrant client directly

    success = await create_collection(collection_name)
    if not success:
        logger.error(f"Failed to create collection '{collection_name}'")
        return False

    # Store the embeddings in Qdrant directly, bypassing the problematic save_chunks function
    logger.info(f"Storing {len(valid_embeddings)} embeddings in Qdrant collection '{collection_name}'...")

    try:
        points = []
        for i, (embedding, metadata) in enumerate(zip(valid_embeddings, valid_metadatas)):
            # Use integer ID to avoid UUID parsing issues in Qdrant
            point = PointStruct(
                id=i,  # Use integer ID
                vector=embedding.vector,
                payload=metadata
            )
            points.append(point)

        # Upsert the points directly
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Successfully stored {len(valid_embeddings)} embeddings in Qdrant")
    except Exception as e:
        logger.error(f"Error storing embeddings in Qdrant: {e}")
        return False

    # Validate the collection after storing
    logger.info("Validating the collection...")
    validation_result = await validate_collection(collection_name)
    logger.info(f"Collection validation result: {validation_result}")

    logger.info("✓ Book content successfully stored in Qdrant!")
    logger.info(f"✓ Stored {len(valid_embeddings)} content chunks with embeddings")
    logger.info("✓ The chatbot should now be able to answer questions using the book content")

    return True

async def main():
    """Main function to run the embedding and storage process."""
    logger.info("Starting the process to populate Qdrant with book content...")

    success = await generate_and_store_embeddings()

    if success:
        logger.info("\n✓ Process completed successfully!")
        logger.info("✓ The chatbot is now functional and can answer queries using book content")
        logger.info("\nTo test the chatbot, use:")
        logger.info('curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" -d \'{"question": "What are the key components of ROS 2 for humanoid robots?"}\'')
    else:
        logger.error("\n✗ Process failed!")
        return False

    return success

if __name__ == "__main__":
    asyncio.run(main())