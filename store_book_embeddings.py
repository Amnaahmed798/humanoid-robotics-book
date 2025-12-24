#!/usr/bin/env python3
"""
Script to generate embeddings for humanoid robotics book content and store them in Qdrant.
This will make the chatbot fully functional with real book content.
"""

import sys
import os
import asyncio
import glob
from typing import List
import re

# Add the backend directory to the Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_dir)

from dotenv import load_dotenv
from services.embedding_service import EmbeddingService
from models.embedding import EmbeddingVector
from models.chunk import TextChunk
from qdrant_operations import create_collection, save_chunks, validate_collection
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

def ensure_min_content_length(content_items: List[dict], min_chars: int = 1200) -> List[dict]:
    """Ensure all content items meet minimum length by combining smaller ones."""
    if not content_items:
        return []

    processed_items = []
    current_combined = {'title': 'Combined Content', 'content': '', 'source_files': []}

    for item in content_items:
        if len(item['content']) >= min_chars:
            # Item is already long enough
            processed_items.append(item)
        else:
            # Add to current combined content
            if current_combined['content']:
                current_combined['content'] += "\n\n" + item['content']
            else:
                current_combined['content'] = item['content']
            current_combined['source_files'].append(item['source_file'])

            # If combined content is now long enough, add it as an item
            if len(current_combined['content']) >= min_chars:
                combined_item = {
                    'title': 'Combined Content',
                    'content': current_combined['content'],
                    'source_file': ', '.join(current_combined['source_files'])
                }
                processed_items.append(combined_item)
                # Reset for next combination
                current_combined = {'title': 'Combined Content', 'content': '', 'source_files': []}

    # If there's remaining content that didn't reach the minimum, combine with last item if possible
    if current_combined['content'] and processed_items:
        last_item = processed_items[-1]
        if len(last_item['content']) + len(current_combined['content']) <= 3000:  # Don't make too large
            processed_items[-1]['content'] = last_item['content'] + "\n\n" + current_combined['content']
            if 'source_file' in processed_items[-1]:
                processed_items[-1]['source_file'] += ', ' + ', '.join(current_combined['source_files'])
            else:
                processed_items[-1]['source_file'] = ', '.join(current_combined['source_files'])
        else:
            # If it can't be combined, add as a separate item (may be too short but we'll try)
            combined_item = {
                'title': 'Combined Content',
                'content': current_combined['content'],
                'source_file': ', '.join(current_combined['source_files'])
            }
            processed_items.append(combined_item)
    elif current_combined['content']:
        # If there's no previous item to combine with, add as is
        combined_item = {
            'title': 'Combined Content',
            'content': current_combined['content'],
            'source_file': ', '.join(current_combined['source_files'])
        }
        processed_items.append(combined_item)

    return processed_items

def chunk_text_simple(text: str, max_chars: int = 2000) -> List[str]:
    """Simple text chunking that ensures minimum requirements are met."""
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Add the sentence to the current chunk
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence

        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            # If current chunk is substantial, save it
            if len(current_chunk) > 500:  # Reasonable minimum
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # If current chunk is too small but we need to split, just save current and start new
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

async def generate_and_store_embeddings():
    """Generate embeddings for book content and store them in Qdrant."""
    logger.info("Getting book content from specs directory...")
    content_items = get_book_content()

    if not content_items:
        logger.error("No content found in specs directory!")
        return False

    logger.info(f"Found {len(content_items)} content items")

    # Ensure all content items meet minimum length requirements
    processed_content_items = ensure_min_content_length(content_items, min_chars=1200)
    logger.info(f"Processed into {len(processed_content_items)} content items that meet minimum length requirements")

    # Prepare all text chunks with metadata
    all_chunks = []
    all_metadatas = []

    for item in processed_content_items:
        # Check if content is long enough
        if len(item['content']) < 1200:  # Too short even after combining
            logger.info(f"Content from {item['source_file']} is too short ({len(item['content'])} chars), skipping...")
            continue

        # For content that's already long enough, we can chunk it if it's very long
        if len(item['content']) > 2000:
            # Chunk very long content
            text_chunks = chunk_text_simple(item['content'], max_chars=2000)
        else:
            # For content that's just the right size, keep as one chunk
            text_chunks = [item['content']]

        for i, chunk_text in enumerate(text_chunks):
            # Skip chunks that are too short (less than 900 chars which is ~675 words)
            if len(chunk_text) < 900:
                logger.info(f"Chunk {i} from {item['source_file']} is too short ({len(chunk_text)} chars), skipping...")
                continue

            # Create a text chunk object with valid data
            chunk_id = f"{i:03d}_{hash(chunk_text) % 10000}"

            text_chunk = TextChunk(
                id=chunk_id,
                page_url=f"/book/{item['source_file'].replace('../', '').replace('./', '')}",
                title=item['title'],
                chunk_index=i,
                text=chunk_text.strip()
            )

            all_chunks.append(text_chunk)
            all_metadatas.append({
                'text': chunk_text,
                'title': item['title'],
                'source_file': item['source_file'],
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

    success = await create_collection(collection_name)
    if not success:
        logger.error(f"Failed to create collection '{collection_name}'")
        return False

    # Store the embeddings in Qdrant
    logger.info(f"Storing {len(valid_embeddings)} embeddings in Qdrant collection '{collection_name}'...")

    try:
        success = await save_chunks(valid_embeddings, collection_name, valid_metadatas)
        if success:
            logger.info(f"Successfully stored {len(valid_embeddings)} embeddings in Qdrant")
        else:
            logger.error("Failed to store embeddings in Qdrant")
            return False
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