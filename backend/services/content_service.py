from typing import List, Dict, Any, Optional
from models.chunk import TextChunk
import requests
from bs4 import BeautifulSoup
import asyncio
import re
from urllib.parse import urljoin, urlparse
import logging

logger = logging.getLogger(__name__)


class ContentService:
    """
    Service for content processing including extraction from Docusaurus sites.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    async def get_all_urls(self, base_url: str) -> List[str]:
        """
        Get all URLs from the Docusaurus site.

        Args:
            base_url: The base URL of the Docusaurus site

        Returns:
            List of URLs found on the site
        """
        urls = set()

        try:
            # Get the main page
            response = self.session.get(base_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(base_url, href)

                # Only add URLs that are part of the same domain
                if self._is_valid_url(full_url, base_url):
                    urls.add(full_url)

            # Also check for sitemap if it exists
            sitemap_url = urljoin(base_url, 'sitemap.xml')
            try:
                sitemap_response = self.session.get(sitemap_url)
                if sitemap_response.status_code == 200:
                    sitemap_soup = BeautifulSoup(sitemap_response.text, 'xml')
                    for loc in sitemap_soup.find_all('loc'):
                        url = loc.get_text().strip()
                        if self._is_valid_url(url, base_url):
                            urls.add(url)
            except:
                # Sitemap might not exist, that's OK
                pass

            return list(urls)

        except Exception as e:
            logger.error(f"Error getting URLs from {base_url}: {str(e)}")
            raise

    async def extract_text_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract clean text content from a single URL.

        Args:
            url: The URL to extract content from

        Returns:
            Dictionary with page_url, title, and cleaned content
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()

            # Try to find the main content area (common selectors for Docusaurus)
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_=re.compile(r'container|docItem|theme|markdown')) or
                soup.body
            )

            if not main_content:
                main_content = soup

            # Extract text, removing extra whitespace
            text = main_content.get_text(separator=' ', strip=True)

            # Clean up excessive whitespace
            text = re.sub(r'\s+', ' ', text)

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else urlparse(url).path.split('/')[-1] or 'Untitled'

            # Remove common navigation elements and repetitive text
            # This is a simplified approach - in a real implementation,
            # we might want more sophisticated content extraction
            content = self._clean_content(text)

            return {
                'page_url': url,
                'title': title,
                'content': content
            }

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            raise

    def _is_valid_url(self, url: str, base_url: str) -> bool:
        """
        Check if a URL is valid and belongs to the same domain as the base URL.

        Args:
            url: The URL to check
            base_url: The base URL for comparison

        Returns:
            True if the URL is valid and in the same domain, False otherwise
        """
        try:
            url_parsed = urlparse(url)
            base_parsed = urlparse(base_url)

            # Check if it's an HTTP/HTTPS URL
            if url_parsed.scheme not in ['http', 'https']:
                return False

            # Check if it's in the same domain (or subdomain)
            # For this implementation, we'll be more permissive and just check if it's related to the base
            return url_parsed.netloc == base_parsed.netloc

        except Exception:
            return False

    def _clean_content(self, content: str) -> str:
        """
        Clean the extracted content by removing common artifacts.

        Args:
            content: Raw extracted content

        Returns:
            Cleaned content
        """
        # Remove common navigation elements that might remain
        patterns_to_remove = [
            r'Previous\s+Next',
            r'Was\s+this\s+page\s+helpful\?',
            r'\s+\d+\s+stars\s+.*?Submit/',
            r'Edit\s+this\s+page',
            r'Copyright\s+\d{4}.*?(?=\n|$)',
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Remove excessive whitespace again after cleaning
        content = re.sub(r'\s+', ' ', content).strip()

        return content

    async def extract_and_chunk_content(self, base_url: str, chunk_size: int = 512) -> List[TextChunk]:
        """
        Extract content from all pages and chunk it.

        Args:
            base_url: The base URL of the Docusaurus site
            chunk_size: Target size for text chunks in tokens (default 512)

        Returns:
            List of TextChunk objects
        """
        # Get all URLs
        urls = await self.get_all_urls(base_url)

        all_chunks = []
        chunk_id_counter = 0

        for url in urls:
            try:
                # Extract content from the URL
                content_data = await self.extract_text_from_url(url)

                # Chunk the content
                content = content_data['content']
                title = content_data['title']

                # Simple chunking by sentence (in a real implementation,
                # we would use proper tokenization)
                chunks = self._chunk_text(content, chunk_size)

                for i, chunk_text in enumerate(chunks):
                    chunk = TextChunk(
                        id=f"chunk_{chunk_id_counter}",
                        page_url=content_data['page_url'],
                        title=title,
                        chunk_index=i,
                        text=chunk_text
                    )
                    all_chunks.append(chunk)
                    chunk_id_counter += 1

            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                # Continue with other URLs even if one fails

        return all_chunks

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Simple text chunking by sentences.

        Args:
            text: Text to chunk
            chunk_size: Target size for chunks (approximate)

        Returns:
            List of text chunks
        """
        # This is a simplified approach - in a real implementation, we would use
        # proper tokenization to ensure chunks are exactly the right token size

        # Split by sentences
        sentences = re.split(r'[.!?]+\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Add some buffer for the approximation
            if len(current_chunk) + len(sentence) < chunk_size * 4:  # Rough char approximation
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If any chunk is too small, try to merge with the next one
        merged_chunks = []
        i = 0
        while i < len(chunks):
            current = chunks[i]
            if len(current) < chunk_size * 2 and i + 1 < len(chunks):  # If too small and not the last
                next_chunk = chunks[i + 1]
                if len(current) + len(next_chunk) < chunk_size * 6:  # If merging keeps it reasonable
                    current = current + " " + next_chunk
                    i += 1  # Skip the next chunk since we merged it
            merged_chunks.append(current)
            i += 1

        return merged_chunks