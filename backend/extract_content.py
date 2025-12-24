"""
Content extraction module for the RAG chatbot backend.
This module provides functions to extract content from Docusaurus sites.
"""
import asyncio
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import logging

logger = logging.getLogger(__name__)

# Create a session for connection pooling
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})


async def get_all_urls(base_url: str) -> List[str]:
    """
    Get all URLs from the Docusaurus site.

    Args:
        base_url: The base URL of the Docusaurus site

    Returns:
        List of URLs found on the site
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_all_urls_sync, base_url)


def _get_all_urls_sync(base_url: str) -> List[str]:
    """
    Synchronous version of get_all_urls to run in thread pool.
    """
    urls = set()

    try:
        # Get the main page
        response = session.get(base_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links on the page
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)

            # Only add URLs that are part of the same domain
            if _is_valid_url(full_url, base_url):
                urls.add(full_url)

        # Also check for sitemap if it exists
        sitemap_url = urljoin(base_url, 'sitemap.xml')
        try:
            sitemap_response = session.get(sitemap_url)
            if sitemap_response.status_code == 200:
                sitemap_soup = BeautifulSoup(sitemap_response.text, 'xml')
                for loc in sitemap_soup.find_all('loc'):
                    url = loc.get_text().strip()
                    if _is_valid_url(url, base_url):
                        urls.add(url)
        except Exception as e:
            # Sitemap might not exist, that's OK
            logger.debug(f"Sitemap not found or error accessing it: {str(e)}")

        return list(urls)

    except Exception as e:
        logger.error(f"Error getting URLs from {base_url}: {str(e)}")
        raise


async def extract_text_from_url(url: str) -> Dict[str, Any]:
    """
    Extract clean text content from a single URL.

    Args:
        url: The URL to extract content from

    Returns:
        Dictionary with page_url, title, and cleaned content
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _extract_text_from_url_sync, url)


def _extract_text_from_url_sync(url: str) -> Dict[str, Any]:
    """
    Synchronous version of extract_text_from_url to run in thread pool.
    """
    try:
        response = session.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'meta']):
            element.decompose()

        # Try to find the main content area (common selectors for Docusaurus)
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_=re.compile(r'docItem|theme|markdown|container|content')) or
            soup.find('div', {'role': 'main'}) or
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

        # Additional title extraction from h1 tags if the title seems to be just the URL path
        if not title or title == urlparse(url).path.split('/')[-1] or ' ' not in title:
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text().strip()

        # Remove common navigation elements and repetitive text
        content = _clean_content(text)

        return {
            'page_url': url,
            'title': title,
            'content': content
        }

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        raise


def _is_valid_url(url: str, base_url: str) -> bool:
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
        # For this implementation, we'll check if the netloc is the same
        return url_parsed.netloc == base_parsed.netloc

    except Exception:
        return False


def _clean_content(content: str) -> str:
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
        r'Docusaurus\s+v\d+\.\d+\.\d+',
        r'Powered\s+by\s+Docusaurus',
        r'\s+«\s+.*?\s+»\s+',  # Navigation arrows
        r'\s+\d+\s+/\s+\d+\s+',  # Pagination
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)

    # Remove excessive whitespace again after cleaning
    content = re.sub(r'\s+', ' ', content).strip()

    return content


async def extract_book_content(base_url: str) -> List[Dict[str, Any]]:
    """
    Extract content from all pages of a Docusaurus book.

    Args:
        base_url: The base URL of the Docusaurus book

    Returns:
        List of dictionaries with page_url, title, and content
    """
    try:
        # Get all URLs from the site
        urls = await get_all_urls(base_url)

        # Extract content from each URL
        content_list = []
        for url in urls:
            try:
                content_data = await extract_text_from_url(url)
                content_list.append(content_data)
                logger.info(f"Extracted content from {url}")
            except Exception as e:
                logger.error(f"Failed to extract content from {url}: {str(e)}")
                continue  # Continue with other URLs even if one fails

        return content_list
    except Exception as e:
        logger.error(f"Error extracting book content from {base_url}: {str(e)}")
        raise


# Example usage if run as a script
if __name__ == "__main__":
    # Example of how to use these functions
    async def example():
        base_url = "https://your-docusaurus-site.com"
        urls = await get_all_urls(base_url)
        print(f"Found {len(urls)} URLs")

        # Process first few URLs as example
        for url in urls[:3]:  # Just first 3 as example
            content = await extract_text_from_url(url)
            print(f"Title: {content['title']}")
            print(f"Content length: {len(content['content'])} characters")
            print("---")

    # Run the example
    # asyncio.run(example())