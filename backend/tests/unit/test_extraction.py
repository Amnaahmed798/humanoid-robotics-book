import pytest
from unittest.mock import patch, MagicMock
from extract_content import get_all_urls, extract_text_from_url, _clean_content
import asyncio


@pytest.mark.asyncio
async def test_get_all_urls():
    """Test getting all URLs from a base URL"""
    base_url = "https://example.com"

    with patch('extract_content._get_all_urls_sync') as mock_sync_func:
        mock_sync_func.return_value = ["https://example.com/page1", "https://example.com/page2"]

        result = await get_all_urls(base_url)

        assert result == ["https://example.com/page1", "https://example.com/page2"]
        mock_sync_func.assert_called_once_with(base_url)


@pytest.mark.asyncio
async def test_extract_text_from_url():
    """Test extracting text from a URL"""
    url = "https://example.com/page"

    with patch('extract_content._extract_text_from_url_sync') as mock_sync_func:
        mock_result = {
            'page_url': url,
            'title': 'Test Page',
            'content': 'This is the page content.'
        }
        mock_sync_func.return_value = mock_result

        result = await extract_text_from_url(url)

        assert result == mock_result
        mock_sync_func.assert_called_once_with(url)


def test_clean_content():
    """Test content cleaning functionality"""
    raw_content = "This is some raw content with Previous Next navigation elements. Was this page helpful? Edit this page Copyright 2023."

    cleaned = _clean_content(raw_content)

    # Check that common navigation elements are removed
    assert "Previous Next" not in cleaned
    assert "Was this page helpful" not in cleaned
    assert "Edit this page" not in cleaned
    assert "Copyright 2023" not in cleaned

    # Check that meaningful content remains
    assert "This is some raw content" in cleaned


def test_is_valid_url():
    """Test URL validation"""
    from extract_content import _is_valid_url

    # Test with same domain
    assert _is_valid_url("https://example.com/page", "https://example.com") == True

    # Test with different domain
    assert _is_valid_url("https://other.com/page", "https://example.com") == False

    # Test with invalid scheme
    assert _is_valid_url("ftp://example.com/page", "https://example.com") == False

    # Test with subdomain
    assert _is_valid_url("https://docs.example.com/page", "https://example.com") == False  # Our implementation is strict


@pytest.mark.asyncio
async def test_extract_text_from_url_exception():
    """Test extract_text_from_url when an exception occurs"""
    url = "https://example.com/bad-page"

    with patch('extract_content._extract_text_from_url_sync') as mock_sync_func:
        mock_sync_func.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            await extract_text_from_url(url)


@pytest.mark.asyncio
async def test_get_all_urls_exception():
    """Test get_all_urls when an exception occurs"""
    base_url = "https://example.com/bad-site"

    with patch('extract_content._get_all_urls_sync') as mock_sync_func:
        mock_sync_func.side_effect = Exception("Network error")

        with pytest.raises(Exception):
            await get_all_urls(base_url)