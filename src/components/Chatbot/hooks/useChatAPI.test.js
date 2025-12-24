import { renderHook, act } from '@testing-library/react';
import useChatAPI from './useChatAPI';

// Mock fetch
global.fetch = jest.fn();

describe('useChatAPI Hook', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  test('should initialize with correct default values', () => {
    const { result } = renderHook(() => useChatAPI());

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  test('should set loading state when querying backend', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ answer: 'Test answer', sources: [], confidence: 0.9, timestamp: new Date().toISOString() })
    });

    const { result } = renderHook(() => useChatAPI());

    await act(async () => {
      // Check initial state
      expect(result.current.loading).toBe(false);

      // Call the API function
      const promise = result.current.queryBackend('Test question');

      // Check loading state during the call
      expect(result.current.loading).toBe(true);

      // Wait for the call to complete
      await promise;
    });

    // Check final state
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  test('should handle successful API response', async () => {
    const mockResponse = {
      answer: 'This is a test answer',
      sources: [
        { id: '1', content: 'Test source content', location: 'Chapter 1', url: '/chapter1', score: 0.9 }
      ],
      confidence: 0.9,
      timestamp: new Date().toISOString()
    };

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse
    });

    const { result } = renderHook(() => useChatAPI());

    let response;
    await act(async () => {
      response = await result.current.queryBackend('Test question');
    });

    expect(response).toEqual(mockResponse);
    expect(result.current.error).toBeNull();
  });

  test('should handle API error', async () => {
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error'
    });

    const { result } = renderHook(() => useChatAPI());

    await act(async () => {
      try {
        await result.current.queryBackend('Test question');
      } catch (error) {
        // Expected to throw
      }
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).not.toBeNull();
  });

  test('should make correct API request', async () => {
    const mockResponse = {
      answer: 'Test answer',
      sources: [],
      confidence: 0.8,
      timestamp: new Date().toISOString()
    };

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse
    });

    const { result } = renderHook(() => useChatAPI());

    const question = 'What is humanoid robotics?';
    const context = { selectedText: 'selected text', pageUrl: '/test-page' };

    await act(async () => {
      await result.current.queryBackend(question, context);
    });

    expect(fetch).toHaveBeenCalledWith(
      `${process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'}/api/v1/query`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          context,
          top_k: 3
        }),
      }
    );
  });
});