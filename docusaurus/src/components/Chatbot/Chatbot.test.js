import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Chatbot from './Chatbot';

// Mock the hooks and API
jest.mock('./hooks/useChatAPI', () => ({
  __esModule: true,
  default: () => ({
    queryBackend: jest.fn().mockResolvedValue({
      answer: 'Test response',
      sources: [{ id: '1', content: 'Test content', location: 'Chapter 1', url: '/chapter1', score: 0.9 }],
      confidence: 0.9,
      timestamp: new Date().toISOString()
    }),
    loading: false,
    error: null
  })
}));

jest.mock('./hooks/useTextSelection', () => ({
  __esModule: true,
  default: () => ({
    selectedText: '',
    selectionContext: { pageUrl: '', surroundingText: '', selectionStart: 0, selectionEnd: 0 },
    getSelectedTextWithContext: jest.fn().mockReturnValue(null),
    clearSelection: jest.fn()
  })
}));

describe('Chatbot Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders chatbot toggle button initially', () => {
    render(<Chatbot />);
    expect(screen.getByText('💬 Ask Book')).toBeInTheDocument();
  });

  test('opens chat window when toggle button is clicked', () => {
    render(<Chatbot />);
    const toggleButton = screen.getByText('💬 Ask Book');
    fireEvent.click(toggleButton);

    expect(screen.getByText('Book Assistant')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Ask about the book content...')).toBeInTheDocument();
  });

  test('submits a question and displays response', async () => {
    render(<Chatbot />);

    // Open the chat
    const toggleButton = screen.getByText('💬 Ask Book');
    fireEvent.click(toggleButton);

    // Type a question
    const input = screen.getByPlaceholderText('Ask about the book content...');
    fireEvent.change(input, { target: { value: 'What is humanoid robotics?' } });

    // Submit the question
    const submitButton = screen.getByText('Send');
    fireEvent.click(submitButton);

    // Wait for the response
    await waitFor(() => {
      expect(screen.getByText('Test response')).toBeInTheDocument();
    });
  });

  test('displays sources in the response', async () => {
    render(<Chatbot />);

    // Open the chat
    const toggleButton = screen.getByText('💬 Ask Book');
    fireEvent.click(toggleButton);

    // Type a question
    const input = screen.getByPlaceholderText('Ask about the book content...');
    fireEvent.change(input, { target: { value: 'What is humanoid robotics?' } });

    // Submit the question
    const submitButton = screen.getByText('Send');
    fireEvent.click(submitButton);

    // Wait for the response with sources
    await waitFor(() => {
      expect(screen.getByText('Chapter 1 (Score: 90.0%)')).toBeInTheDocument();
    });
  });

  test('shows loading state during API request', async () => {
    // Create a mock that resolves after a delay
    const mockQueryBackend = jest.fn(
      () => new Promise(resolve =>
        setTimeout(() => resolve({
          answer: 'Delayed response',
          sources: [],
          confidence: 0.8,
          timestamp: new Date().toISOString()
        }), 100)
      )
    );

    jest.mock('./hooks/useChatAPI', () => ({
      __esModule: true,
      default: () => ({
        queryBackend: mockQueryBackend,
        loading: false, // Will be updated by the hook during the call
        error: null
      })
    }));

    render(<Chatbot />);

    // Open the chat
    const toggleButton = screen.getByText('💬 Ask Book');
    fireEvent.click(toggleButton);

    // Type a question
    const input = screen.getByPlaceholderText('Ask about the book content...');
    fireEvent.change(input, { target: { value: 'Delayed question' } });

    // Submit the question
    const submitButton = screen.getByText('Send');
    fireEvent.click(submitButton);

    // Check that loading state is shown (this would require more complex mocking to fully test)
    expect(submitButton).toBeDisabled();
  });
});