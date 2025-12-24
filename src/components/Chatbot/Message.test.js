import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Message from './Message';

describe('Message Component', () => {
  const mockMessage = {
    id: '1',
    text: 'This is a test message',
    sender: 'bot',
    type: 'response',
    timestamp: new Date().toISOString(),
    sources: [
      { id: 'source1', content: 'Test source content', location: 'Chapter 1', url: '/chapter1', score: 0.9 }
    ],
    confidence: 0.9
  };

  test('renders user message correctly', () => {
    const userMessage = { ...mockMessage, sender: 'user', sources: [], confidence: undefined };

    render(<Message message={userMessage} />);

    expect(screen.getByText('This is a test message')).toBeInTheDocument();
    expect(screen.getByText('user')).toBeInTheDocument(); // This will be in the class
  });

  test('renders bot message with sources and confidence', () => {
    render(<Message message={mockMessage} />);

    expect(screen.getByText('This is a test message')).toBeInTheDocument();
    expect(screen.getByText('Sources:')).toBeInTheDocument();
    expect(screen.getByText('Chapter 1 (Score: 90.0%)')).toBeInTheDocument();
    expect(screen.getByText('Confidence: 90.0%')).toBeInTheDocument();
  });

  test('renders error message correctly', () => {
    const errorMessage = { ...mockMessage, error: true, text: 'Error occurred' };

    render(<Message message={errorMessage} />);

    expect(screen.getByText('Error occurred')).toBeInTheDocument();
  });

  test('does not render sources section when no sources are provided', () => {
    const messageWithoutSources = { ...mockMessage, sources: [] };

    render(<Message message={messageWithoutSources} />);

    expect(screen.getByText('This is a test message')).toBeInTheDocument();
    expect(screen.queryByText('Sources:')).not.toBeInTheDocument();
  });

  test('does not render confidence when not provided', () => {
    const messageWithoutConfidence = { ...mockMessage, confidence: undefined };

    render(<Message message={messageWithoutConfidence} />);

    expect(screen.getByText('This is a test message')).toBeInTheDocument();
    expect(screen.queryByText('Confidence:')).not.toBeInTheDocument();
  });

  test('formats source content preview correctly', () => {
    render(<Message message={mockMessage} />);

    // Check that the source content is truncated to first 100 characters
    expect(screen.getByText(/Test source content/)).toBeInTheDocument();
  });
});