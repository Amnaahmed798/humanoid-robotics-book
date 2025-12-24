import React, { useState, useCallback, useMemo } from 'react';
import ChatWindow from './ChatWindow';
import InputArea from './InputArea';
import useChatAPI from './hooks/useChatAPI';
import useTextSelection from './hooks/useTextSelection';
import analytics from './utils/analytics';
import './chatbot.css';

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

  const { queryBackend, loading } = useChatAPI();
  const { getSelectedTextWithContext, clearSelection } = useTextSelection();

  const toggleChat = useCallback(() => {
    setIsOpen(prev => !prev);
  }, []);

  const handleInputChange = useCallback((e) => {
    setInputValue(e.target.value);
  }, []);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || loading) return;

    // Track the query event
    analytics.trackEvent('query_submitted', {
      query_length: inputValue.length,
      has_selection: !!getSelectedTextWithContext()
    });

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      type: 'query',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');

    try {
      // Check if there's selected text context
      const selectedContext = getSelectedTextWithContext();
      let context = {};

      if (selectedContext) {
        context = {
          selectedText: selectedContext.text,
          pageUrl: selectedContext.context.pageUrl,
          contextBefore: selectedContext.context.surroundingText.substring(0, 100),
          contextAfter: selectedContext.context.surroundingText.substring(100)
        };
        clearSelection(); // Clear the selection after using it
      }

      const response = await queryBackend(inputValue, context);

      // Track successful query
      analytics.trackQuery(inputValue, response);

      // Format the response to match our message structure
      const botMessage = {
        id: Date.now() + 1,
        text: response.answer,
        sender: 'bot',
        type: 'response',
        sources: response.sources || [],
        confidence: response.confidence,
        timestamp: response.timestamp || new Date().toISOString()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      // Track error
      analytics.trackError(error, 'query_backend');

      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error processing your request.",
        sender: 'bot',
        type: 'error',
        error: true,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [inputValue, loading, queryBackend, getSelectedTextWithContext, clearSelection]);

  // Memoize the messages array to prevent unnecessary re-renders of ChatWindow
  const memoizedMessages = useMemo(() => messages, [messages]);

  return (
    <div className="chatbot-container">
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <h3>Book Assistant</h3>
            <button className="chatbot-close" onClick={toggleChat}>
              ×
            </button>
          </div>
          <ChatWindow messages={memoizedMessages} isLoading={loading} />
          <InputArea
            inputValue={inputValue}
            onInputChange={handleInputChange}
            onSubmit={handleSubmit}
            isLoading={loading}
          />
        </div>
      ) : (
        <button className="chatbot-toggle" onClick={toggleChat}>
          💬 Ask Book
        </button>
      )}
    </div>
  );
};

export default Chatbot;