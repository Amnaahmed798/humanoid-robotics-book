import React from 'react';
import Message from './Message';

const ChatWindow = React.memo(({ messages, isLoading }) => {
  return (
    <div className="chatbot-messages">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      {isLoading && (
        <div className="message bot">
          <div className="message-text loading">Thinking...</div>
        </div>
      )}
    </div>
  );
});

export default ChatWindow;