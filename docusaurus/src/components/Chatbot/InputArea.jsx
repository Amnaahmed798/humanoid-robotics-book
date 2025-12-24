import React from 'react';

const InputArea = React.memo(({ inputValue, onInputChange, onSubmit, isLoading }) => {
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(e);
  };

  return (
    <form className="chatbot-input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        value={inputValue}
        onChange={onInputChange}
        placeholder="Ask about the book content..."
        disabled={isLoading}
      />
      <button type="submit" disabled={isLoading}>
        Send
      </button>
    </form>
  );
});

export default InputArea;