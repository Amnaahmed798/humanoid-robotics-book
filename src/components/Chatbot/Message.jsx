import React, { useCallback } from 'react';
import analytics from './utils/analytics';

const Message = React.memo(({ message }) => {
  const { text, sender, sources, error, timestamp, confidence } = message;

  const handleSourceClick = useCallback((source, index) => {
    analytics.trackSourceClick(source.id, source.location);
  }, []);

  return (
    <div className={`message ${sender} ${error ? 'error' : ''}`}>
      <div className="message-text">{text}</div>
      {sources && sources.length > 0 && (
    <div className="message-sources">
      <h4>Sources:</h4>
      <ul>
        {sources.map((source, index) => (──────────────────────────────────────────────────────────────────────────────────────────
          <li key={index}>
            <a
              href={source.url || "#"}  // Use fallback if url doesn't exist
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => handleSourceClick(source, index)}
            >
              {source.source_location || source.location} (Score: {((source.similarity_score || source.score) * 100).toFixed(1)}%)
            </a>
            <p className="source-content">{(source.text || source.content || '').substring(0, 100)}...</p>
          </li>
        ))}
      </ul>
    </div>
  )}
      {confidence !== undefined && (
        <div className="message-confidence">
          Confidence: {(confidence * 100).toFixed(1)}%
        </div>
      )}
      <div className="message-timestamp">
        {new Date(timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
});

export default Message;