import React, { useCallback } from 'react';
import analytics from './utils/analytics';

const Message = React.memo(({ message }) => {
  const { text, sender, sources, error, timestamp, confidence } = message;

  const handleSourceClick = useCallback((source, index) => {
    analytics.trackSourceClick(source.id, source.location || source.source_location);
  }, []);

  return (
    <div className={`message ${sender} ${error ? 'error' : ''}`}>
      <div className="message-text">
        {text.split('\n').map((line, i) => (
          <div key={i} dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              .replace(/\*(.*?)\*/g, '<em>$1</em>')
              .replace(/•/g, '• ')
              .replace(/^- (.*$)/gm, '<li>$1</li>')
          }} />
        ))}
      </div>
      {sources && sources.length > 0 && (
        <div className="message-sources">
          <h4>Sources:</h4>
          <ul>
            {sources.map((source, index) => (
              <li key={index}>
                <a
                  href={source.url || '#'}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={() => handleSourceClick(source, index)}
                >
                  {source.location || source.source_location} (Score: {((source.score || source.similarity_score) * 100).toFixed(1)}%)
                </a>
                <p className="source-content">{(source.content || source.text || '').substring(0, 100)}...</p>
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