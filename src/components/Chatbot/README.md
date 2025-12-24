# Chatbot Component Documentation

## Overview
The Chatbot component is an interactive assistant that allows users to ask questions about the humanoid robotics book content. It integrates with a backend RAG (Retrieval Augmented Generation) system to provide accurate, source-backed answers.

## Features
- Floating UI that can be toggled open/closed
- Natural language question input
- Source citation with clickable links to original content
- Confidence scoring for responses
- Selected text integration - users can select text on the page and ask questions about it
- Loading states and error handling
- Analytics tracking for usage patterns

## Installation
The component is built as a standalone React component that can be integrated into any React application, specifically designed for Docusaurus sites.

### Prerequisites
- React 16.8+ (for hooks)
- Environment variable: `REACT_APP_BACKEND_URL` pointing to your backend API

### Integration
```jsx
import Chatbot from './components/Chatbot/Chatbot';

function App() {
  return (
    <div className="App">
      {/* Your app content */}
      <Chatbot />
    </div>
  );
}
```

## API Integration
The component communicates with the backend through the `/api/v1/query` endpoint:

### Request Format
```json
{
  "question": "Your question here",
  "context": {
    "selectedText": "Text user selected on the page",
    "pageUrl": "Current page URL",
    "contextBefore": "Text before selection",
    "contextAfter": "Text after selection"
  },
  "top_k": 3
}
```

### Response Format
```json
{
  "answer": "The answer to your question",
  "sources": [
    {
      "id": "source-identifier",
      "content": "The source content",
      "location": "Chapter/Section reference",
      "url": "Link to source",
      "score": 0.87
    }
  ],
  "confidence": 0.92,
  "timestamp": "2025-12-13T10:30:00Z"
}
```

## Components Structure

### Main Components
- `Chatbot.jsx` - Main container component with toggle functionality
- `ChatWindow.jsx` - Container for message history
- `Message.jsx` - Individual message display with sources
- `InputArea.jsx` - Input form with submit button

### Hooks
- `useChatAPI.js` - Handles backend API communication
- `useTextSelection.js` - Captures selected text on the page

### Utilities
- `analytics.js` - Tracks usage events
- `types.ts` - TypeScript interfaces

## Environment Variables
- `REACT_APP_BACKEND_URL` - URL of the backend API (defaults to http://localhost:8000)
- `REACT_APP_ANALYTICS_ENABLED` - Enable/disable analytics (defaults to false)
- `REACT_APP_ANALYTICS_ID` - Analytics service ID

## Analytics Events
The component tracks the following events:
- `query_submitted` - When a user submits a question
- `chatbot_query` - When a successful query response is received
- `source_clicked` - When a user clicks on a source link
- `chatbot_error` - When an error occurs

## Styling
The component uses CSS modules for styling and is designed to match Docusaurus themes. All styles are in `chatbot.css` and can be customized as needed.

## Accessibility
- Keyboard navigation support
- Proper ARIA labels
- Screen reader friendly
- High contrast mode compatible

## Browser Support
- Chrome 70+
- Firefox 65+
- Safari 12+
- Edge 79+

## Error Handling
The component gracefully handles:
- Network errors
- API errors
- Invalid responses
- Timeout conditions

Error messages are displayed to users in a user-friendly format.

## Performance Considerations
- Lazy loading of components
- Efficient state management
- Debounced API calls
- Optimized rendering