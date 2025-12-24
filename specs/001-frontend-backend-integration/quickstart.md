# Quickstart: Frontend-Backend Integration & Embedded Chatbot

## Overview
This guide provides a quick walkthrough of how to implement and test the embedded chatbot feature that connects the Docusaurus frontend with the FastAPI backend.

## Prerequisites
- Node.js 18+ for Docusaurus development
- Python 3.10+ for FastAPI backend
- Access to the existing FastAPI backend with `/query` endpoint
- Docusaurus project set up and running

## Setup Steps

### 1. Backend Verification
First, ensure the backend API is accessible:
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is humanoid robotics?", "top_k": 3}'
```

### 2. Frontend Component Installation
Add the chatbot component to your Docusaurus project:
```bash
# Navigate to your Docusaurus project
cd your-docusaurus-project

# The chatbot component will be added to src/components/Chatbot
# Configuration will be added to docusaurus.config.js
```

### 3. Configuration
Update your Docusaurus configuration to include the backend API endpoint:

In `docusaurus.config.js`:
```javascript
module.exports = {
  // ... other config
  themeConfig: {
    // ... other theme config
    chatbot: {
      backendUrl: process.env.BACKEND_URL || 'http://localhost:8000',
      enabled: true,
      position: 'bottom-right' // or 'bottom-left'
    }
  }
};
```

### 4. Environment Variables
Set up environment variables for different environments:
```bash
# For development
BACKEND_URL=http://localhost:8000

# For production
BACKEND_URL=https://your-api-domain.com
```

## Testing the Integration

### Test 1: Basic Query
1. Start your Docusaurus development server: `npm run start`
2. Open any documentation page
3. Open the chatbot interface
4. Enter a question about the book content
5. Verify that you receive a response with sources

### Test 2: Selected Text Query
1. Highlight some text on a documentation page
2. Use the chatbot's text selection feature (if available)
3. Verify that the selected text is sent as context
4. Check that the response is relevant to the selected text

### Test 3: Error Handling
1. Temporarily stop the backend server
2. Try to submit a query
3. Verify that appropriate error messages are displayed

## Component Structure
The chatbot implementation includes:

```
src/components/Chatbot/
├── Chatbot.jsx          # Main component
├── ChatWindow.jsx       # Chat interface
├── Message.jsx          # Individual message display
├── InputArea.jsx        # Query input controls
├── hooks/
│   ├── useChatAPI.js    # API communication logic
│   └── useTextSelection.js # Text selection logic
└── styles/
    └── chatbot.css      # Component styling
```

## API Communication Flow
1. User submits a question or selects text
2. Frontend sends POST request to `/api/v1/query`
3. Backend processes the query using RAG system
4. Backend returns response with sources and confidence
5. Frontend displays the response in the chat interface

## Common Issues and Solutions

### Issue: CORS Errors
**Solution**: Ensure the backend has proper CORS configuration for your frontend domain.

### Issue: Backend Unreachable
**Solution**: Check that the BACKEND_URL is correctly configured and the server is running.

### Issue: Slow Response Times
**Solution**: Verify that the backend is properly optimized and the network connection is stable.

## Next Steps
1. Integrate the chatbot component into your Docusaurus layout
2. Customize the styling to match your site's theme
3. Add analytics to track usage patterns
4. Implement conversation history persistence if needed