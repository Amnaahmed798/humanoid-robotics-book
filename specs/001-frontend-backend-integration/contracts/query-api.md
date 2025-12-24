# API Contract: Query Endpoint

## Endpoint
`POST /api/v1/query`

## Purpose
Accepts user questions and returns grounded responses from book content using the RAG system.

## Request

### Headers
- `Content-Type: application/json`
- `Accept: application/json`

### Body Schema
```json
{
  "question": {
    "type": "string",
    "minLength": 1,
    "maxLength": 1000,
    "description": "The user's question",
    "required": true
  },
  "context": {
    "type": "object",
    "properties": {
      "selectedText": {
        "type": "string",
        "description": "Text selected by the user on the current page",
        "required": false
      },
      "pageUrl": {
        "type": "string",
        "description": "URL of the current page for additional context",
        "required": false
      },
      "contextBefore": {
        "type": "string",
        "description": "Text content before the selected text",
        "required": false
      },
      "contextAfter": {
        "type": "string",
        "description": "Text content after the selected text",
        "required": false
      }
    }
  },
  "top_k": {
    "type": "integer",
    "minimum": 1,
    "maximum": 10,
    "default": 3,
    "description": "Number of results to retrieve from vector database",
    "required": false
  }
}
```

### Example Request
```json
{
  "question": "What are the main challenges in humanoid robot locomotion?",
  "context": {
    "selectedText": "dynamic walking",
    "pageUrl": "/docs/locomotion/dynamic-walking",
    "contextBefore": "Humanoid robots face several challenges in locomotion, particularly when it comes to",
    "contextAfter": "which requires sophisticated balance control algorithms."
  },
  "top_k": 3
}
```

## Response

### Success Response (200 OK)
```json
{
  "answer": {
    "type": "string",
    "description": "The answer to the user's question",
    "required": true
  },
  "sources": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "id": {
          "type": "string",
          "description": "Unique identifier for the source",
          "required": true
        },
        "content": {
          "type": "string",
          "description": "The text content from the source",
          "required": true
        },
        "location": {
          "type": "string",
          "description": "Location in the book (e.g., chapter, section)",
          "required": true
        },
        "url": {
          "type": "string",
          "description": "URL to navigate to the source location",
          "required": true
        },
        "score": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Similarity score between query and source",
          "required": true
        }
      },
      "required": ["id", "content", "location", "url", "score"]
    },
    "description": "List of sources that support the answer",
    "required": true
  },
  "confidence": {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
    "description": "Confidence score for the response",
    "required": true
  },
  "timestamp": {
    "type": "string",
    "format": "date-time",
    "description": "When the response was generated",
    "required": true
  }
}
```

### Example Success Response
```json
{
  "answer": "The main challenges in humanoid robot locomotion include maintaining balance during dynamic movements, adapting to uneven terrain, and achieving energy-efficient walking patterns.",
  "sources": [
    {
      "id": "ch3-sec2-para5",
      "content": "Dynamic walking in humanoid robots requires sophisticated balance control algorithms to maintain stability during movement transitions.",
      "location": "Chapter 3, Section 2, Paragraph 5",
      "url": "/docs/locomotion/dynamic-walking#balance-control",
      "score": 0.87
    },
    {
      "id": "ch4-sec1-para3",
      "content": "Terrain adaptation remains a significant challenge, as humanoid robots must adjust their gait patterns in real-time based on surface conditions.",
      "location": "Chapter 4, Section 1, Paragraph 3",
      "url": "/docs/locomotion/terrain-adaptation#real-time",
      "score": 0.79
    }
  ],
  "confidence": 0.92,
  "timestamp": "2025-12-13T10:30:00Z"
}
```

### Error Responses

#### 400 Bad Request
Returned when the request body doesn't match the expected schema.

```json
{
  "error": {
    "type": "string",
    "description": "Description of the validation error",
    "required": true
  },
  "details": {
    "type": "object",
    "description": "Additional details about the validation error",
    "required": false
  }
}
```

#### 404 Not Found
Returned when the requested resource doesn't exist.

#### 500 Internal Server Error
Returned when there's an unexpected server error.

```json
{
  "error": {
    "type": "string",
    "description": "Description of the server error",
    "required": true
  }
}
```

## Security Considerations
- Input validation to prevent injection attacks
- Rate limiting to prevent abuse
- Content filtering to prevent malicious input

## Performance Requirements
- Response time under 10 seconds for 90% of requests
- Support for concurrent users without degradation
- Efficient handling of large text inputs