// User Query interface
export interface UserQuery {
  id: string;
  text: string;
  type: 'normal' | 'selected-text';
  selectedText?: string;
  pageContext?: string;
  timestamp: string;
  context?: string;
}

// Source interface
export interface Source {
  id: string;
  content: string;
  location: string;
  score: number;
  url: string;
}

// Chatbot Response interface
export interface ChatbotResponse {
  id: string;
  answer: string;
  sources: Source[];
  confidence: number;
  timestamp: string;
  queryId: string;
}

// Message interface
export interface Message {
  id: string;
  conversationId: string;
  sender: 'user' | 'bot';
  content: string;
  timestamp: string;
  type: 'query' | 'response' | 'error';
}

// Conversation interface
export interface Conversation {
  id: string;
  userId?: string;
  sessionId: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

// Selected Text Context interface
export interface SelectedTextContext {
  id: string;
  text: string;
  pageUrl: string;
  selectionStart: number;
  selectionEnd: number;
  surroundingText?: string;
  timestamp: string;
}