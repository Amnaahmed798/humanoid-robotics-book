import { useState } from 'react';

const useChatAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const queryBackend = async (question, context = {}) => {
    setLoading(true);
    setError(null);

    try {
      // Hardcoded backend URL for Docusaurus compatibility
      const BACKEND_URL = "http://localhost:8000";

      // In a real implementation, this would call the backend API
      // This corresponds to task T103: Implement API request to POST /api/v1/query endpoint with fetch
      const response = await fetch(`${BACKEND_URL}/api/v1/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          context,
          top_k: 3
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    queryBackend,
    loading,
    error,
  };
};

export default useChatAPI;