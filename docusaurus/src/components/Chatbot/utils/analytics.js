// Analytics utility for tracking chatbot usage
class ChatbotAnalytics {
  constructor() {
    // Safely check for environment variables in both Node.js and browser environments
    const getEnvVar = (name, defaultValue = null) => {
      try {
        if (typeof process !== 'undefined' && process.env) {
          // Node.js environment
          return process.env[name] || defaultValue;
        } else if (typeof window !== 'undefined') {
          // Browser environment - Docusaurus should expose these via webpack DefinePlugin
          // If not, we'll return the default value
          return defaultValue;
        }
      } catch (e) {
        // If there's any error accessing environment variables, return default
        return defaultValue;
      }
      return defaultValue;
    };

    this.enabled = getEnvVar('REACT_APP_ANALYTICS_ENABLED', 'false') === 'true';
    this.trackingId = getEnvVar('REACT_APP_ANALYTICS_ID', null);
  }

  trackEvent(eventName, properties = {}) {
    if (!this.enabled) return;

    // Log the event (in a real implementation, this would send to an analytics service)
    console.log('Analytics Event:', {
      event: eventName,
      properties,
      timestamp: new Date().toISOString()
    });

    // Example of how to integrate with Google Analytics or similar
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', eventName, properties);
    }
  }

  trackQuery(question, response) {
    this.trackEvent('chatbot_query', {
      question: question,
      has_sources: response.sources && response.sources.length > 0,
      response_length: response.answer ? response.answer.length : 0,
      confidence: response.confidence
    });
  }

  trackSourceClick(sourceId, location) {
    this.trackEvent('source_clicked', {
      source_id: sourceId,
      location: location
    });
  }

  trackError(error, context) {
    this.trackEvent('chatbot_error', {
      error: error.message || error,
      context: context
    });
  }
}

export default new ChatbotAnalytics();