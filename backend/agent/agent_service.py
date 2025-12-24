import os
import logging
from typing import List
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.agent_models import AgentResponse
from models.retrieval_models import RetrievedChunk
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize clients lazily (in __init__ method to avoid import-time errors)
def _get_clients():
    """
    Initialize OpenAI clients based on configuration.
    This is called when needed to avoid import-time errors when API keys are missing.
    """
    # Check if using Gemini API
    if os.getenv("USE_GEMINI_API", "false").lower() == "true":
        # Configure for Gemini API using OpenAI-compatible endpoint
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY environment variable is required")

        client = OpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        async_client = AsyncOpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        # Use regular OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        client = OpenAI(api_key=openai_api_key)
        async_client = AsyncOpenAI(api_key=openai_api_key)

    return client, async_client


class AgentService:
    """
    Service for interacting with OpenAI to generate grounded responses.
    """

    def __init__(self):
        """
        Initialize the agent service.
        """
        # Don't raise error on init, just store config - client will be created when needed
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Default to gpt-3.5-turbo
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))  # Low temperature for consistency
        # Initialize clients when first needed
        self._client = None
        self._async_client = None
        # Check if API keys are properly configured
        self.api_keys_configured = self._check_api_keys_configured()

    def _check_api_keys_configured(self):
        """
        Check if API keys are properly configured.
        """
        if os.getenv("USE_GEMINI_API", "false").lower() == "true":
            # Check if using Gemini API
            gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
            return gemini_api_key is not None and len(gemini_api_key.strip()) > 0
        else:
            # Check if using OpenAI
            openai_api_key = os.getenv("OPENAI_API_KEY")
            return openai_api_key is not None and len(openai_api_key.strip()) > 0

    async def generate_response(self, question: str, context: List[RetrievedChunk]) -> AgentResponse:
        """
        Generate a response to the user's question using the provided context.

        Args:
            question: The user's question
            context: List of RetrievedChunk objects containing relevant book content

        Returns:
            AgentResponse object containing the answer, sources, and confidence score
        """
        try:
            # Check if API keys are properly configured
            if not self.api_keys_configured:
                # Return a mock response when API keys are not configured
                mock_answer = f"Based on the provided context, here's what I can tell you about '{question}':\n\n"

                # Include some relevant content from the context
                if context:
                    # Take the first chunk as a sample response
                    sample_chunk = context[0]
                    # Limit the content to make a reasonable response
                    content_preview = sample_chunk.text[:500] + "..." if len(sample_chunk.text) > 500 else sample_chunk.text
                    mock_answer += f"From {sample_chunk.source_location}:\n{content_preview}"
                else:
                    mock_answer += "I couldn't find specific information about this topic in the provided content."

                # Create and return a mock agent response
                agent_response = AgentResponse(
                    answer=mock_answer,
                    sources=context,
                    confidence=0.5,  # Moderate confidence for mock response
                    raw_response=mock_answer
                )

                logging.info(f"API keys not configured, returning mock response for question: {question[:50]}...")
                return agent_response

            # Prepare the context content by combining all retrieved chunks
            context_text = "\n\n".join([
                f"Source {i+1} ({chunk.source_location}): {chunk.text}"
                for i, chunk in enumerate(context)
            ])

            # Create the system message to ground the agent in the book content
            system_message = {
                "role": "system",
                "content": (
                    "You are an expert assistant for the humanoid robotics book. "
                    "Answer the user's question based ONLY on the provided book content. "
                    "Do not make up information or use knowledge beyond what's provided. "
                    "If the answer is not available in the provided context, clearly state that. "
                    "Always cite the source locations of the information you use in your response."
                )
            }

            # Create the user message with the question and context
            user_message = {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Book Content Context:\n{context_text}\n\n"
                    f"Please answer the question based only on the provided book content. "
                    f"If the information is not available in the context, say so explicitly."
                )
            }

            # Get clients (initialize if needed)
            if self._async_client is None:
                client, async_client = _get_clients()
                self._client = client
                self._async_client = async_client

            # Call the API using the configured client
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=[system_message, user_message],
                temperature=self.temperature,
                max_tokens=1000,  # Limit response length
                timeout=30  # 30 second timeout
            )

            # Extract the response text
            answer = response.choices[0].message.content.strip()

            # Calculate a basic confidence score based on response length and context usage
            # In a more sophisticated implementation, this would involve more complex analysis
            confidence = self._calculate_confidence(answer, context)

            # Create and return the agent response
            agent_response = AgentResponse(
                answer=answer,
                sources=context,
                confidence=confidence,
                raw_response=response.choices[0].message.content
            )

            logging.info(f"Generated response for question: {question[:50]}...")
            return agent_response

        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")

            # Return a mock response in case of any error
            mock_answer = f"I encountered an issue while processing your question about '{question}'. "
            if context:
                mock_answer += f"Based on the provided content: {context[0].text[:200]}..." if len(context[0].text) > 200 else f"Based on the provided content: {context[0].text}"
            else:
                mock_answer += "I couldn't find relevant information to answer your question."

            return AgentResponse(
                answer=mock_answer,
                sources=context,
                confidence=0.3,  # Lower confidence for error case
                raw_response=mock_answer
            )

    def _calculate_confidence(self, answer: str, context: List[RetrievedChunk]) -> float:
        """
        Calculate a basic confidence score based on response and context characteristics.

        Args:
            answer: The generated answer
            context: The context used to generate the answer

        Returns:
            Confidence score between 0 and 1
        """
        # If the answer indicates insufficient information, return low confidence
        if "not available" in answer.lower() or "not mentioned" in answer.lower():
            return 0.1  # Low confidence when information is not in context

        # Calculate confidence based on how much of the context was potentially used
        # This is a simple heuristic - a more sophisticated implementation would use semantic similarity
        context_length = sum(len(chunk.text) for chunk in context)
        answer_length = len(answer)

        # Normalize the answer length relative to the context length
        if context_length > 0:
            # Ensure answer is substantial relative to context
            relative_length = min(1.0, answer_length / (context_length * 0.3))  # Assume 30% of context is typical
            base_confidence = 0.5 + (relative_length * 0.5)  # Base 0.5 + up to 0.5 based on length
        else:
            base_confidence = 0.1  # Very low confidence if no context

        # Additional heuristics could be added here
        # For example, checking if the answer references specific source locations

        return min(1.0, base_confidence)  # Ensure confidence is between 0 and 1