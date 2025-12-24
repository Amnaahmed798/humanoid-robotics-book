import logging
from typing import List
import sys
import os
# Add the backend directory to the Python path to allow imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.retrieval_models import RetrievedChunk
from models.agent_models import AgentResponse


class ValidationService:
    """
    Service for validating that agent responses are grounded in the retrieved context.
    This service ensures that the agent doesn't hallucinate information beyond what's
    in the retrieved book content.
    """

    def __init__(self):
        """
        Initialize the validation service.
        """
        pass

    async def validate_response(self, response: AgentResponse, context: List[RetrievedChunk]) -> bool:
        """
        Validate that the agent's response is grounded in the provided context.

        Args:
            response: The agent's response to validate
            context: The context that was provided to the agent

        Returns:
            True if the response is adequately grounded in the context, False otherwise
        """
        try:
            # Extract key information from the response
            answer = response.answer.lower() if response and response.answer else ""

            # Combine all context content to check against
            # Handle empty context or context with missing text fields
            valid_chunks = []
            for chunk in context:
                if chunk and hasattr(chunk, 'text') and chunk.text:
                    valid_chunks.append(chunk.text.lower())

            context_content = " ".join(valid_chunks)

            # Check if the response contains information that can be traced back to context
            # This is a simplified check - in practice, you might want more sophisticated NLP techniques
            is_adequately_grounded = self._check_grounding(answer, context_content)

            # Also check if the response explicitly mentions lack of information
            lacks_info_indicators = [
                "not available in the provided context",
                "not mentioned in the book content",
                "not found in the provided text",
                "not specified in the context",
                "no information provided about",
                "not contained in the book content"
            ]

            lacks_info = any(indicator in answer for indicator in lacks_info_indicators)

            # If the response indicates lack of information, it's valid (no hallucination)
            if lacks_info:
                logging.info("Response indicates lack of information in context - valid")
                return True

            # Log the validation result
            if is_adequately_grounded:
                logging.info("Response validation passed - grounded in context")
            else:
                logging.warning("Response validation failed - may contain hallucinations")

            return is_adequately_grounded

        except Exception as e:
            logging.error(f"Error during response validation: {str(e)}")
            # In case of validation error, it's safer to assume the response is not properly grounded
            return False

    def _check_grounding(self, answer: str, context_content: str) -> bool:
        """
        Check if the answer is grounded in the context content.

        This is a simplified implementation. A more sophisticated approach would:
        - Use semantic similarity to check if concepts in the answer appear in the context
        - Check if specific claims in the answer can be supported by the context
        - Use NLP techniques to identify if the answer contains information not present in context

        Args:
            answer: The agent's answer to check
            context_content: The combined content of all retrieved chunks

        Returns:
            True if the answer appears to be grounded in the context, False otherwise
        """
        # Clean up the content for comparison
        import re

        # Remove extra whitespace and convert to lowercase
        answer_clean = re.sub(r'\s+', ' ', answer.strip().lower())
        context_clean = re.sub(r'\s+', ' ', context_content.strip().lower())

        # For a more sophisticated check, we could:
        # 1. Extract key terms/phrases from the answer
        # 2. Check if they appear in the context
        # 3. Use semantic similarity models

        # Simple approach: check if a reasonable portion of key terms from the answer
        # appear in the context
        answer_words = set(answer_clean.split())

        # Filter out common stop words that don't indicate grounding
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }

        answer_content_words = {word for word in answer_words if word not in stop_words and len(word) > 2}

        if not answer_content_words:
            # If there are no content words, the answer might be just "I don't know" or similar
            return True  # This is valid if it indicates lack of information

        context_words = set(context_clean.split())

        # Calculate overlap between answer content words and context
        matching_words = answer_content_words.intersection(context_words)
        overlap_ratio = len(matching_words) / len(answer_content_words) if answer_content_words else 0

        # Consider the response grounded if at least 30% of content words appear in context
        # This threshold can be adjusted based on requirements
        is_adequately_grounded = overlap_ratio >= 0.3

        return is_adequately_grounded