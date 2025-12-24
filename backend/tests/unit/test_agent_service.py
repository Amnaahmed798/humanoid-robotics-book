import pytest
import os
from unittest.mock import AsyncMock, patch
from backend.agent.agent_service import AgentService
from backend.models.retrieval_models import RetrievedChunk


@pytest.fixture
def agent_service():
    # Mock the OpenAI API key for testing
    os.environ["OPENAI_API_KEY"] = "test-key"
    return AgentService()


@pytest.mark.asyncio
async def test_generate_response_success(agent_service):
    """Test successful response generation."""
    # Create mock context
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    question = "What are humanoid robots?"

    # Create a mock response object that behaves like OpenAI's response
    class MockChoice:
        class MockMessage:
            def __init__(self, content):
                self.content = content
        def __init__(self, content):
            self.message = self.MockMessage(content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    mock_response = MockResponse("Humanoid robots are robots with physical features resembling humans.")

    with patch("backend.agent.agent_service.openai.ChatCompletion.acreate",
               new=AsyncMock(return_value=mock_response)):
        result = await agent_service.generate_response(question, context)

        assert result.answer == "Humanoid robots are robots with physical features resembling humans."
        assert len(result.sources) == 1
        assert result.sources[0].id == "1"
        assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_generate_response_with_empty_context(agent_service):
    """Test response generation with empty context."""
    context = []
    question = "What are humanoid robots?"

    # Create a mock response object that behaves like OpenAI's response
    class MockChoice:
        class MockMessage:
            def __init__(self, content):
                self.content = content
        def __init__(self, content):
            self.message = self.MockMessage(content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    mock_response = MockResponse("The information is not available in the provided context.")

    with patch("backend.agent.agent_service.openai.ChatCompletion.acreate",
               new=AsyncMock(return_value=mock_response)):
        result = await agent_service.generate_response(question, context)

        assert "not available" in result.answer.lower()
        assert result.confidence <= 0.2  # Low confidence for no context


def test_calculate_confidence_with_insufficient_info(agent_service):
    """Test confidence calculation when response indicates insufficient information."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    answer = "The information is not available in the provided context."

    confidence = agent_service._calculate_confidence(answer, context)

    assert confidence <= 0.2  # Low confidence when information is not in context


def test_calculate_confidence_with_good_response(agent_service):
    """Test confidence calculation with a good response."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    answer = "Humanoid robots are robots with physical features resembling humans."

    confidence = agent_service._calculate_confidence(answer, context)

    # Should have higher confidence since it's based on the context
    assert confidence >= 0.5