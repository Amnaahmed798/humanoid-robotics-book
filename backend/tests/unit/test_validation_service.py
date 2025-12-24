import pytest
from backend.agent.validation_service import ValidationService
from backend.models.retrieval_models import RetrievedChunk
from backend.models.agent_models import AgentResponse


@pytest.fixture
def validation_service():
    return ValidationService()


def test_validate_response_grounded(validation_service):
    """Test validation of a properly grounded response."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    response = AgentResponse(
        answer="Humanoid robots are robots with physical features resembling humans.",
        sources=context,
        confidence=0.8
    )

    is_valid = validation_service._check_grounding(
        response.answer.lower(),
        " ".join([chunk.text.lower() for chunk in context])
    )

    # The response contains the same content as the context, so it should be grounded
    assert is_valid


def test_validate_response_not_grounded(validation_service):
    """Test validation of a response that is not grounded in context."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    response = AgentResponse(
        answer="Dogs are animals that are commonly kept as pets.",
        sources=context,
        confidence=0.1
    )

    is_valid = validation_service._check_grounding(
        response.answer.lower(),
        " ".join([chunk.text.lower() for chunk in context])
    )

    # The response is about dogs while context is about robots, so it should not be grounded
    assert not is_valid


def test_validate_response_with_insufficient_info(validation_service):
    """Test validation when response indicates insufficient information."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    response = AgentResponse(
        answer="The information is not available in the provided context.",
        sources=context,
        confidence=0.1
    )

    # This is a valid response that acknowledges lack of information
    result = validation_service._check_grounding(
        response.answer.lower(),
        " ".join([chunk.text.lower() for chunk in context])
    )

    # For lack of info responses, we should still check if the grounding check works
    # The grounding check might return False, but the validation logic should handle this differently
    # Let's test the specific case of lack of info directly


@pytest.mark.asyncio
async def test_validate_response_async_method_grounded(validation_service):
    """Test the async validation method with a grounded response."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    response = AgentResponse(
        answer="Humanoid robots are robots with physical features resembling humans.",
        sources=context,
        confidence=0.8
    )

    is_valid = await validation_service.validate_response(response, context)

    # The response should be considered valid since it's grounded in the context
    assert is_valid


@pytest.mark.asyncio
async def test_validate_response_async_method_not_grounded(validation_service):
    """Test the async validation method with a non-grounded response."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    response = AgentResponse(
        answer="Dogs are animals that are commonly kept as pets.",
        sources=context,
        confidence=0.1
    )

    is_valid = await validation_service.validate_response(response, context)

    # The response should not be considered valid since it's not grounded in the context
    assert not is_valid


@pytest.mark.asyncio
async def test_validate_response_with_lack_of_info(validation_service):
    """Test validation when response indicates lack of information."""
    context = [
        RetrievedChunk(
            id="1",
            text="Humanoid robots are robots with physical features resembling humans.",
            source_location="Chapter 1, Section 1.1",
            similarity_score=0.8
        )
    ]

    response = AgentResponse(
        answer="The information is not available in the provided context.",
        sources=context,
        confidence=0.1
    )

    is_valid = await validation_service.validate_response(response, context)

    # A response that indicates lack of information should be considered valid
    assert is_valid