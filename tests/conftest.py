import pytest
import instructor

@pytest.fixture
def instructor_client():
    """
    Creates a real instructor client for testing.
    Uses OpenAI by default - requires API key in environment.
    """
    return instructor.from_provider("openai/gpt-4o-mini")

@pytest.fixture
def sample_context():
    """
    Provides a sample context for testing RAG evaluations.
    """
    return [
        "Regular physical activity improves heart health and circulation.",
        "Weight training builds muscle strength and increases bone density.",
        "The earliest Olympic games were held in Ancient Greece."
    ]

@pytest.fixture
def sample_question():
    """
    Provides a sample question for testing.
    """
    return "What are the benefits of exercise?"

@pytest.fixture
def sample_answer():
    """
    Provides a sample answer for testing.
    """
    return "Regular exercise improves cardiovascular health and increases strength."