from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_rag_with_mock(monkeypatch):
    """Test the /query endpoint by mocking the RAG service."""

    def mock_answer_question(question: str) -> str:
        return "This is a mocked Gemini response."

    monkeypatch.setattr("main.rag_service.answer_question", mock_answer_question)

    response = client.post("/query", json={"question": "What is Jupiter?"})

    assert response.status_code == 200
    response_data = response.json()
    assert "answer" in response_data
    assert response_data["answer"] == "This is a mocked Gemini response."


def test_query_with_empty_question():
    """Test that an empty question returns a 400 Bad Request error."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Question field cannot be empty."}
