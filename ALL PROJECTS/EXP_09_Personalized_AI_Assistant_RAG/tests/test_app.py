import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import app as project_app
from assistant.documents import TextDocument


class FakeAssistant:
    def __init__(self, passages=(), error=None):
        self.passages = passages
        self.error = error

    def retrieve(self, query, top_k=4):
        return self.passages

    def answer(self, query, top_k=4):
        if self.error:
            raise self.error
        return "RAG retrieves relevant context before answering.", []

    def index_documents(self, chunks):
        raise RuntimeError("embedding model unavailable")


class TestChatEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = project_app.app.test_client()

    def test_chat_returns_generated_answer(self):
        assistant = FakeAssistant()
        with patch.object(project_app, "get_assistant", return_value=assistant):
            response = self.client.post("/api/chat", json={"message": "What is RAG?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["answer"], "RAG retrieves relevant context before answering.")

    def test_chat_reports_generation_failure_without_passages(self):
        assistant = FakeAssistant(error=RuntimeError("model unavailable"))
        with patch.object(project_app, "get_assistant", return_value=assistant):
            response = self.client.post("/api/chat", json={"message": "What is RAG?"})

        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])
        self.assertIn("could not generate", response.get_json()["error"])

    def test_chat_returns_passages_when_generation_fails(self):
        passages = [(TextDocument("RAG retrieves relevant context.", "notes.txt"), 0.9)]
        assistant = FakeAssistant(passages=passages, error=RuntimeError("model unavailable"))
        with patch.object(project_app, "get_assistant", return_value=assistant):
            response = self.client.post("/api/chat", json={"message": "What is RAG?"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["degraded"])
        self.assertIn("RAG retrieves relevant context.", response.get_json()["answer"])

    def test_upload_does_not_claim_success_when_indexing_fails(self):
        assistant = FakeAssistant()
        with patch.object(project_app, "get_assistant", return_value=assistant):
            response = self.client.post(
                "/api/upload",
                data={"files": (BytesIO(b"Relevant notes."), "notes.txt")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
