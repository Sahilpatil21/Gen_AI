"""
test_documents.py
=================
Unit tests for Experiment 9 document parsing and chunking.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.documents import TextDocument, split_documents, read_file_bytes


class TestDocuments(unittest.TestCase):
    def test_split_documents(self):
        doc = TextDocument(text="This is a sentence. " * 30, source="test.txt")
        chunks = split_documents([doc], chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].source, "test.txt")

    def test_read_txt_bytes(self):
        sample = b"Hello from GenAI Lab assistant!"
        docs = read_file_bytes("greeting.txt", sample)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "Hello from GenAI Lab assistant!")
        self.assertEqual(docs[0].source, "greeting.txt")


if __name__ == "__main__":
    unittest.main()
