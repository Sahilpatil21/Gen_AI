"""
app.py
======
Flask Web Studio for Experiment 9: Personalized AI Assistant (RAG).
Handles document ingestion (PDF, DOCX, TXT, MD), semantic chunking, and grounded Q&A.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from assistant.config import DEFAULT_MODEL, EMBEDDING_MODEL, MAX_NEW_TOKENS
from assistant.documents import read_file_bytes, split_documents, TextDocument
from assistant.rag import RAGAssistant

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# Global assistant singleton
assistant_instance = None
indexed_chunks: list[TextDocument] = []


def get_assistant() -> RAGAssistant:
    global assistant_instance
    if assistant_instance is None:
        assistant_instance = RAGAssistant(DEFAULT_MODEL, EMBEDDING_MODEL)
    return assistant_instance


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "experiment": "EXP_09_Personalized_AI_Assistant_RAG",
        "indexed_chunks": len(indexed_chunks),
    })


@app.route("/api/system-info")
def system_info():
    import torch
    return jsonify({
        "status": "online",
        "model_name": DEFAULT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "indexed_chunks": len(indexed_chunks),
    })


@app.route("/api/upload", methods=["POST"])
def upload_files():
    global indexed_chunks
    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400

    docs = []
    file_count = 0
    for f in files:
        if not f.filename:
            continue
        try:
            content = f.read()
            extracted = read_file_bytes(f.filename, content)
            docs.extend(extracted)
            file_count += 1
        except Exception as exc:
            return jsonify({"success": False, "error": f"Failed reading {f.filename}: {exc}"}), 400

    new_chunks = split_documents(docs)
    updated_chunks = indexed_chunks + new_chunks
    try:
        assistant = get_assistant()
        assistant.index_documents(updated_chunks)
    except Exception:
        app.logger.exception("Document indexing failed")
        return jsonify({
            "success": False,
            "error": "Could not index documents. Check the application logs and try again.",
        }), 503

    indexed_chunks = updated_chunks

    return jsonify({
        "success": True,
        "indexed_count": file_count,
        "total_chunks": len(indexed_chunks),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = (data.get("message") or "").strip()
    if not query:
        return jsonify({"success": False, "error": "Message is required"}), 400

    found = []
    sources = []
    degraded = False
    try:
        assistant = get_assistant()
        found = assistant.retrieve(query, top_k=4)
        sources = [{
            "source": doc.source,
            "excerpt": doc.text[:220] + "..." if len(doc.text) > 220 else doc.text,
            "score": round(float(score), 4),
        } for doc, score in found]
        answer, _ = assistant.answer(query, top_k=4)
    except Exception:
        app.logger.exception("Chat answer generation failed")
        if found:
            top_excerpts = "\n\n".join(f"- {d.text}" for d, _ in found[:2])
            answer = (
                "I couldn't generate a model response, so here are the most relevant "
                f"passages I found ({len(found)}):\n\n{top_excerpts}"
            )
            degraded = True
        else:
            return jsonify({
                "success": False,
                "error": "The assistant could not generate an answer. Check the application logs and try again.",
            }), 503

    return jsonify({
        "success": True,
        "query": query,
        "answer": answer,
        "sources": sources,
        "degraded": degraded,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5009))
    print(f"\n* Launching Personalized AI Assistant on http://127.0.0.1:{port} ...\n")
    app.run(host="127.0.0.1", port=port, debug=False)
