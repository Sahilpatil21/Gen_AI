from assistant.config import KNOWLEDGE_DIR, VECTOR_DIR, EMBEDDING_MODEL
from assistant.documents import load_folder, split_documents
from sentence_transformers import SentenceTransformer
import numpy as np, json

def main():
    chunks = split_documents(load_folder(KNOWLEDGE_DIR))
    if not chunks:
        print(f"Add PDF, DOCX, TXT, or MD files to {KNOWLEDGE_DIR} first.")
        return
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vectors = embedder.encode([c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTOR_DIR / "embeddings.npy", vectors.astype("float32"))
    (VECTOR_DIR / "chunks.json").write_text(json.dumps([{"text":c.text,"source":c.source} for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(chunks)} chunks to {VECTOR_DIR}. The Streamlit app indexes uploaded files in memory.")
if __name__ == "__main__":
    main()
