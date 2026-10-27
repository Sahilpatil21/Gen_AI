from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge_base"
VECTOR_DIR = DATA_DIR / "vector_store"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
MAX_NEW_TOKENS = 384
