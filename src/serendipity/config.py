"""Central settings so every stage uses the same paths and model names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:7b-instruct"  # revisited in Stage 6

COLLECTION = "met_objects"
RANDOM_SEED = 42
