"""Stage 0: prove Ollama is reachable and an embedding call works end to end."""
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.config import EMBED_MODEL, OLLAMA_URL
from serendipity.embeddings import NomicEmbeddings

# 1. Is the Ollama server up, and is the embedding model pulled?
tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
names = [m["name"] for m in tags["models"]]
print("Ollama reachable. Models:", names)
assert any(n.startswith(EMBED_MODEL) for n in names), f"run: ollama pull {EMBED_MODEL}"

# 2. Embed one document and a few queries through our wrapper.
emb = NomicEmbeddings()
doc = emb.embed_documents(["A bronze ritual vessel, Chinese, Shang dynasty"])[0]
print(f"Document vector: length={len(doc)}, norm={np.linalg.norm(doc):.2f}")

def cos(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

for q in ["ancient bronze cooking pot", "impressionist painting of a garden"]:
    print(f"cosine({q!r}, doc) = {cos(emb.embed_query(q), doc):.3f}")
