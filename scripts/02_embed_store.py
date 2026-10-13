"""Stage 2: embed the 3000 sampled documents and persist them in Chroma (safe to re-run)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.data import build_documents
from serendipity.store import add_missing, get_store

docs, _ = build_documents(3000)
store = get_store()
print("before:", store._collection.count(), "items in Chroma")
result = add_missing(store, docs)
print("result:", result)
print("after: ", store._collection.count(), "items in Chroma")
