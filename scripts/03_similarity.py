"""Stage 3: plain similarity search with scores for 10 seed words."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.store import get_store

SEEDS = ["winter", "war", "love", "ocean", "mourning",
         "gold", "music", "storm", "childhood", "ritual"]

store = get_store()
for seed in SEEDS:
    print(f"\n=== {seed} ===")
    # Returns (Document, distance) pairs; Chroma cosine distance = 1 - cosine similarity.
    for doc, dist in store.similarity_search_with_score(seed, k=5):
        m = doc.metadata
        print(f"  dist={dist:.3f} sim={1 - dist:.3f}  [{m['object_id']}] {m['title'][:60]} | {m['department']}")
