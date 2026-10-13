"""Stage 4: MMR retrieval at several lambda_mult values, shown side by side."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.store import get_store

SEEDS = ["winter", "war", "love", "ocean", "mourning",
         "gold", "music", "storm", "childhood", "ritual"]
LAMBDAS = [1.0, 0.7, 0.4, 0.1]
K, FETCH_K, W = 5, 50, 34

store = get_store()
emb = store.embeddings

def vecs(docs):
    ids = [str(d.metadata["object_id"]) for d in docs]
    got = store.get(ids=ids, include=["embeddings"])
    by_id = dict(zip(got["ids"], np.array(got["embeddings"])))
    return np.array([by_id[i] for i in ids])

summary = {l: {"rel": [], "red": []} for l in LAMBDAS}
for seed in SEEDS:
    q = np.array(emb.embed_query(seed))
    cols = {}
    for lam in LAMBDAS:
        docs = store.max_marginal_relevance_search(seed, k=K, fetch_k=FETCH_K, lambda_mult=lam)
        V = vecs(docs)
        rel = float((V @ q).mean())
        red = float(np.mean([V[i] @ V[j] for i, j in combinations(range(K), 2)]))
        summary[lam]["rel"].append(rel); summary[lam]["red"].append(red)
        cols[lam] = [d.metadata["title"][:W - 2] for d in docs] + [f"rel={rel:.3f} red={red:.3f}"]
    print(f"\n=== {seed} ===")
    print("  ".join(f"lambda={l:<{W - 7}}" for l in LAMBDAS))
    for row in range(K + 1):
        print("  ".join(f"{cols[l][row]:<{W}}" for l in LAMBDAS))

print("\n=== Averages over all 10 seeds ===")
print("lambda   mean sim to query   mean pairwise sim among results")
for l in LAMBDAS:
    print(f"{l:<8} {np.mean(summary[l]['rel']):.3f}               {np.mean(summary[l]['red']):.3f}")
