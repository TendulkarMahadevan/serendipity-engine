"""Stage 5a: sanity checks, then inspect every KMeans cluster."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serendipity.store import get_store
from serendipity.strategies import embed_seed, fit_kmeans, load_corpus

C = load_corpus()
n = len(C.ids)
print(f"loaded {n} vectors, shape {C.X.shape}")

# Check 1: does Chroma's returned distance equal 1 - cosine similarity?
seed = "gold"
q = embed_seed(seed)
sims = C.X @ q
for doc, dist in get_store().similarity_search_with_score(seed, k=3):
    i = C.ids.index(str(doc.metadata["object_id"]))
    print(f"check: chroma dist={dist:.4f}  1-cos={1 - sims[i]:.4f}")

# Check 2: baseline similarity between random pairs (context for Stage 4's 0.70).
rng = np.random.default_rng(0)
a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
m = a != b
pair = np.sum(C.X[a[m]] * C.X[b[m]], axis=1)
print(f"random pairs: mean cos={pair.mean():.3f}, p5={np.percentile(pair, 5):.3f}, p95={np.percentile(pair, 95):.3f}")

# Distribution of similarity from a seed to everything.
print(f"seed 'gold' vs all: min={sims.min():.3f} p5={np.percentile(sims, 5):.3f} "
      f"p10={np.percentile(sims, 10):.3f} p30={np.percentile(sims, 30):.3f} "
      f"median={np.median(sims):.3f} max={sims.max():.3f}")

km = fit_kmeans(C.X, 30)
lens = np.array([len(t) for t in C.texts])
print(f"\nKMeans k=30, inertia={km.inertia_:.1f}, sizes sorted: {sorted(Counter(km.labels_).values())}")

for c in range(30):
    members = np.where(km.labels_ == c)[0]
    cent = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
    order = members[np.argsort(-(C.X[members] @ cent))][:10]
    depts = Counter(C.metas[i]["department"] for i in members).most_common(2)
    short = (lens[members] < 50).mean() * 100
    print(f"\n--- cluster {c:2d}: size={len(members):4d}, mean text len={lens[members].mean():5.0f}, "
          f"short(<50)={short:4.1f}%, top depts={depts}")
    for i in order:
        print(f"      {C.metas[i]['title'][:70]}")
