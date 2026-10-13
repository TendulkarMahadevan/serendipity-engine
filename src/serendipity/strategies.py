"""Stage 5: pick 1 close match + 2 distant items from the full embedding matrix.

All vectors are L2-normalised, so cosine similarity is just a dot product.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from .config import RANDOM_SEED
from .store import get_store


@dataclass
class Corpus:
    ids: list[str]
    texts: list[str]
    metas: list[dict]
    X: np.ndarray  # (n, 768), unit-length rows


def load_corpus() -> Corpus:
    """Pull every vector, text and metadata record out of Chroma, sorted by id for stability."""
    store = get_store()
    r = store.get(include=["embeddings", "documents", "metadatas"])
    order = np.argsort([int(i) for i in r["ids"]])
    X = np.array(r["embeddings"])[order].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # re-normalise defensively
    return Corpus(
        ids=[r["ids"][i] for i in order],
        texts=[r["documents"][i] for i in order],
        metas=[r["metadatas"][i] for i in order],
        X=X,
    )


def embed_seed(seed: str) -> np.ndarray:
    """Embed a seed as a *query* (adds the search_query prefix) and normalise."""
    v = np.array(get_store().embeddings.embed_query(seed), dtype=np.float64)
    return v / np.linalg.norm(v)


def fit_kmeans(X: np.ndarray, k: int = 30) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X)


def percentile_from_bottom(sims: np.ndarray) -> np.ndarray:
    """0 = least similar item, 100 = most similar item."""
    ranks = np.empty(len(sims), dtype=int)
    ranks[np.argsort(sims)] = np.arange(len(sims))
    return ranks / (len(sims) - 1) * 100


def close_match(sims: np.ndarray) -> int:
    return int(np.argmax(sims))


def band_sample(sims, lo=10.0, hi=30.0, floor=5.0, n=2, seed=None) -> list[int]:
    """Strategy A: sample n items whose similarity percentile (from the bottom) is in [lo, hi].
    `floor` excludes the very bottom (usually odd outliers); lo is raised to floor if lower."""
    lo = max(lo, floor)
    pct = percentile_from_bottom(sims)
    pool = np.where((pct >= lo) & (pct <= hi))[0]
    pool = pool[pool != close_match(sims)]
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(pool, size=n, replace=False)]


def farthest_cluster_sample(X, seed_vec, km: KMeans, n=2, seed=None) -> tuple[list[int], int]:
    """Strategy B: sample n items from the cluster whose centroid is least similar to the seed."""
    cents = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    far = int(np.argmin(cents @ seed_vec))
    members = np.where(km.labels_ == far)[0]
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(members, size=min(n, len(members)), replace=False)], far


def random_sample(n_items, exclude: int, n=2, seed=None) -> list[int]:
    """Strategy C: uniform random baseline."""
    rng = np.random.default_rng(seed)
    pool = np.array([i for i in range(n_items) if i != exclude])
    return [int(i) for i in rng.choice(pool, size=n, replace=False)]
