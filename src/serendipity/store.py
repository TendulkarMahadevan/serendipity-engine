"""Stage 2: the persisted Chroma vector store, built idempotently."""
import time

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_DIR, COLLECTION
from .embeddings import NomicEmbeddings


def get_store() -> Chroma:
    """Open (or create) the persisted collection. Cosine distance suits unit-length vectors."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=NomicEmbeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def add_missing(store: Chroma, docs: list[Document], batch_size: int = 100) -> dict:
    """Embed and store only documents whose object_id isn't in the store yet."""
    ids = [str(d.metadata["object_id"]) for d in docs]
    existing = set(store.get(ids=ids, include=[])["ids"])
    todo = [(i, d) for i, d in zip(ids, docs) if i not in existing]
    t0 = time.time()
    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        store.add_documents([d for _, d in chunk], ids=[i for i, _ in chunk])
        print(f"  embedded {min(start + batch_size, len(todo))}/{len(todo)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return {"already_stored": len(existing), "newly_added": len(todo),
            "seconds": round(time.time() - t0, 1)}
