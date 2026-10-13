"""nomic-embed-text wrapper that adds the task prefixes the model was trained with.

The model card (huggingface.co/nomic-ai/nomic-embed-text-v1.5) says every input must
start with a task instruction. Stored items use "search_document: ", queries use
"search_query: ". LangChain calls embed_documents() when indexing and embed_query()
when searching, so we add the right prefix in each.
"""
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from .config import EMBED_MODEL, OLLAMA_URL

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class NomicEmbeddings(Embeddings):
    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_URL):
        self._inner = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents([DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(QUERY_PREFIX + text)
